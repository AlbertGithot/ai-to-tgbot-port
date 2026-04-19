from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATES_ROOT = PROJECT_ROOT / "site_dashboard_templates"
STATIC_ROOT = PROJECT_ROOT / "site_dashboard_static"
RUNTIME_ROOT = PROJECT_ROOT / "web_panel_runtime"
DEFAULT_STATE_PATH = RUNTIME_ROOT / "panel_state.json"
DEFAULT_HOST = os.getenv("SITE_DASHBOARD_HOST", "127.0.0.1").strip() or "127.0.0.1"
DEFAULT_PORT = int(os.getenv("SITE_DASHBOARD_PORT", "5080").strip() or "5080")
REFRESH_INTERVAL_SECONDS = max(2, int(os.getenv("SITE_DASHBOARD_REFRESH_SECONDS", "4") or "4"))
ACCESS_CODE_TOKEN_BYTES = 12
MIN_ACCESS_CODE_LENGTH = 12
LOGIN_FAILURE_WINDOW_SECONDS = 15 * 60
LOGIN_FAILURE_LIMIT = 8
LOGIN_LOCKOUT_SECONDS = 5 * 60
SITE_PROCESS_KEYWORDS = (
    "site_dashboard.py",
    "bot.py",
    "launcher_cli.py",
    "llama-server",
)


def iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def hash_access_code(access_code: str) -> str:
    return hashlib.sha256(access_code.encode("utf-8")).hexdigest()


def generate_access_code() -> str:
    return secrets.token_urlsafe(ACCESS_CODE_TOKEN_BYTES)


def default_panel_state(access_code: str | None = None) -> tuple[dict[str, Any], str]:
    generated_code = access_code or generate_access_code()
    state = {
        "site_enabled": True,
        "access_code_hash": hash_access_code(generated_code),
        "session_secret": secrets.token_hex(32),
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "access_code_changed_at": iso_now(),
    }
    return state, generated_code


def ensure_panel_state(state_path: Path) -> tuple[dict[str, Any], str | None]:
    existing = read_json_file(state_path)
    if existing:
        state = {
            "site_enabled": bool(existing.get("site_enabled", True)),
            "access_code_hash": str(existing.get("access_code_hash") or ""),
            "session_secret": str(existing.get("session_secret") or ""),
            "created_at": str(existing.get("created_at") or iso_now()),
            "updated_at": str(existing.get("updated_at") or iso_now()),
            "access_code_changed_at": str(existing.get("access_code_changed_at") or existing.get("updated_at") or iso_now()),
        }
        updated = False
        if not state["access_code_hash"]:
            replacement, generated_code = default_panel_state()
            state["access_code_hash"] = replacement["access_code_hash"]
            updated = True
        else:
            generated_code = None
        if not state["session_secret"]:
            state["session_secret"] = secrets.token_hex(32)
            updated = True
        if updated:
            state["updated_at"] = iso_now()
            atomic_write_json(state_path, state)
        return state, generated_code

    state, generated_code = default_panel_state()
    atomic_write_json(state_path, state)
    return state, generated_code


def load_panel_state(state_path: Path) -> dict[str, Any]:
    state, _ = ensure_panel_state(state_path)
    return state


def save_panel_state(state_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    state["updated_at"] = iso_now()
    atomic_write_json(state_path, state)
    return state


def is_panel_authenticated() -> bool:
    return bool(session.get("panel_authenticated"))


def verify_access_code(state: dict[str, Any], access_code: str) -> bool:
    return bool(access_code) and hash_access_code(access_code.strip()) == str(state.get("access_code_hash") or "")


def get_client_identity() -> str:
    forwarded_for = str(request.headers.get("X-Forwarded-For") or "").strip()
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip() or "unknown"
    return str(request.remote_addr or "unknown").strip() or "unknown"


def prune_login_failures(store: dict[str, dict[str, float]], now: float | None = None) -> None:
    current_time = now if now is not None else time.time()
    stale_keys = [
        client_id
        for client_id, payload in store.items()
        if current_time - float(payload.get("updated_at") or 0.0) > LOGIN_FAILURE_WINDOW_SECONDS
    ]
    for client_id in stale_keys:
        store.pop(client_id, None)


def register_failed_login(store: dict[str, dict[str, float]], client_id: str, now: float | None = None) -> None:
    current_time = now if now is not None else time.time()
    prune_login_failures(store, current_time)
    payload = store.get(client_id, {"count": 0.0, "updated_at": current_time, "blocked_until": 0.0})
    count = int(payload.get("count") or 0) + 1
    blocked_until = float(payload.get("blocked_until") or 0.0)
    if count >= LOGIN_FAILURE_LIMIT:
        blocked_until = current_time + LOGIN_LOCKOUT_SECONDS
    store[client_id] = {
        "count": float(count),
        "updated_at": current_time,
        "blocked_until": blocked_until,
    }


def clear_failed_login(store: dict[str, dict[str, float]], client_id: str) -> None:
    store.pop(client_id, None)


def get_login_block_remaining_seconds(
    store: dict[str, dict[str, float]],
    client_id: str,
    now: float | None = None,
) -> int:
    current_time = now if now is not None else time.time()
    prune_login_failures(store, current_time)
    payload = store.get(client_id) or {}
    blocked_until = float(payload.get("blocked_until") or 0.0)
    if blocked_until <= current_time:
        return 0
    return int(blocked_until - current_time + 0.999)


def parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def sort_payloads_by_timestamp(payloads: list[dict[str, Any]], key_names: tuple[str, ...]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[float, str]:
        for key_name in key_names:
            parsed = parse_iso_datetime(item.get(key_name))
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
                return (parsed.timestamp(), str(item.get(key_name) or ""))
        return (0.0, "")

    return sorted(payloads, key=sort_key, reverse=True)


def summarize_text(text: str, max_chars: int = 240) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rsplit(" ", 1)[0].rstrip(".,;:-") + "…"


def format_duration(seconds: Any) -> str:
    try:
        total = max(0, int(seconds))
    except Exception:
        return "--:--:--"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def collect_relevant_processes(project_root: Path) -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,etime=,pcpu=,pmem=,command="],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
    except Exception:
        return []

    if completed.returncode != 0:
        return []

    project_marker = str(project_root)
    processes: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, elapsed, cpu, memory, command = parts
        lowered = command.lower()
        if project_marker not in command and not any(keyword in lowered for keyword in SITE_PROCESS_KEYWORDS):
            continue
        processes.append(
            {
                "pid": int(pid) if pid.isdigit() else pid,
                "elapsed": elapsed,
                "cpu_percent": cpu,
                "memory_percent": memory,
                "command": command,
            }
        )
    return processes[:40]


def list_long_think_jobs(project_root: Path) -> list[dict[str, Any]]:
    jobs_root = project_root / "deep_think_jobs"
    if not jobs_root.is_dir():
        return []

    jobs: list[dict[str, Any]] = []
    for state_path in jobs_root.glob("*/state.json"):
        state_payload = read_json_file(state_path)
        if not state_payload:
            continue
        result_path = state_path.with_name("result.json")
        result_payload = read_json_file(result_path) if result_path.is_file() else {}
        payload = result_payload or state_payload
        final_answer = str(payload.get("final_answer") or "")
        progress_percent = int(payload.get("progress_percent") or 0)
        jobs.append(
            {
                "job_id": str(payload.get("job_id") or state_path.parent.name),
                "request_text": str(payload.get("request_text") or "").strip(),
                "status": str(payload.get("status") or "unknown"),
                "phase": str(payload.get("phase") or "-"),
                "progress_percent": progress_percent,
                "progress_banner": str(payload.get("progress_banner") or ""),
                "planned_iterations": int(payload.get("planned_iterations") or 0),
                "completed_iterations": int(payload.get("completed_iterations") or 0),
                "remaining": format_duration(payload.get("remaining_seconds")),
                "elapsed": format_duration(payload.get("elapsed_seconds")),
                "updated_at": str(payload.get("updated_at") or payload.get("completed_at") or ""),
                "result_path": str(payload.get("result_path") or result_path),
                "artifact_dir": str(payload.get("artifact_dir") or state_path.parent),
                "has_result_file": result_path.is_file(),
                "answer_completed_fully": bool(payload.get("answer_completed_fully")),
                "character_count": len(final_answer),
                "answer_preview": summarize_text(final_answer, max_chars=280),
                "error": str(payload.get("error") or ""),
            }
        )
    return sort_payloads_by_timestamp(jobs, ("updated_at",))


def list_generated_result_files(project_root: Path) -> list[dict[str, Any]]:
    jobs_root = project_root / "deep_think_jobs"
    if not jobs_root.is_dir():
        return []

    files: list[dict[str, Any]] = []
    for result_path in jobs_root.glob("*/result.json"):
        payload = read_json_file(result_path)
        if not payload:
            continue
        final_answer = str(payload.get("final_answer") or "")
        files.append(
            {
                "job_id": str(payload.get("job_id") or result_path.parent.name),
                "request_text": str(payload.get("request_text") or ""),
                "status": str(payload.get("status") or "unknown"),
                "updated_at": str(payload.get("updated_at") or payload.get("completed_at") or ""),
                "path": str(result_path),
                "character_count": len(final_answer),
                "answer_completed_fully": bool(payload.get("answer_completed_fully")),
                "preview": summarize_text(final_answer, max_chars=240),
            }
        )
    return sort_payloads_by_timestamp(files, ("updated_at",))


def list_background_tasks(project_root: Path) -> list[dict[str, Any]]:
    tasks_root = project_root / "task_queue"
    if not tasks_root.is_dir():
        return []
    tasks: list[dict[str, Any]] = []
    for task_path in tasks_root.glob("task_*.json"):
        payload = read_json_file(task_path)
        if not payload:
            continue
        tasks.append(
            {
                "task_id": str(payload.get("task_id") or task_path.stem.replace("task_", "")),
                "kind": str(payload.get("kind") or ""),
                "status": str(payload.get("status") or ""),
                "description": str(payload.get("description") or ""),
                "progress_current": int(payload.get("progress_current") or 0),
                "progress_total": int(payload.get("progress_total") or 0),
                "progress_label": str(payload.get("progress_label") or ""),
                "updated_at": str(payload.get("updated_at") or payload.get("completed_at") or ""),
                "error": str(payload.get("error") or ""),
            }
        )
    return sort_payloads_by_timestamp(tasks, ("updated_at",))


def build_overview_payload(project_root: Path) -> dict[str, Any]:
    jobs = list_long_think_jobs(project_root)
    result_files = list_generated_result_files(project_root)
    tasks = list_background_tasks(project_root)
    processes = collect_relevant_processes(project_root)
    active_job_count = sum(1 for job in jobs if job["status"] in {"queued", "running"})
    running_task_count = sum(1 for task in tasks if task["status"] in {"queued", "running"})
    return {
        "generated_at": iso_now(),
        "refresh_interval_seconds": REFRESH_INTERVAL_SECONDS,
        "stats": {
            "active_job_count": active_job_count,
            "result_file_count": len(result_files),
            "running_task_count": running_task_count,
            "process_count": len(processes),
        },
        "processes": processes,
        "long_think_jobs": jobs[:16],
        "result_files": result_files[:20],
        "background_tasks": tasks[:20],
    }


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if is_panel_authenticated():
            return view(*args, **kwargs)
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "auth_required"}), 401
        flash("Сначала введи код доступа, а потом уже командуй сайтом.", "error")
        return redirect(url_for("index"))

    return wrapped


def create_app(project_root: Path | None = None, state_path: Path | None = None) -> Flask:
    root = (project_root or PROJECT_ROOT).resolve()
    panel_state_path = state_path or (root / "web_panel_runtime" / "panel_state.json")
    panel_state, generated_code = ensure_panel_state(panel_state_path)
    login_failures: dict[str, dict[str, float]] = {}

    app = Flask(
        __name__,
        template_folder=str(root / "site_dashboard_templates"),
        static_folder=str(root / "site_dashboard_static"),
        static_url_path="/panel-static",
    )
    app.config["PROJECT_ROOT"] = root
    app.config["PANEL_STATE_PATH"] = panel_state_path
    app.config["REFRESH_INTERVAL_SECONDS"] = REFRESH_INTERVAL_SECONDS
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_NAME"] = "heymate_panel"
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)
    app.secret_key = panel_state["session_secret"]

    def require_enabled_site():
        state = load_panel_state(app.config["PANEL_STATE_PATH"])
        if bool(state.get("site_enabled", True)):
            return None
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "site_disabled"}), 403
        flash("Панель сейчас выключена. Включи её сначала в manage-разделе.", "error")
        return redirect(url_for("manage"))

    @app.context_processor
    def inject_global_state() -> dict[str, Any]:
        state = load_panel_state(app.config["PANEL_STATE_PATH"])
        return {
            "site_enabled": bool(state.get("site_enabled", True)),
            "panel_authenticated": is_panel_authenticated(),
        }

    @app.get("/")
    def index():
        state = load_panel_state(app.config["PANEL_STATE_PATH"])
        return render_template(
            "site_dashboard_index.html",
            site_enabled=bool(state.get("site_enabled", True)),
            panel_authenticated=is_panel_authenticated(),
        )

    @app.post("/login")
    def login():
        state = load_panel_state(app.config["PANEL_STATE_PATH"])
        client_id = get_client_identity()
        lock_remaining = get_login_block_remaining_seconds(login_failures, client_id)
        if lock_remaining > 0:
            flash(
                f"Слишком много неудачных попыток входа. Подожди {lock_remaining} сек. и попробуй ещё раз.",
                "error",
            )
            return redirect(url_for("index"))
        access_code = request.form.get("access_code", "").strip()
        if verify_access_code(state, access_code):
            clear_failed_login(login_failures, client_id)
            session["panel_authenticated"] = True
            session.permanent = True
            flash("Код доступа принят. Теперь можно крутить эту панель как хочешь.", "success")
            target = request.form.get("next") or url_for("dashboard")
            return redirect(target)
        register_failed_login(login_failures, client_id)
        flash("Код доступа мимо кассы. Проверь, что вводишь, а не набор боли.", "error")
        return redirect(url_for("index"))

    @app.post("/logout")
    def logout():
        session.clear()
        flash("Сессию закрыл. Панель снова под замком.", "info")
        return redirect(url_for("index"))

    @app.route("/manage", methods=["GET", "POST"])
    @login_required
    def manage():
        state = load_panel_state(app.config["PANEL_STATE_PATH"])
        if request.method == "POST":
            action = request.form.get("action", "").strip()
            if action == "toggle-site":
                state["site_enabled"] = not bool(state.get("site_enabled", True))
                save_panel_state(app.config["PANEL_STATE_PATH"], state)
                flash(
                    "Сайт включил. Теперь панель снова торчит наружу."
                    if state["site_enabled"]
                    else "Сайт выключил. Процесс жив, но сама панель ушла в закрытый режим.",
                    "success",
                )
                return redirect(url_for("manage"))
            if action == "change-access-code":
                new_code = request.form.get("new_access_code", "").strip()
                confirm_code = request.form.get("confirm_access_code", "").strip()
                if len(new_code) < MIN_ACCESS_CODE_LENGTH:
                    flash(
                        f"Код доступа слишком короткий. Нужен минимум {MIN_ACCESS_CODE_LENGTH} символов.",
                        "error",
                    )
                elif new_code != confirm_code:
                    flash("Подтверждение не совпало. Даже два поля между собой не договорились.", "error")
                else:
                    state["access_code_hash"] = hash_access_code(new_code)
                    state["access_code_changed_at"] = iso_now()
                    save_panel_state(app.config["PANEL_STATE_PATH"], state)
                    flash("Код доступа обновил. Старый теперь бесполезен, как обещания кривого VPN.", "success")
                return redirect(url_for("manage"))
            flash("Неизвестное действие. Панель не умеет читать мысли, только формы.", "error")
            return redirect(url_for("manage"))

        return render_template(
            "site_dashboard_manage.html",
            site_enabled=bool(state.get("site_enabled", True)),
            access_code_changed_at=str(state.get("access_code_changed_at") or "-"),
        )

    @app.get("/dashboard")
    @login_required
    def dashboard():
        blocked = require_enabled_site()
        if blocked is not None:
            return blocked
        state = load_panel_state(app.config["PANEL_STATE_PATH"])
        overview = build_overview_payload(app.config["PROJECT_ROOT"])
        return render_template(
            "site_dashboard_dashboard.html",
            site_enabled=bool(state.get("site_enabled", True)),
            overview=overview,
            refresh_interval_seconds=app.config["REFRESH_INTERVAL_SECONDS"],
        )

    @app.get("/api/overview")
    @login_required
    def api_overview():
        blocked = require_enabled_site()
        if blocked is not None:
            return blocked
        state = load_panel_state(app.config["PANEL_STATE_PATH"])
        payload = build_overview_payload(app.config["PROJECT_ROOT"])
        payload["site_enabled"] = bool(state.get("site_enabled", True))
        payload["ok"] = True
        return jsonify(payload)

    @app.get("/api/file/<job_id>")
    @login_required
    def api_file(job_id: str):
        blocked = require_enabled_site()
        if blocked is not None:
            return blocked
        safe_job_id = str(job_id or "").strip().lower()
        if not safe_job_id:
            abort(404)
        jobs_root = app.config["PROJECT_ROOT"] / "deep_think_jobs"
        for result_path in jobs_root.glob("*/result.json"):
            payload = read_json_file(result_path)
            if str(payload.get("job_id") or "").strip().lower() == safe_job_id:
                return jsonify(
                    {
                        "ok": True,
                        "job_id": payload.get("job_id"),
                        "path": str(result_path),
                        "payload": payload,
                    }
                )
        abort(404)

    if generated_code:
        print(
            f"[site_dashboard] Первый запуск. Код доступа создан: {generated_code}\n"
            f"[site_dashboard] Состояние панели лежит тут: {panel_state_path}",
            flush=True,
        )

    return app


app = create_app()


def main() -> int:
    print(
        f"[site_dashboard] Запускаю панель на http://{DEFAULT_HOST}:{DEFAULT_PORT}\n"
        f"[site_dashboard] Runtime: {DEFAULT_STATE_PATH}",
        flush=True,
    )
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
