from __future__ import annotations

import hashlib
import ipaddress
import json
import mimetypes
import os
import secrets
import shutil
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
    send_file,
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
DEFAULT_HOST = os.getenv("SITE_DASHBOARD_HOST", "0.0.0.0").strip() or "0.0.0.0"
DEFAULT_PORT = int(os.getenv("SITE_DASHBOARD_PORT", "5080").strip() or "5080")
REFRESH_INTERVAL_SECONDS = max(2, int(os.getenv("SITE_DASHBOARD_REFRESH_SECONDS", "4") or "4"))
ACCESS_CODE_TOKEN_BYTES = 12
MIN_ACCESS_CODE_LENGTH = 12
DEFAULT_LOGIN_WINDOW_SECONDS = max(30, int(os.getenv("SITE_DASHBOARD_LOGIN_WINDOW_SECONDS", "600") or "600"))
DEFAULT_LOGIN_MAX_ATTEMPTS = max(2, int(os.getenv("SITE_DASHBOARD_LOGIN_MAX_ATTEMPTS", "8") or "8"))
RESOURCE_HISTORY_MAX_POINTS = max(30, int(os.getenv("SITE_DASHBOARD_RESOURCE_HISTORY_MAX_POINTS", "90") or "90"))
LOG_TAIL_CHARS = max(2000, int(os.getenv("SITE_DASHBOARD_LOG_TAIL_CHARS", "18000") or "18000"))
ARTIFACT_PREVIEW_CHARS = 16000
PANEL_STATE_ALLOWED_ENDPOINTS_WHEN_DISABLED = {
    "index",
    "login",
    "logout",
    "manage",
    "static",
}
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
        "ip_whitelist_text": "",
        "login_rate_limit_window_seconds": DEFAULT_LOGIN_WINDOW_SECONDS,
        "login_rate_limit_max_attempts": DEFAULT_LOGIN_MAX_ATTEMPTS,
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
            "ip_whitelist_text": str(existing.get("ip_whitelist_text") or ""),
            "login_rate_limit_window_seconds": max(
                30,
                int(existing.get("login_rate_limit_window_seconds") or DEFAULT_LOGIN_WINDOW_SECONDS),
            ),
            "login_rate_limit_max_attempts": max(
                2,
                int(existing.get("login_rate_limit_max_attempts") or DEFAULT_LOGIN_MAX_ATTEMPTS),
            ),
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


def runtime_root_for_project(project_root: Path) -> Path:
    return project_root / "web_panel_runtime"


def login_guard_state_path(project_root: Path) -> Path:
    return runtime_root_for_project(project_root) / "login_guard_state.json"


def resource_history_state_path(project_root: Path) -> Path:
    return runtime_root_for_project(project_root) / "resource_history.json"


def read_text_tail(path: Path, max_chars: int = LOG_TAIL_CHARS) -> str:
    if not path.is_file():
        return "Лог пока не найден."
    return path.read_text(encoding="utf-8", errors="ignore")[-max_chars:].strip() or "Лог пуст."


def normalize_ip_whitelist_text(value: str) -> str:
    lines = [line.strip() for line in str(value or "").replace(",", "\n").splitlines()]
    normalized: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if not line or line in seen:
            continue
        seen.add(line)
        normalized.append(line)
    return "\n".join(normalized)


def parse_ip_whitelist_entries(raw_value: str) -> list[str]:
    parsed: list[str] = []
    for line in normalize_ip_whitelist_text(raw_value).splitlines():
        try:
            if "/" in line:
                parsed.append(str(ipaddress.ip_network(line, strict=False)))
            else:
                parsed.append(str(ipaddress.ip_address(line)))
        except ValueError:
            continue
    return parsed


def get_request_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return (request.remote_addr or "").strip()


def is_ip_allowed(state: dict[str, Any], ip_value: str) -> bool:
    whitelist_entries = parse_ip_whitelist_entries(str(state.get("ip_whitelist_text") or ""))
    if not whitelist_entries:
        return True
    try:
        ip_obj = ipaddress.ip_address(ip_value)
    except ValueError:
        return False
    for entry in whitelist_entries:
        if "/" in entry:
            if ip_obj in ipaddress.ip_network(entry, strict=False):
                return True
        elif ip_obj == ipaddress.ip_address(entry):
            return True
    return False


def load_login_guard_state(project_root: Path) -> dict[str, Any]:
    return read_json_file(login_guard_state_path(project_root))


def save_login_guard_state(project_root: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(login_guard_state_path(project_root), payload)


def prune_login_guard_state(payload: dict[str, Any], *, now_ts: float, window_seconds: int) -> dict[str, Any]:
    attempts = payload.get("attempts")
    if not isinstance(attempts, dict):
        attempts = {}
    cleaned: dict[str, list[float]] = {}
    for ip_value, stamps in attempts.items():
        if not isinstance(stamps, list):
            continue
        recent = [float(stamp) for stamp in stamps if now_ts - float(stamp) <= window_seconds]
        if recent:
            cleaned[str(ip_value)] = recent[-64:]
    return {"attempts": cleaned}


def is_login_rate_limited(project_root: Path, state: dict[str, Any], ip_value: str) -> tuple[bool, int]:
    window_seconds = max(30, int(state.get("login_rate_limit_window_seconds") or DEFAULT_LOGIN_WINDOW_SECONDS))
    max_attempts = max(2, int(state.get("login_rate_limit_max_attempts") or DEFAULT_LOGIN_MAX_ATTEMPTS))
    now_ts = time.time()
    payload = prune_login_guard_state(load_login_guard_state(project_root), now_ts=now_ts, window_seconds=window_seconds)
    save_login_guard_state(project_root, payload)
    attempts = list(payload.get("attempts", {}).get(ip_value, []))
    limited = len(attempts) >= max_attempts
    return limited, max_attempts


def record_failed_login_attempt(project_root: Path, state: dict[str, Any], ip_value: str) -> tuple[int, int]:
    window_seconds = max(30, int(state.get("login_rate_limit_window_seconds") or DEFAULT_LOGIN_WINDOW_SECONDS))
    max_attempts = max(2, int(state.get("login_rate_limit_max_attempts") or DEFAULT_LOGIN_MAX_ATTEMPTS))
    now_ts = time.time()
    payload = prune_login_guard_state(load_login_guard_state(project_root), now_ts=now_ts, window_seconds=window_seconds)
    attempts = list(payload.get("attempts", {}).get(ip_value, []))
    attempts.append(now_ts)
    payload.setdefault("attempts", {})[ip_value] = attempts[-64:]
    save_login_guard_state(project_root, payload)
    remaining = max(0, max_attempts - len(payload["attempts"][ip_value]))
    return remaining, window_seconds


def clear_failed_login_attempts(project_root: Path, ip_value: str) -> None:
    payload = load_login_guard_state(project_root)
    attempts = payload.get("attempts")
    if isinstance(attempts, dict) and ip_value in attempts:
        attempts.pop(ip_value, None)
        save_login_guard_state(project_root, {"attempts": attempts})


def collect_panel_log_payload(project_root: Path) -> list[dict[str, Any]]:
    runtime_root = runtime_root_for_project(project_root)
    sources = [
        ("runtime.log", project_root / "bot_logs" / "runtime.log"),
        ("systemd_supervisor.log", project_root / "bot_logs" / "systemd_supervisor.log"),
        ("llama_server.log", project_root / "bot_logs" / "llama_server.log"),
        ("site_dashboard.log", runtime_root / "site_dashboard.log"),
    ]
    payload: list[dict[str, Any]] = []
    for label, path in sources:
        payload.append(
            {
                "name": label,
                "path": str(path),
                "exists": path.is_file(),
                "tail": read_text_tail(path),
            }
        )
    return payload


def collect_artifact_files(project_root: Path, *, limit: int = 120) -> list[dict[str, Any]]:
    roots = [
        project_root / "deep_think_jobs",
        project_root / "terminal_sessions" / "exports",
        project_root / "backups",
    ]
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(root.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            marker = str(resolved)
            if marker in seen:
                continue
            seen.add(marker)
            try:
                relative_path = resolved.relative_to(project_root)
            except Exception:
                relative_path = resolved
            suffix = resolved.suffix.lower()
            mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
            try:
                stat = resolved.stat()
                size_bytes = int(stat.st_size)
                updated_at = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat()
            except OSError:
                size_bytes = 0
                updated_at = ""
            files.append(
                {
                    "relative_path": str(relative_path),
                    "path": str(resolved),
                    "name": resolved.name,
                    "directory": str(relative_path.parent) if hasattr(relative_path, "parent") else "",
                    "suffix": suffix,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "updated_at": updated_at,
                    "is_previewable": suffix in {".json", ".md", ".txt", ".log", ".yml", ".yaml", ".ini", ".cfg"},
                }
            )
            if len(files) >= limit:
                return sort_payloads_by_timestamp(files, ("updated_at",))
    return sort_payloads_by_timestamp(files, ("updated_at",))


def resolve_artifact_path(project_root: Path, relative_path: str) -> Path | None:
    normalized = str(relative_path or "").strip().lstrip("/").replace("\\", "/")
    if not normalized:
        return None
    candidate = (project_root / normalized).resolve()
    allowed_roots = [
        (project_root / "deep_think_jobs").resolve(),
        (project_root / "terminal_sessions" / "exports").resolve(),
        (project_root / "backups").resolve(),
    ]
    if not candidate.is_file():
        return None
    if not any(str(candidate).startswith(str(root) + os.sep) or candidate == root for root in allowed_roots if root.exists()):
        return None
    return candidate


def preview_artifact_file(project_root: Path, relative_path: str) -> dict[str, Any] | None:
    artifact_path = resolve_artifact_path(project_root, relative_path)
    if artifact_path is None:
        return None
    suffix = artifact_path.suffix.lower()
    if suffix == ".json":
        payload = read_json_file(artifact_path)
        if payload:
            preview_text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            preview_text = artifact_path.read_text(encoding="utf-8", errors="ignore")
    else:
        preview_text = artifact_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "relative_path": relative_path,
        "path": str(artifact_path),
        "preview_text": preview_text[:ARTIFACT_PREVIEW_CHARS],
        "truncated": len(preview_text) > ARTIFACT_PREVIEW_CHARS,
    }


def read_proc_cpu_snapshot() -> tuple[int, int] | None:
    proc_stat_path = Path("/proc/stat")
    if not proc_stat_path.is_file():
        return None
    try:
        for raw_line in proc_stat_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not raw_line.startswith("cpu "):
                continue
            parts = raw_line.split()
            values = [int(value) for value in parts[1:8]]
            idle = values[3] + values[4]
            total = sum(values)
            return total, idle
    except Exception:
        return None
    return None


def compute_cpu_percent(previous: tuple[int, int] | None, current: tuple[int, int] | None) -> float | None:
    if previous is None or current is None:
        return None
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    return round(max(0.0, min(100.0, (1.0 - (idle_delta / total_delta)) * 100.0)), 2)


def read_ram_percent() -> float | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.is_file():
        return None
    totals: dict[str, int] = {}
    try:
        for raw_line in meminfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            parts = value.strip().split()
            if not parts or not parts[0].isdigit():
                continue
            totals[key.strip()] = int(parts[0])
    except Exception:
        return None
    total_kib = totals.get("MemTotal")
    available_kib = totals.get("MemAvailable")
    if not total_kib or not available_kib:
        return None
    used_ratio = 1.0 - (available_kib / total_kib)
    return round(max(0.0, min(100.0, used_ratio * 100.0)), 2)


def read_gpu_percent() -> float | None:
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            completed = subprocess.run(
                [executable, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                values = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
                if values:
                    numeric = [float(value) for value in values if value.replace(".", "", 1).isdigit()]
                    if numeric:
                        return round(sum(numeric) / len(numeric), 2)
        except Exception:
            return None
    return None


def update_resource_history(project_root: Path) -> dict[str, Any]:
    history_path = resource_history_state_path(project_root)
    payload = read_json_file(history_path)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        samples = []
    previous_cpu_total = payload.get("previous_cpu_total")
    previous_cpu_idle = payload.get("previous_cpu_idle")
    previous_cpu_snapshot = None
    if isinstance(previous_cpu_total, int) and isinstance(previous_cpu_idle, int):
        previous_cpu_snapshot = (previous_cpu_total, previous_cpu_idle)
    current_cpu_snapshot = read_proc_cpu_snapshot()
    sample = {
        "timestamp": iso_now(),
        "cpu_percent": compute_cpu_percent(previous_cpu_snapshot, current_cpu_snapshot),
        "ram_percent": read_ram_percent(),
        "gpu_percent": read_gpu_percent(),
    }
    samples.append(sample)
    samples = samples[-RESOURCE_HISTORY_MAX_POINTS:]
    new_payload = {
        "previous_cpu_total": current_cpu_snapshot[0] if current_cpu_snapshot is not None else previous_cpu_total,
        "previous_cpu_idle": current_cpu_snapshot[1] if current_cpu_snapshot is not None else previous_cpu_idle,
        "samples": samples,
    }
    atomic_write_json(history_path, new_payload)
    return {
        "current": sample,
        "history": samples,
    }


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
    artifacts = collect_artifact_files(project_root)
    resources = update_resource_history(project_root)
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
        "artifacts": artifacts[:80],
        "logs": collect_panel_log_payload(project_root),
        "system_resources": resources["current"],
        "resource_history": resources["history"],
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
            return jsonify({"ok": False, "error": "site_disabled"}), 423
        flash("Панель сейчас выключена. Включи её в управлении сайтом, и потом уже командуй.", "error")
        return redirect(url_for("manage" if is_panel_authenticated() else "index"))

    @app.before_request
    def enforce_panel_restrictions():
        state = load_panel_state(app.config["PANEL_STATE_PATH"])
        request_ip = get_request_ip()
        if request.endpoint == "static":
            return None
        if not is_ip_allowed(state, request_ip):
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "ip_forbidden"}), 403
            flash(f"IP {request_ip or '-'} не входит в whitelist панели.", "error")
            return redirect(url_for("index"))
        if not bool(state.get("site_enabled", True)) and request.endpoint not in PANEL_STATE_ALLOWED_ENDPOINTS_WHEN_DISABLED:
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "site_disabled"}), 423
            flash("Панель сейчас выключена. Включи её в управлении сайтом, и потом уже командуй.", "error")
            return redirect(url_for("manage" if is_panel_authenticated() else "index"))
        return None

    @app.context_processor
    def inject_global_state() -> dict[str, Any]:
        state = load_panel_state(app.config["PANEL_STATE_PATH"])
        return {
            "site_enabled": bool(state.get("site_enabled", True)),
            "panel_authenticated": is_panel_authenticated(),
            "panel_ip_whitelist_text": str(state.get("ip_whitelist_text") or ""),
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
        project_root = app.config["PROJECT_ROOT"]
        request_ip = get_request_ip()
        limited, max_attempts = is_login_rate_limited(project_root, state, request_ip)
        if limited:
            flash(
                f"Слишком много неудачных входов с IP {request_ip or '-'}. Подожди и попробуй снова.",
                "error",
            )
            return redirect(url_for("index"))
        access_code = request.form.get("access_code", "").strip()
        if verify_access_code(state, access_code):
            session["panel_authenticated"] = True
            session.permanent = True
            clear_failed_login_attempts(project_root, request_ip)
            flash("Код доступа принят. Теперь можно крутить эту панель как хочешь.", "success")
            target = request.form.get("next") or url_for("dashboard")
            return redirect(target)
        remaining, window_seconds = record_failed_login_attempt(project_root, state, request_ip)
        flash("Код доступа мимо кассы. Проверь, что вводишь, а не набор боли.", "error")
        flash(
            f"Осталось попыток в окне {window_seconds} сек.: {remaining}/{max_attempts}.",
            "info",
        )
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
            if action == "update-security":
                state["ip_whitelist_text"] = normalize_ip_whitelist_text(
                    request.form.get("ip_whitelist_text", "")
                )
                try:
                    state["login_rate_limit_window_seconds"] = max(
                        30,
                        int(request.form.get("login_rate_limit_window_seconds", DEFAULT_LOGIN_WINDOW_SECONDS)),
                    )
                except (TypeError, ValueError):
                    state["login_rate_limit_window_seconds"] = DEFAULT_LOGIN_WINDOW_SECONDS
                try:
                    state["login_rate_limit_max_attempts"] = max(
                        2,
                        int(request.form.get("login_rate_limit_max_attempts", DEFAULT_LOGIN_MAX_ATTEMPTS)),
                    )
                except (TypeError, ValueError):
                    state["login_rate_limit_max_attempts"] = DEFAULT_LOGIN_MAX_ATTEMPTS
                save_panel_state(app.config["PANEL_STATE_PATH"], state)
                flash("Настройки whitelist и rate-limit обновил.", "success")
                return redirect(url_for("manage"))
            flash("Неизвестное действие. Панель не умеет читать мысли, только формы.", "error")
            return redirect(url_for("manage"))

        return render_template(
            "site_dashboard_manage.html",
            site_enabled=bool(state.get("site_enabled", True)),
            access_code_changed_at=str(state.get("access_code_changed_at") or "-"),
            ip_whitelist_text=str(state.get("ip_whitelist_text") or ""),
            login_rate_limit_window_seconds=int(state.get("login_rate_limit_window_seconds") or DEFAULT_LOGIN_WINDOW_SECONDS),
            login_rate_limit_max_attempts=int(state.get("login_rate_limit_max_attempts") or DEFAULT_LOGIN_MAX_ATTEMPTS),
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

    @app.get("/api/logs")
    @login_required
    def api_logs():
        return jsonify({"ok": True, "logs": collect_panel_log_payload(app.config["PROJECT_ROOT"])})

    @app.get("/api/artifacts")
    @login_required
    def api_artifacts():
        return jsonify({"ok": True, "artifacts": collect_artifact_files(app.config["PROJECT_ROOT"])})

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

    @app.get("/api/artifact")
    @login_required
    def api_artifact():
        relative_path = request.args.get("path", "").strip()
        payload = preview_artifact_file(app.config["PROJECT_ROOT"], relative_path)
        if payload is None:
            abort(404)
        return jsonify({"ok": True, **payload})

    @app.get("/api/artifact-download")
    @login_required
    def api_artifact_download():
        relative_path = request.args.get("path", "").strip()
        artifact_path = resolve_artifact_path(app.config["PROJECT_ROOT"], relative_path)
        if artifact_path is None:
            abort(404)
        return send_file(artifact_path, as_attachment=True, download_name=artifact_path.name)

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
