from __future__ import annotations

import importlib.util
import ipaddress
import itertools
import json
import locale
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import urllib.parse
import urllib.request
import webbrowser
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import readline  # noqa: F401
except ImportError:
    readline = None


REPO_PAGE_URL = "https://github.com/AlbertGithot/ai-to-tgbot-port"
STATE_FILE_NAME = ".launcher_state.json"
ENV_FILE_NAME = ".env"
ENV_EXAMPLE_FILE_NAME = ".env.example"
RECOMMENDED_MODEL_REPO = "HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive"
RECOMMENDED_MODEL_FILE = "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf"
LLAMA_RELEASE_API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
BOT_ENTRYPOINT = "bot.py"
SITE_DASHBOARD_ENTRYPOINT = "site_dashboard.py"
UI_STEP_DELAY_SECONDS = 1.0
SYSTEMD_SERVICE_NAME = "heymate-bot.service"
SITE_DASHBOARD_SYSTEMD_SERVICE_NAME = "heymate-site-dashboard.service"
TERMINAL_SESSIONS_DIR_NAME = "terminal_sessions"
SITE_DASHBOARD_RUNTIME_DIR_NAME = "web_panel_runtime"
SITE_DASHBOARD_STATE_FILE_NAME = "panel_state.json"
SITE_DASHBOARD_PROCESS_FILE_NAME = "site_dashboard_process.json"
SITE_DASHBOARD_LOG_FILE_NAME = "site_dashboard.log"
SITE_DASHBOARD_DEFAULT_HOST = "127.0.0.1"
SITE_DASHBOARD_DEFAULT_PORT = 5080
SITE_DASHBOARD_DEFAULT_REFRESH_SECONDS = 4
SITE_DASHBOARD_REVERSE_PROXY_DIR_NAME = "reverse_proxy"
SITE_DASHBOARD_CADDY_FILE_NAME = "heymate-site-dashboard.Caddyfile"
SITE_DASHBOARD_NGINX_FILE_NAME = "heymate-site-dashboard.nginx.conf"
BACKUPS_DIR_NAME = "backups"
UPDATE_HISTORY_LIMIT = 12
MODEL_SCAN_COMMON_ROOT_LIMIT = 200
MODEL_SCAN_SYSTEM_ROOT_LIMIT = 800
MODEL_SCAN_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
}
MODEL_SCAN_SKIP_ABSOLUTE_PREFIXES = (
    "/proc",
    "/sys",
    "/dev",
    "/run",
)
GIT_REMOTE_NAME = "origin"
UPDATE_CHECK_TIMEOUT_SECONDS = 20
LIVE_STATUS_BAR_WIDTH = 22
LOG_PREVIEW_CHARS = 12000
live_banner_render_len = 0
bot_runtime_module_cache: Any | None = None
bot_runtime_module_error: str | None = None


def build_llama_runtime_env(llama_dir: Path, llama_server_exe: Path) -> dict[str, str]:
    env = os.environ.copy()
    library_dirs: list[str] = []
    for candidate in (
        llama_server_exe.parent,
        llama_server_exe.parent / "lib",
        llama_server_exe.parent.parent / "lib",
        llama_dir,
        llama_dir / "lib",
        llama_dir.parent / "lib",
    ):
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.is_dir():
            value = str(resolved)
            if value not in library_dirs:
                library_dirs.append(value)

    if library_dirs:
        existing = env.get("LD_LIBRARY_PATH", "").strip()
        env["LD_LIBRARY_PATH"] = ":".join([*library_dirs, existing] if existing else library_dirs)
    return env


def ensure_executable_permissions(path: Path) -> None:
    if os.name == "nt" or not path.is_file():
        return
    try:
        current_mode = path.stat().st_mode
        path.chmod(current_mode | 0o111)
    except Exception:
        return


def validate_llama_runtime(llama_dir: Path, llama_server_exe: Path) -> None:
    ensure_executable_permissions(llama_server_exe)
    try:
        completed = subprocess.run(
            [str(llama_server_exe), "--version"],
            cwd=str(llama_dir),
            env=build_llama_runtime_env(llama_dir, llama_server_exe),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=20,
        )
    except OSError as exc:
        raise RuntimeError(f"Не удалось запустить llama-server: {exc}") from exc
    except subprocess.TimeoutExpired:
        return

    if completed.returncode == 0:
        return

    detail = (completed.stderr or completed.stdout or "").strip()
    if detail:
        detail = detail.splitlines()[-1].strip()
    else:
        detail = f"exit code {completed.returncode}"

    if completed.returncode == 127:
        raise RuntimeError(
            "llama-server найден, но не запускается на этой Linux-системе. "
            f"Деталь: {detail}. Скорее всего не хватает системных библиотек "
            "или скачана несовместимая сборка."
        )
    raise RuntimeError(f"llama-server не прошёл проверку запуска: {detail}")


def ensure_utf8_output() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def cls() -> None:
    if not sys.stdout.isatty():
        return
    if not os.getenv("TERM"):
        return
    os.system("clear")


def print_block(text: str) -> None:
    print(textwrap.dedent(text).strip() + "\n", flush=True)
    time.sleep(UI_STEP_DELAY_SECONDS)


def render_live_banner(
    label: str,
    detail: str,
    *,
    progress: float | None = None,
    done: bool = False,
) -> str:
    title = "Все готово :3" if done else label
    body = f"{title}: {detail}" if detail else title
    width = LIVE_STATUS_BAR_WIDTH
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        try:
            columns = shutil.get_terminal_size((140, 24)).columns
        except OSError:
            columns = 140
        max_side_width = max(8, (columns - len(body) - 4) // 2)
        width = max(8, min(width, max_side_width))

    if done:
        filled = width
    elif progress is None:
        filled = 0
    else:
        clamped = max(0.0, min(1.0, float(progress)))
        filled = min(width, int(round(clamped * width)))
    padding = max(0, width - filled)
    left = ("-" * padding) + ("=" * filled)
    right = ("=" * filled) + ("-" * padding)
    return f"<{left}|{body}|{right}>"


def print_live_banner(
    label: str,
    detail: str,
    *,
    progress: float | None = None,
    done: bool = False,
    final_newline: bool = False,
) -> None:
    global live_banner_render_len

    text = render_live_banner(label, detail, progress=progress, done=done)
    live_banner_render_len = max(live_banner_render_len, len(text))
    sys.stdout.write("\r" + text.ljust(live_banner_render_len))
    if final_newline:
        sys.stdout.write("\n")
        live_banner_render_len = 0
    sys.stdout.flush()


def pause(prompt: str = "Нажми Enter, чтобы продолжить...") -> None:
    read_console_input(prompt)


def count_problem_console_glyphs(text: str) -> int:
    return sum(
        1
        for char in text
        if (
            0x2500 <= ord(char) <= 0x259F
            or char in {"�", "■", "□", "▪", "▫"}
        )
    )


def strip_terminal_control_sequences(text: str) -> str:
    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", cleaned)
    cleaned = re.sub(r"\x1b[@-Z\\-_]", "", cleaned)
    cleaned = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", cleaned)
    cleaned = cleaned.replace("\x1b[200~", "").replace("\x1b[201~", "")
    return cleaned


def count_strange_console_symbols(text: str) -> int:
    return sum(
        1
        for char in text
        if (
            char.isprintable()
            and not re.match(
                r"""[A-Za-zА-Яа-яЁё0-9\s.,!?/:;()\[\]{}'"@#%&*_+\-=<>]""",
                char,
            )
        )
    )


def looks_broken_console_text(text: str) -> bool:
    stripped = strip_terminal_control_sequences(text).strip()
    if not stripped:
        return False

    bad_glyphs = count_problem_console_glyphs(stripped)
    alpha_count = len(re.findall(r"[A-Za-zА-Яа-яЁё]", stripped))
    strange_symbol_count = count_strange_console_symbols(stripped)
    if bad_glyphs >= max(2, alpha_count // 2):
        return True
    if strange_symbol_count >= max(3, len(stripped) // 6) and strange_symbol_count >= max(2, alpha_count // 4):
        return True
    return False


def score_console_text_quality(text: str) -> tuple[int, int, int, int]:
    stripped = strip_terminal_control_sequences(text).strip()
    if not stripped:
        return (-10, 0, 0, 0)

    cyrillic_count = len(re.findall(r"[А-Яа-яЁё]", stripped))
    latin_count = len(re.findall(r"[A-Za-z]", stripped))
    digit_count = len(re.findall(r"\d", stripped))
    replacement_count = stripped.count("\ufffd")
    mojibake_fragments = (
        "Р°",
        "Рб",
        "Рв",
        "Рг",
        "Рґ",
        "Ре",
        "Рё",
        "Рй",
        "Рк",
        "Рл",
        "Рм",
        "Рн",
        "Ро",
        "Рп",
        "Рр",
        "Рс",
        "Рт",
        "Ру",
        "Рф",
        "Рх",
        "Рц",
        "Рч",
        "Рш",
        "Рщ",
        "Рэ",
        "Рю",
        "Ря",
        "С‚",
        "СЊ",
        "СЏ",
        "С‡",
        "Ñ",
        "Ð",
        "Ã",
        "Â",
    )
    mojibake_count = sum(stripped.count(fragment) for fragment in mojibake_fragments)
    problem_glyph_count = count_problem_console_glyphs(stripped)
    strange_symbol_count = count_strange_console_symbols(stripped)
    printable_count = sum(
        1 for char in stripped if char.isprintable() or char in "\r\n\t"
    )
    weird_count = max(0, len(stripped) - printable_count)
    return (
        (cyrillic_count * 6)
        + (latin_count * 2)
        + digit_count
        - (mojibake_count * 30)
        - (problem_glyph_count * 35)
        - (strange_symbol_count * 20)
        - (replacement_count * 40)
        - (weird_count * 20),
        cyrillic_count,
        -(mojibake_count + problem_glyph_count),
        -replacement_count,
    )


def iter_console_text_variants(text: str) -> list[str]:
    variants = [text]
    for source_encoding in ("cp1251", "cp866", "latin-1"):
        try:
            repaired = text.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired not in variants:
            variants.append(repaired)
    return variants


def choose_best_console_text(candidates: list[str]) -> str:
    best_text = ""
    best_score: tuple[int, int, int, int] | None = None
    seen: set[str] = set()

    for candidate in candidates:
        for variant in iter_console_text_variants(candidate):
            normalized = strip_terminal_control_sequences(variant).rstrip("\r\n")
            if normalized in seen:
                continue
            seen.add(normalized)
            score = score_console_text_quality(normalized)
            if best_score is None or score > best_score:
                best_text = normalized
                best_score = score

    return best_text


def decode_console_input_bytes(data: bytes) -> str:
    try:
        utf8_text = strip_terminal_control_sequences(data.decode("utf-8")).rstrip("\r\n")
    except UnicodeDecodeError:
        utf8_text = ""
    else:
        if utf8_text and not looks_broken_console_text(utf8_text):
            return utf8_text

    encodings: list[str] = []
    for encoding in (
        "utf-8",
        "utf-8-sig",
        sys.stdin.encoding,
        locale.getpreferredencoding(False),
        "cp866",
        "cp1251",
        "latin-1",
    ):
        normalized = str(encoding or "").strip()
        if normalized and normalized.lower() not in {item.lower() for item in encodings}:
            encodings.append(normalized)

    decoded_candidates: list[str] = []
    for encoding in encodings:
        try:
            decoded_candidates.append(data.decode(encoding))
        except UnicodeDecodeError:
            continue

    if decoded_candidates:
        best_text = choose_best_console_text(decoded_candidates)
        if best_text:
            return best_text

    fallback = encodings[0] if encodings else "utf-8"
    fallback_text = strip_terminal_control_sequences(
        data.decode(fallback, errors="replace")
    )
    repaired_fallback = choose_best_console_text([fallback_text])
    best_text = (repaired_fallback or fallback_text).rstrip("\r\n")
    if utf8_text:
        repaired_utf8 = choose_best_console_text([utf8_text])
        if repaired_utf8 and not looks_broken_console_text(repaired_utf8):
            return repaired_utf8.rstrip("\r\n")
    return best_text


def read_console_input(prompt: str = "") -> str:
    if (
        readline is not None
        and hasattr(sys.stdin, "isatty")
        and sys.stdin.isatty()
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    ):
        raw_line = input(prompt)
        best_text = choose_best_console_text([raw_line])
        return (best_text or strip_terminal_control_sequences(raw_line)).rstrip("\r\n")

    if prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    if hasattr(sys.stdin, "buffer"):
        raw_line = sys.stdin.buffer.readline()
        if raw_line == b"":
            raise EOFError
        return decode_console_input_bytes(raw_line)

    raw_line = sys.stdin.readline()
    if raw_line == "":
        raise EOFError
    return choose_best_console_text([raw_line]) or raw_line.rstrip("\r\n")


def prompt_text(prompt: str, *, default: str | None = None, allow_empty: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = normalize_path_input(read_console_input(f"{prompt}{suffix}: ").strip())
        if value:
            return value
        if default is not None:
            return default
        if allow_empty:
            return ""
        print("Нужно что-то ввести.\n", flush=True)


def prompt_choice(title: str, options: list[str]) -> int:
    while True:
        print(title, flush=True)
        for index, option in enumerate(options, start=1):
            print(f"{index}. {option}", flush=True)
        raw = read_console_input("Выбор: ").strip()
        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= len(options):
                return value
        print("Не понял выбор. Попробуй еще раз.\n", flush=True)


def prompt_yes_no(prompt: str, *, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw_value = read_console_input(f"{prompt} {suffix}: ").strip().lower()
        if not raw_value:
            return default
        if raw_value in {"y", "yes", "д", "да"}:
            return True
        if raw_value in {"n", "no", "н", "нет"}:
            return False
        print("Ответь 'да' или 'нет'.\n", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bot_runtime_module(project_root: Path) -> Any | None:
    global bot_runtime_module_cache
    global bot_runtime_module_error

    if bot_runtime_module_cache is not None:
        return bot_runtime_module_cache
    if bot_runtime_module_error is not None:
        return None

    bot_path = project_root / BOT_ENTRYPOINT
    try:
        spec = importlib.util.spec_from_file_location("heymate_launcher_runtime", bot_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Не удалось создать spec для {bot_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        bot_runtime_module_error = str(exc)
        return None

    bot_runtime_module_cache = module
    return module


def require_bot_runtime_module(project_root: Path) -> Any | None:
    module = load_bot_runtime_module(project_root)
    if module is not None:
        return module
    print(
        "Не удалось подгрузить runtime-инструменты из bot.py.\n"
        f"Причина: {bot_runtime_module_error or 'неизвестно'}\n"
        "Скорее всего не хватает зависимостей или bot.py сейчас сам себя закопал.\n",
        flush=True,
    )
    pause()
    return None


def parse_positive_int(raw_value: Any) -> int | None:
    text = str(raw_value or "").strip()
    if not text.isdigit():
        return None
    value = int(text)
    if value < 1:
        return None
    return value


def coerce_nonnegative_int(raw_value: Any, default: int = 0) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(0, value)


def terminal_sessions_root(project_root: Path) -> Path:
    return project_root / TERMINAL_SESSIONS_DIR_NAME


def format_terminal_session_timestamp(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return "-"
    return text.replace("T", " ")[:19]


def list_terminal_sessions(project_root: Path) -> list[dict[str, Any]]:
    root = terminal_sessions_root(project_root)
    if not root.is_dir():
        return []

    sessions: list[dict[str, Any]] = []
    for path in root.glob("*.json"):
        session_number = parse_positive_int(path.stem)
        if session_number is None:
            continue
        payload = load_json(path)
        history = payload.get("history", [])
        settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
        tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
        sessions.append(
            {
                "session_number": session_number,
                "path": path,
                "title": str(payload.get("title") or "").strip() or "Без названия",
                "updated_at": format_terminal_session_timestamp(
                    payload.get("updated_at")
                    or payload.get("last_opened_at")
                    or payload.get("created_at")
                ),
                "request_count": coerce_nonnegative_int(payload.get("request_count")),
                "history_items": len(history) if isinstance(history, list) else 0,
                "tags": [str(item).strip() for item in tags if str(item).strip()],
                "response_mode": str(settings.get("response_mode") or "chat"),
                "active_project_id": str(settings.get("active_project_id") or ""),
                "attached_model_name": str(payload.get("attached_model_name") or ""),
            }
        )
    sessions.sort(key=lambda item: int(item["session_number"]))
    return sessions


def print_terminal_sessions_overview(project_root: Path) -> list[dict[str, Any]]:
    sessions = list_terminal_sessions(project_root)
    if not sessions:
        print("Сохранённых terminal-сессий пока нет.\n", flush=True)
        return []

    print("Сохранённые terminal-сессии:", flush=True)
    for session in sessions:
        tags = f" | теги: {', '.join(session['tags'][:3])}" if session.get("tags") else ""
        model_name = f" | модель: {session['attached_model_name']}" if session.get("attached_model_name") else ""
        print(
            f"#{session['session_number']} | {session['title']} | "
            f"запросов: {session['request_count']} | "
            f"сообщений в памяти: {session['history_items']} | "
            f"режим: {session.get('response_mode') or 'chat'} | "
            f"обновлена: {session['updated_at']}{tags}{model_name}",
            flush=True,
        )
    print("", flush=True)
    return sessions


def search_terminal_sessions(project_root: Path, query: str) -> list[dict[str, Any]]:
    needle = str(query or "").strip().casefold()
    if not needle:
        return list_terminal_sessions(project_root)
    matches: list[dict[str, Any]] = []
    for session in list_terminal_sessions(project_root):
        haystack = " ".join(
            [
                str(session.get("title") or ""),
                str(session.get("attached_model_name") or ""),
                str(session.get("response_mode") or ""),
                " ".join(str(tag) for tag in session.get("tags") or []),
                str(session.get("active_project_id") or ""),
            ]
        ).casefold()
        if needle in haystack:
            matches.append(session)
    return matches


def read_text_tail(path: Path, max_chars: int = LOG_PREVIEW_CHARS) -> str:
    if not path.is_file():
        return "Лог пока не найден."
    return path.read_text(encoding="utf-8", errors="ignore")[-max_chars:].strip() or "Лог пуст."


def site_dashboard_runtime_root(project_root: Path) -> Path:
    return project_root / SITE_DASHBOARD_RUNTIME_DIR_NAME


def site_dashboard_state_path(project_root: Path) -> Path:
    return site_dashboard_runtime_root(project_root) / SITE_DASHBOARD_STATE_FILE_NAME


def site_dashboard_process_state_path(project_root: Path) -> Path:
    return site_dashboard_runtime_root(project_root) / SITE_DASHBOARD_PROCESS_FILE_NAME


def site_dashboard_log_path(project_root: Path) -> Path:
    return site_dashboard_runtime_root(project_root) / SITE_DASHBOARD_LOG_FILE_NAME


def coerce_site_dashboard_port(raw_value: Any) -> int:
    try:
        port = int(str(raw_value or "").strip())
    except (TypeError, ValueError):
        return SITE_DASHBOARD_DEFAULT_PORT
    return port if 1 <= port <= 65535 else SITE_DASHBOARD_DEFAULT_PORT


def coerce_site_dashboard_refresh_seconds(raw_value: Any) -> int:
    try:
        value = int(str(raw_value or "").strip())
    except (TypeError, ValueError):
        return SITE_DASHBOARD_DEFAULT_REFRESH_SECONDS
    return max(2, value)


def normalize_network_host_candidate(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    text = text.strip("[]")
    try:
        candidate_ip = ipaddress.ip_address(text)
    except ValueError:
        return ""
    if candidate_ip.is_loopback or candidate_ip.is_unspecified or candidate_ip.is_multicast or candidate_ip.is_link_local:
        return ""
    return candidate_ip.compressed


def format_http_url(host: str, port: int) -> str:
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return f"http://{host}:{port}"
    if parsed.version == 6:
        return f"http://[{parsed.compressed}]:{port}"
    return f"http://{parsed.compressed}:{port}"


def collect_site_dashboard_access_hosts(public_host_override: str = "") -> list[str]:
    hosts: list[str] = []
    seen: set[str] = set()

    def add_host(candidate: Any) -> None:
        normalized = normalize_network_host_candidate(candidate)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        hosts.append(normalized)

    add_host(public_host_override)

    ssh_connection = os.getenv("SSH_CONNECTION", "").strip()
    if ssh_connection:
        ssh_parts = ssh_connection.split()
        if len(ssh_parts) >= 4:
            add_host(ssh_parts[2])

    hostname_i = run_capture_command(["hostname", "-I"], timeout=5)
    if hostname_i is not None and hostname_i.returncode == 0:
        for token in (hostname_i.stdout or "").split():
            add_host(token)

    ip_addr = run_capture_command(["ip", "-o", "-4", "addr", "show", "scope", "global"], timeout=5)
    if ip_addr is not None and ip_addr.returncode == 0:
        for raw_line in (ip_addr.stdout or "").splitlines():
            parts = raw_line.split()
            if len(parts) >= 4 and "/" in parts[3]:
                add_host(parts[3].split("/", 1)[0])

    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None):
            if family == socket.AF_INET and sockaddr:
                add_host(sockaddr[0])
    except Exception:
        pass

    return hosts


def load_site_dashboard_settings(project_root: Path) -> dict[str, Any]:
    env_values = parse_env_file(project_root / ENV_FILE_NAME)
    host = str(
        env_values.get("SITE_DASHBOARD_HOST")
        or os.getenv("SITE_DASHBOARD_HOST")
        or SITE_DASHBOARD_DEFAULT_HOST
    ).strip() or SITE_DASHBOARD_DEFAULT_HOST
    public_host_override = str(
        env_values.get("SITE_DASHBOARD_PUBLIC_HOST")
        or os.getenv("SITE_DASHBOARD_PUBLIC_HOST")
        or ""
    ).strip()
    port = coerce_site_dashboard_port(
        env_values.get("SITE_DASHBOARD_PORT")
        or os.getenv("SITE_DASHBOARD_PORT")
        or SITE_DASHBOARD_DEFAULT_PORT
    )
    refresh_seconds = coerce_site_dashboard_refresh_seconds(
        env_values.get("SITE_DASHBOARD_REFRESH_SECONDS")
        or os.getenv("SITE_DASHBOARD_REFRESH_SECONDS")
        or SITE_DASHBOARD_DEFAULT_REFRESH_SECONDS
    )
    listen_url = format_http_url(host, port)
    if host in {"0.0.0.0", "::"}:
        access_hosts = collect_site_dashboard_access_hosts(public_host_override)
        if not access_hosts:
            access_hosts = ["127.0.0.1"]
    else:
        access_hosts = [public_host_override or host]
    access_urls = [format_http_url(candidate_host, port) for candidate_host in access_hosts]
    return {
        "host": host,
        "public_host_override": public_host_override,
        "port": port,
        "refresh_seconds": refresh_seconds,
        "listen_url": listen_url,
        "access_hosts": access_hosts,
        "access_urls": access_urls,
        "url": access_urls[0],
    }


def site_dashboard_reverse_proxy_root(project_root: Path) -> Path:
    return site_dashboard_runtime_root(project_root) / SITE_DASHBOARD_REVERSE_PROXY_DIR_NAME


def is_probably_domain_name(raw_value: str) -> bool:
    text = str(raw_value or "").strip().strip(".")
    if not text or " " in text:
        return False
    try:
        ipaddress.ip_address(text.strip("[]"))
    except ValueError:
        return "." in text and all(part and re.match(r"^[A-Za-z0-9-]+$", part) for part in text.split("."))
    return False


def build_site_dashboard_caddyfile(project_root: Path, server_name: str) -> str:
    settings = load_site_dashboard_settings(project_root)
    upstream = f"127.0.0.1:{settings['port']}"
    normalized_name = str(server_name or "").strip()
    if not normalized_name:
        normalized_name = settings.get("public_host_override") or settings["access_hosts"][0]
    site_label = normalized_name
    if not is_probably_domain_name(normalized_name):
        site_label = f"http://{normalized_name}"
    return textwrap.dedent(
        f"""
        {site_label} {{
            reverse_proxy {upstream}

            header {{
                X-Forwarded-Proto {{scheme}}
                X-Forwarded-Host {{host}}
                X-Real-IP {{remote_host}}
            }}

            encode zstd gzip
        }}
        """
    ).strip() + "\n"


def build_site_dashboard_nginx_config(project_root: Path, server_name: str) -> str:
    settings = load_site_dashboard_settings(project_root)
    normalized_name = str(server_name or "").strip() or "_"
    return textwrap.dedent(
        f"""
        server {{
            listen 80;
            server_name {normalized_name};

            client_max_body_size 32m;

            location / {{
                proxy_pass http://127.0.0.1:{settings['port']};
                proxy_http_version 1.1;
                proxy_set_header Host $host;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_read_timeout 3600s;
                proxy_send_timeout 3600s;
            }}
        }}
        """
    ).strip() + "\n"


def write_site_dashboard_reverse_proxy_configs(project_root: Path, server_name: str) -> tuple[Path, Path]:
    root = site_dashboard_reverse_proxy_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    caddy_path = root / SITE_DASHBOARD_CADDY_FILE_NAME
    nginx_path = root / SITE_DASHBOARD_NGINX_FILE_NAME
    caddy_path.write_text(build_site_dashboard_caddyfile(project_root, server_name), encoding="utf-8")
    nginx_path.write_text(build_site_dashboard_nginx_config(project_root, server_name), encoding="utf-8")
    return caddy_path, nginx_path


def check_local_tcp_port(host: str, port: int, timeout: float = 0.7) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def collect_port_listener_details(port: int) -> list[str]:
    commands = [
        ["ss", "-ltnp"],
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
    ]
    results: list[str] = []
    seen: set[str] = set()
    for command in commands:
        if not shutil.which(command[0]):
            continue
        completed = run_capture_command(command, timeout=8)
        if completed is None or completed.returncode != 0:
            continue
        for raw_line in (completed.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if command[0] == "ss" and f":{port}" not in line:
                continue
            if line in seen:
                continue
            seen.add(line)
            results.append(line)
    return results[:20]


def collect_firewall_status_lines(port: int) -> list[str]:
    lines: list[str] = []
    if shutil.which("ufw"):
        completed = run_capture_command(["ufw", "status"], timeout=10)
        if completed is not None and completed.returncode == 0:
            output = (completed.stdout or "").strip()
            if output:
                lowered = output.lower()
                if "inactive" in lowered:
                    lines.append("ufw: inactive")
                else:
                    matching = [line.strip() for line in output.splitlines() if str(port) in line]
                    if matching:
                        lines.extend(f"ufw: {line}" for line in matching[:4])
                    else:
                        lines.append(f"ufw: rules for {port}/tcp не нашёл")
    if shutil.which("firewall-cmd"):
        state_completed = run_capture_command(["firewall-cmd", "--state"], timeout=8)
        if state_completed is not None and state_completed.returncode == 0:
            state = (state_completed.stdout or "").strip()
            if state == "running":
                ports_completed = run_capture_command(["firewall-cmd", "--list-ports"], timeout=8)
                ports_text = (ports_completed.stdout or "").strip() if ports_completed is not None else ""
                if f"{port}/tcp" in ports_text.split():
                    lines.append(f"firewalld: {port}/tcp открыт")
                else:
                    lines.append(f"firewalld: {port}/tcp не найден в открытых портах")
            else:
                lines.append(f"firewalld: state={state or 'unknown'}")
    return lines


def save_site_dashboard_network_settings(
    project_root: Path,
    *,
    host: str,
    public_host: str,
    port: int,
    refresh_seconds: int,
) -> None:
    order, values = load_env_template(project_root)
    values["SITE_DASHBOARD_HOST"] = str(host).strip() or SITE_DASHBOARD_DEFAULT_HOST
    values["SITE_DASHBOARD_PORT"] = str(coerce_site_dashboard_port(port))
    values["SITE_DASHBOARD_REFRESH_SECONDS"] = str(coerce_site_dashboard_refresh_seconds(refresh_seconds))
    normalized_public_host = str(public_host or "").strip()
    if normalized_public_host:
        values["SITE_DASHBOARD_PUBLIC_HOST"] = normalized_public_host
    else:
        values.pop("SITE_DASHBOARD_PUBLIC_HOST", None)
    write_env_file(project_root, values, order)


def build_site_dashboard_launch_env(project_root: Path) -> dict[str, str]:
    settings = load_site_dashboard_settings(project_root)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SITE_DASHBOARD_HOST"] = str(settings["host"])
    env["SITE_DASHBOARD_PORT"] = str(settings["port"])
    env["SITE_DASHBOARD_REFRESH_SECONDS"] = str(settings["refresh_seconds"])
    return env


def extract_site_dashboard_access_code(log_text: str) -> str:
    match = re.search(r"Код доступа создан:\s*([^\s]+)", str(log_text or ""))
    return match.group(1).strip() if match else ""


def read_process_command(pid: int) -> str:
    if pid < 1:
        return ""
    completed = run_capture_command(["ps", "-p", str(pid), "-o", "command="], timeout=5)
    if completed is None or completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def find_running_site_dashboard_process(project_root: Path) -> dict[str, Any] | None:
    completed = run_capture_command(["ps", "-axo", "pid=,command="], timeout=5)
    if completed is None or completed.returncode != 0:
        return None

    project_marker = str(project_root.resolve())
    for raw_line in (completed.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        if not pid_text.isdigit():
            continue
        if SITE_DASHBOARD_ENTRYPOINT not in command:
            continue
        if project_marker not in command:
            continue
        return {"pid": int(pid_text), "command": command}
    return None


def get_site_dashboard_status(project_root: Path, *, cleanup_stale: bool = True) -> dict[str, Any]:
    settings = load_site_dashboard_settings(project_root)
    process_state = load_json(site_dashboard_process_state_path(project_root))
    pid = parse_positive_int(process_state.get("pid"))
    command = ""
    if pid is not None:
        command = read_process_command(pid)
        if command and SITE_DASHBOARD_ENTRYPOINT in command and str(project_root.resolve()) in command:
            return {
                **settings,
                "running": True,
                "pid": pid,
                "command": command,
                "log_path": site_dashboard_log_path(project_root),
                "state_path": site_dashboard_state_path(project_root),
                "process_state_path": site_dashboard_process_state_path(project_root),
                "recovered": False,
            }

    recovered = find_running_site_dashboard_process(project_root)
    if recovered is not None:
        process_payload = {
            "pid": int(recovered["pid"]),
            "host": settings["host"],
            "port": settings["port"],
            "url": settings["url"],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        process_state_path = site_dashboard_process_state_path(project_root)
        process_state_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(process_state_path, process_payload)
        return {
            **settings,
            "running": True,
            "pid": int(recovered["pid"]),
            "command": str(recovered.get("command") or ""),
            "log_path": site_dashboard_log_path(project_root),
            "state_path": site_dashboard_state_path(project_root),
            "process_state_path": process_state_path,
            "recovered": True,
        }

    process_state_path = site_dashboard_process_state_path(project_root)
    if cleanup_stale and process_state_path.is_file():
        try:
            process_state_path.unlink()
        except OSError:
            pass

    return {
        **settings,
        "running": False,
        "pid": pid,
        "command": command,
        "log_path": site_dashboard_log_path(project_root),
        "state_path": site_dashboard_state_path(project_root),
        "process_state_path": process_state_path,
        "recovered": False,
    }


def show_site_dashboard_status(project_root: Path) -> None:
    cls()
    status = get_site_dashboard_status(project_root)
    access_urls = status.get("access_urls") or [status["url"]]
    print(
        f"Статус веб-панели: {'работает' if status['running'] else 'остановлена'}\n"
        f"Слушает: {status['listen_url']}\n"
        f"Заходить можно по: {access_urls[0]}\n"
        f"PID: {status.get('pid') or '-'}\n"
        f"Runtime: {status['state_path']}\n"
        f"Лог: {status['log_path']}\n",
        flush=True,
    )
    if len(access_urls) > 1:
        print("Другие найденные адреса входа:", flush=True)
        for url in access_urls[1:]:
            print(f"- {url}", flush=True)
        print("", flush=True)
    if status["host"] in {"127.0.0.1", "::1", "localhost"}:
        print(
            "Сейчас панель привязана только к loopback. Это безопасный режим по умолчанию.\n"
            "Если нужен внешний доступ, выставь SITE_DASHBOARD_HOST=0.0.0.0 и не забудь про firewall.\n",
            flush=True,
        )
    if status["running"] and status["recovered"]:
        print("Процесс нашёл по `ps` и снова привязал к лаунчеру. Хоть кто-то тут умеет чинить бардак автоматически.\n", flush=True)
    pause()


def open_site_dashboard_in_browser(project_root: Path) -> None:
    cls()
    status = get_site_dashboard_status(project_root)
    if not status["running"]:
        print("Веб-панель пока не запущена. Сначала подними её, а потом уже зови браузер на работу.\n", flush=True)
        pause()
        return

    opened = False
    try:
        opened = bool(webbrowser.open(status["url"], new=2))
    except Exception:
        opened = False

    if opened:
        print(f"Открыл панель в браузере: {status['url']}\n", flush=True)
    else:
        print(
            f"Браузер сам не стартанул. Открой руками: {status['url']}\n"
            "Да, иногда автоматизация тоже уходит в эмоциональный отпуск.\n",
            flush=True,
        )
    pause()


def launch_site_dashboard(project_root: Path, *, open_browser: bool = False) -> None:
    cls()
    status = get_site_dashboard_status(project_root)
    access_urls = status.get("access_urls") or [status["url"]]
    if status["running"]:
        print(
            f"Веб-панель уже крутится: {access_urls[0]} (PID {status['pid']}).\n"
            f"Слушает: {status['listen_url']}\n"
            f"Лог: {status['log_path']}\n",
            flush=True,
        )
        if len(access_urls) > 1:
            print("Другие адреса входа:", flush=True)
            for url in access_urls[1:]:
                print(f"- {url}", flush=True)
            print("", flush=True)
        if open_browser:
            open_site_dashboard_in_browser(project_root)
            return
        pause()
        return

    ensure_python_dependencies(project_root)
    settings = load_site_dashboard_settings(project_root)
    runtime_root = site_dashboard_runtime_root(project_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    log_path = site_dashboard_log_path(project_root)
    state_path = site_dashboard_state_path(project_root)
    first_launch = not state_path.is_file()
    command = [*python_command(), str(project_root / SITE_DASHBOARD_ENTRYPOINT)]
    launch_env = build_site_dashboard_launch_env(project_root)

    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            env=launch_env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    time.sleep(1.2)
    if process.poll() is not None:
        print("Веб-панель рухнула почти сразу после старта. Лови хвост лога ниже.\n", flush=True)
        print(read_text_tail(log_path), flush=True)
        pause()
        return

    process_state_path = site_dashboard_process_state_path(project_root)
    process_state_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        process_state_path,
        {
            "pid": process.pid,
            "host": settings["host"],
            "port": settings["port"],
            "url": settings["url"],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    log_tail = read_text_tail(log_path, max_chars=4000)
    access_code = extract_site_dashboard_access_code(log_tail)
    access_urls = settings.get("access_urls") or [settings["url"]]
    print(
        f"Веб-панель поднял в фоне.\n"
        f"Слушает: {settings['listen_url']}\n"
        f"Заходить можно по: {access_urls[0]}\n"
        f"PID: {process.pid}\n"
        f"Runtime: {state_path}\n"
        f"Лог: {log_path}\n",
        flush=True,
    )
    if len(access_urls) > 1:
        print("Другие адреса входа:", flush=True)
        for url in access_urls[1:]:
            print(f"- {url}", flush=True)
        print("", flush=True)
    if first_launch:
        if access_code:
            print(f"Код доступа первого запуска: {access_code}\n", flush=True)
        else:
            print("Первый запуск был, но код доступа смотри в логе. Он не потерялся, просто спрятался по старой доброй традиции.\n", flush=True)

    if open_browser:
        opened = False
        try:
            opened = bool(webbrowser.open(settings["url"], new=2))
        except Exception:
            opened = False
        if opened:
            print("Браузер тоже пнул. Если он не открылся, значит браузер решил внезапно показать характер.\n", flush=True)
        else:
            print(f"Браузер не открылся автоматически. Тогда просто зайди сам: {settings['url']}\n", flush=True)
    pause()


def stop_site_dashboard(project_root: Path) -> None:
    cls()
    status = get_site_dashboard_status(project_root, cleanup_stale=False)
    if not status["running"] or not status.get("pid"):
        print("Веб-панель и так не запущена. Тут даже останавливать особенно нечего.\n", flush=True)
        process_state_path = site_dashboard_process_state_path(project_root)
        if process_state_path.is_file():
            try:
                process_state_path.unlink()
            except OSError:
                pass
        pause()
        return

    pid = int(status["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        print(f"Не удалось остановить веб-панель: {exc}\n", flush=True)
        pause()
        return

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not read_process_command(pid):
            break
        time.sleep(0.2)
    if read_process_command(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(0.2)

    process_state_path = site_dashboard_process_state_path(project_root)
    if process_state_path.is_file():
        try:
            process_state_path.unlink()
        except OSError:
            pass

    print("Веб-панель остановил. Маленький Flask-ларёк больше не болтается в фоне.\n", flush=True)
    pause()


def show_site_dashboard_log(project_root: Path) -> None:
    cls()
    log_path = site_dashboard_log_path(project_root)
    print("=== site_dashboard.log ===", flush=True)
    print(read_text_tail(log_path), flush=True)
    pause()


def configure_site_dashboard_network_settings_from_launcher(project_root: Path) -> None:
    cls()
    settings = load_site_dashboard_settings(project_root)
    current_host = settings["host"]
    current_public_host = settings.get("public_host_override") or ""
    current_port = settings["port"]
    current_refresh = settings["refresh_seconds"]
    print(
        "Сетевые настройки веб-панели.\n"
        "Если панель висит за reverse proxy, обычно лучше слушать 127.0.0.1, а наружу отдавать домен через Caddy/Nginx.\n",
        flush=True,
    )
    host = prompt_text("HOST", default=current_host)
    public_host = prompt_text("PUBLIC_HOST (домен/IP для ссылок, можно пусто)", default=current_public_host, allow_empty=True)
    try:
        port = coerce_site_dashboard_port(prompt_text("PORT", default=str(current_port)))
        refresh = coerce_site_dashboard_refresh_seconds(prompt_text("REFRESH_SECONDS", default=str(current_refresh)))
    except Exception:
        print("С числами не срослось. Нормально впиши порт и refresh.\n", flush=True)
        pause()
        return
    save_site_dashboard_network_settings(
        project_root,
        host=host,
        public_host=public_host,
        port=port,
        refresh_seconds=refresh,
    )
    print(
        "Сетевые настройки панели сохранил.\n"
        f"- Слушать: {host}:{port}\n"
        f"- Публичный адрес: {public_host or '(автоопределение)'}\n"
        f"- Обновление UI: {refresh} сек.\n",
        flush=True,
    )
    pause()


def show_site_dashboard_network_diagnostics(project_root: Path) -> None:
    cls()
    settings = load_site_dashboard_settings(project_root)
    status = get_site_dashboard_status(project_root)
    state = load_json(site_dashboard_state_path(project_root))
    whitelist = str(state.get("ip_whitelist_text") or "").strip()
    local_probe_hosts = ["127.0.0.1"]
    if settings["host"] not in {"0.0.0.0", "::", "127.0.0.1", "::1", "localhost"}:
        local_probe_hosts.insert(0, settings["host"])
    print(
        "Диагностика веб-панели.\n"
        f"- Слушает: {settings['listen_url']}\n"
        f"- Публичные ссылки: {', '.join(settings['access_urls'])}\n"
        f"- Процесс панели: {'жив' if status['running'] else 'не найден'}\n"
        f"- PID: {status.get('pid') or '-'}\n"
        f"- PUBLIC_HOST: {settings.get('public_host_override') or '(не задан)'}\n"
        f"- Whitelist: {whitelist or 'пустой, доступ не ограничен по IP'}\n",
        flush=True,
    )
    for host in local_probe_hosts:
        probe_ok = check_local_tcp_port(host, settings["port"])
        print(f"- TCP probe {host}:{settings['port']}: {'открыт' if probe_ok else 'не отвечает'}", flush=True)
    listener_lines = collect_port_listener_details(settings["port"])
    if listener_lines:
        print("\nЧто слушает порт:", flush=True)
        for line in listener_lines:
            print(f"- {line}", flush=True)
    else:
        print("\nПо `ss/lsof` слушателя на этом порту не увидел.\n", flush=True)
    firewall_lines = collect_firewall_status_lines(settings["port"])
    if firewall_lines:
        print("Firewall/доступ:", flush=True)
        for line in firewall_lines:
            print(f"- {line}", flush=True)
        print("", flush=True)
    proxy_root = site_dashboard_reverse_proxy_root(project_root)
    if proxy_root.is_dir():
        print("Сгенерированные proxy-конфиги:", flush=True)
        for path in sorted(proxy_root.glob("*")):
            print(f"- {path}", flush=True)
        print("", flush=True)
    pause()


def generate_site_dashboard_reverse_proxy_files(project_root: Path) -> None:
    cls()
    settings = load_site_dashboard_settings(project_root)
    default_name = settings.get("public_host_override") or settings["access_hosts"][0]
    server_name = prompt_text("Домен или IP для reverse proxy", default=default_name)
    caddy_path, nginx_path = write_site_dashboard_reverse_proxy_configs(project_root, server_name)
    print(
        "Сгенерировал reverse proxy-конфиги.\n"
        f"- Caddy: {caddy_path}\n"
        f"- Nginx: {nginx_path}\n\n"
        "Подсказка:\n"
        "- Для HTTPS без цирка лучше домен и Caddy.\n"
        "- Если используешь reverse proxy, часто разумно держать SITE_DASHBOARD_HOST=127.0.0.1.\n",
        flush=True,
    )
    pause()


def site_dashboard_network_menu(project_root: Path) -> None:
    while True:
        cls()
        settings = load_site_dashboard_settings(project_root)
        print_block(
            f"""
            Сетевой слой веб-панели.
            Сейчас панель слушает {settings['listen_url']}, а наружу светится как {settings['url']}.
            Тут можно подкрутить host/port, проверить порт и сгенерировать reverse proxy-конфиги.
            """
        )
        choice = prompt_choice(
            "Сеть и HTTPS",
            [
                "Показать диагностику порта и firewall",
                "Изменить host/port/public host",
                "Сгенерировать Caddy/Nginx конфиги",
                "Назад",
            ],
        )
        if choice == 1:
            show_site_dashboard_network_diagnostics(project_root)
        elif choice == 2:
            configure_site_dashboard_network_settings_from_launcher(project_root)
        elif choice == 3:
            generate_site_dashboard_reverse_proxy_files(project_root)
        else:
            return


def install_site_dashboard_systemd_service(project_root: Path) -> None:
    cls()
    context = get_systemd_context(SITE_DASHBOARD_SYSTEMD_SERVICE_NAME)
    if context is None:
        print("systemctl не найден. На этой системе некуда ставить service для веб-панели.\n", flush=True)
        pause()
        return

    command_prefix, service_path, mode = context
    service_path.parent.mkdir(parents=True, exist_ok=True)
    service_path.write_text(
        build_site_dashboard_systemd_service_text(project_root, mode),
        encoding="utf-8",
    )

    daemon_reload = subprocess.run([*command_prefix, "daemon-reload"], check=False)
    enable_now = subprocess.run(
        [*command_prefix, "enable", "--now", SITE_DASHBOARD_SYSTEMD_SERVICE_NAME],
        check=False,
    )

    if daemon_reload.returncode != 0 or enable_now.returncode != 0:
        print("Не удалось включить systemd service для веб-панели. Проверь вывод выше.\n", flush=True)
        pause()
        return

    print(f"Service веб-панели установлен: {service_path}", flush=True)
    settings = load_site_dashboard_settings(project_root)
    access_urls = settings.get("access_urls") or [settings["url"]]
    print(
        f"Панель слушает: {settings['listen_url']}\n"
        f"Заходить можно по: {access_urls[0]}\n",
        flush=True,
    )
    if len(access_urls) > 1:
        print("Другие адреса входа:", flush=True)
        for url in access_urls[1:]:
            print(f"- {url}", flush=True)
        print("", flush=True)
    if mode == "user":
        print(
            "Если это headless Linux-сервер, может понадобиться loginctl enable-linger $USER.\n",
            flush=True,
        )
    pause()


def show_site_dashboard_systemd_status() -> None:
    cls()
    context = get_systemd_context(SITE_DASHBOARD_SYSTEMD_SERVICE_NAME)
    if context is None:
        print("systemctl не найден.\n", flush=True)
        pause()
        return

    command_prefix, _, _ = context
    subprocess.run(
        [*command_prefix, "status", SITE_DASHBOARD_SYSTEMD_SERVICE_NAME, "--no-pager", "--full"],
        check=False,
    )
    print("", flush=True)
    pause()


def manage_site_dashboard_systemd_service(action: str) -> None:
    cls()
    context = get_systemd_context(SITE_DASHBOARD_SYSTEMD_SERVICE_NAME)
    if context is None:
        print("systemctl не найден.\n", flush=True)
        pause()
        return

    command_prefix, _, _ = context
    completed = subprocess.run(
        [*command_prefix, action, SITE_DASHBOARD_SYSTEMD_SERVICE_NAME],
        check=False,
    )
    if completed.returncode == 0:
        print(f"systemd service веб-панели action completed: {action}\n", flush=True)
    else:
        print(f"Не удалось выполнить action '{action}' для systemd service веб-панели.\n", flush=True)
    pause()


def show_site_dashboard_systemd_logs(project_root: Path) -> None:
    cls()
    context = get_systemd_context(SITE_DASHBOARD_SYSTEMD_SERVICE_NAME)
    journalctl = shutil.which("journalctl")

    if context is not None and journalctl:
        command_prefix, _, mode = context
        journal_command = [journalctl]
        if mode == "user":
            journal_command.append("--user")
        journal_command.extend(
            ["-u", SITE_DASHBOARD_SYSTEMD_SERVICE_NAME, "-n", "200", "--no-pager"]
        )
        subprocess.run(journal_command, check=False)
        print("", flush=True)
        pause()
        return

    print("=== site_dashboard.log ===", flush=True)
    print(read_text_tail(site_dashboard_log_path(project_root)), flush=True)
    pause()


def remove_site_dashboard_systemd_service() -> None:
    cls()
    context = get_systemd_context(SITE_DASHBOARD_SYSTEMD_SERVICE_NAME)
    if context is None:
        print("systemctl не найден.\n", flush=True)
        pause()
        return

    command_prefix, service_path, _ = context
    subprocess.run(
        [*command_prefix, "disable", "--now", SITE_DASHBOARD_SYSTEMD_SERVICE_NAME],
        check=False,
    )
    if service_path.is_file():
        service_path.unlink()
    subprocess.run([*command_prefix, "daemon-reload"], check=False)
    print("systemd service веб-панели удалён.\n", flush=True)
    pause()


def list_recent_problem_long_think_jobs(project_root: Path, limit: int = 3) -> list[dict[str, Any]]:
    root = project_root / "deep_think_jobs"
    if not root.is_dir():
        return []

    jobs: list[dict[str, Any]] = []
    for result_path in root.glob("*/result.json"):
        payload = load_json(result_path)
        status = str(payload.get("status") or "")
        if status not in {"failed", "cancelled", "interrupted"}:
            continue
        jobs.append(payload)

    jobs.sort(key=lambda item: str(item.get("updated_at") or item.get("completed_at") or ""), reverse=True)
    return jobs[:limit]


def show_error_logs(project_root: Path) -> None:
    cls()
    runtime_log = project_root / "bot_logs" / "runtime.log"
    supervisor_log = project_root / "bot_logs" / "systemd_supervisor.log"
    llama_log = project_root / "bot_logs" / "llama_server.log"
    print("=== runtime.log ===", flush=True)
    print(read_text_tail(runtime_log), flush=True)
    print("\n=== systemd_supervisor.log ===", flush=True)
    print(read_text_tail(supervisor_log), flush=True)
    print("\n=== llama_server.log ===", flush=True)
    print(read_text_tail(llama_log), flush=True)

    problem_jobs = list_recent_problem_long_think_jobs(project_root)
    if problem_jobs:
        print("\n=== deep_think_jobs (проблемные) ===", flush=True)
        for job in problem_jobs:
            print(
                f"[{str(job.get('job_id') or '')[:8]}] статус: {job.get('status')} | "
                f"фаза: {job.get('phase') or '-'}",
                flush=True,
            )
            print(f"Ошибка: {job.get('error') or 'без текста'}", flush=True)
            print(f"JSON: {job.get('result_path') or '-'}\n", flush=True)
    pause()


def is_active_long_think_status(status: str) -> bool:
    return status in {"queued", "running", "sleeping", "finalizing"}


def list_active_long_think_jobs(project_root: Path, limit: int = 200) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    root = project_root / "deep_think_jobs"
    if not root.is_dir():
        return jobs

    for state_path in root.glob("*/state.json"):
        payload = load_json(state_path)
        if not payload:
            continue
        if not is_active_long_think_status(str(payload.get("status") or "")):
            continue
        jobs.append(payload)

    jobs.sort(
        key=lambda item: str(item.get("updated_at") or item.get("started_at") or item.get("created_at") or ""),
        reverse=True,
    )
    return jobs[:limit]


def is_systemd_service_active() -> bool:
    context = get_systemd_context()
    if context is None:
        return False

    command_prefix, _, _ = context
    completed = subprocess.run(
        [*command_prefix, "is-active", "--quiet", SYSTEMD_SERVICE_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def build_update_impact_warnings(project_root: Path) -> list[str]:
    warnings: list[str] = []
    active_jobs = list_active_long_think_jobs(project_root)
    if active_jobs:
        terminal_jobs = sum(1 for job in active_jobs if str(job.get("mode") or "") == "terminal")
        telegram_jobs = sum(1 for job in active_jobs if str(job.get("mode") or "") == "telegram")
        mode_bits: list[str] = []
        if terminal_jobs:
            mode_bits.append(f"terminal: {terminal_jobs}")
        if telegram_jobs:
            mode_bits.append(f"telegram: {telegram_jobs}")
        if not mode_bits:
            mode_bits.append("режим не определился")

        warnings.append(
            f"Сейчас активны {len(active_jobs)} long-think job'ов ({', '.join(mode_bits)})."
        )
        warnings.append(
            "Сам git update просто обновит файлы на диске, но ручной restart бота, stop/restart systemd service "
            "или смена модели после обновления могут оборвать эти job'ы."
        )
        if any(str(job.get("systemd_unit") or "").strip() for job in active_jobs):
            warnings.append(
                "Часть long-think job'ов отцеплена через systemd-run. Logout они обычно переживают, "
                "а вот явный stop/kill уже нет."
            )

    if is_systemd_service_active():
        warnings.append(
            "systemd service heymate-bot.service сейчас запущен. До ручного рестарта он продолжит жить старым процессом, "
            "а после рестарта поднимется уже на новой версии."
        )

    if active_jobs and is_systemd_service_active():
        warnings.append(
            "Если сразу после обновления дёрнуть restart service, текущие ответы и активные job'ы могут прерваться."
        )

    return warnings


def build_port_manual_text() -> str:
    return (
        "Что умеет этот порт:\n"
        "- Запускать Telegram-бота на локальной GGUF-модели через llama.cpp.\n"
        "- Работать в terminal-режиме с постоянными сессиями.\n"
        "- Сохранять terminal-сессии и возвращаться в них позже.\n"
        "- Держать локальную БЗ и индексы проектов, но не навязывать их, если они тебе не нужны.\n"
        "- Переключать режим ответа между обычным чатом и режимом написания кода.\n"
        "- Управлять локальными моделями, задачами индексации и проектным контекстом.\n"
        "- Пускать long-think job'ы через /deepthink и сохранять итог в JSON.\n"
        "- Показывать логи ошибок, состояние env, системный статус и проверять обновления.\n\n"
        "Основные команды в Telegram и terminal:\n"
        "/help - подробная справка.\n"
        "/mode show|chat|code - показать или сменить режим ответа.\n"
        "/reset - сброс памяти диалога.\n"
        "/ineedmore - до трёх независимых запросов в одной пачке.\n"
        "/kb ... - локальная БЗ: add/list/search/remove/clear/on/off.\n"
        "/project ... - индексы проектов: add/list/use/search/rescan/remove.\n"
        "/tasks ... - очередь и история фоновых задач индексации.\n"
        "/model ... и /models - показать, выбрать или удалить локальную модель.\n"
        "/deepthink <длительность> <запрос> - длинная фоновая проработка задачи.\n"
        "/deepplan <запрос> - сначала оценить адекватный срок, потом уже запускать deepthink.\n"
        "  Внутри long-think сначала строится шаблон ответа, потом идёт прогресс и собирается сводка по CPU/RAM/GPU.\n"
        "/deepstatus - статус long-think job'ов.\n"
        "/deepcancel [job_id] - отмена long-think job.\n"
        "/errors - свежие runtime, supervisor-логи и ошибки.\n"
        "/status - состояние модели и llama-server.\n"
        "/source / /license - исходники и лицензия.\n\n"
        "Как указывать длительность для /deepthink:\n"
        "- Суффиксами: 4d, 4h, 4m, 4s, 1h30m.\n"
        "- Через двоеточия: 00:30:00 (часы:минуты:секунды), 05:20 (минуты:секунды).\n"
        "- Полный формат: 01:12:30:45 (дни:часы:минуты:секунды).\n\n"
        "Что есть в лаунчере:\n"
        "- Запуск бота и terminal-режима.\n"
        "- Сессии: посмотреть, войти по номеру, удалить по номеру.\n"
        "- Локальная БЗ: быстрый импорт, поиск и чистка документов.\n"
        "- Проекты: индексация папок, просмотр деталей, перескан и удаление индексов.\n"
        "- Фоновые задачи: история очереди и чистка хвостов после индексации.\n"
        "- Веб-панель: запуск в фоне, статус, лог и открытие в браузере.\n"
        "- Веб-панель через systemd: отдельный сервис, чтобы Flask-пульт жил независимо от SSH-сессии.\n"
        "- Менеджер моделей: список найденных .gguf, быстрая активация, удаление и настройка env.\n"
        "- Проверка обновления и установка новой версии.\n"
        "- Анализ живых long-think/systemd процессов перед установкой обновления.\n"
        "- Логи ошибок.\n"
        "- Управление systemd отдельным меню."
    )


def show_port_manual() -> None:
    cls()
    print(build_port_manual_text() + "\n", flush=True)
    pause()


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def write_env_file(project_root: Path, values: dict[str, str], order: list[str]) -> None:
    output_lines: list[str] = []
    written: set[str] = set()
    for key in order:
        if key in values:
            output_lines.append(f"{key}={values[key]}")
            written.add(key)
    for key in sorted(values):
        if key not in written:
            output_lines.append(f"{key}={values[key]}")
    (project_root / ENV_FILE_NAME).write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def normalize_path_input(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    text = ansi_escape.sub("", text)
    text = "".join(char for char in text if char.isprintable())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


def resolve_existing_file_path(raw_value: Any, project_root: Path, *, suffix: str | None = None) -> Path | None:
    normalized_value = normalize_path_input(raw_value)
    if not normalized_value:
        return None
    try:
        candidate = Path(normalized_value).expanduser()
        candidate = candidate if candidate.is_absolute() else (project_root / candidate).resolve()
        candidate = candidate.resolve()
    except Exception:
        return None
    if not candidate.is_file():
        return None
    if suffix and candidate.suffix.lower() != suffix.lower():
        return None
    return candidate


def resolve_existing_dir_path(raw_value: Any, project_root: Path) -> Path | None:
    normalized_value = normalize_path_input(raw_value)
    if not normalized_value:
        return None
    try:
        candidate = Path(normalized_value).expanduser()
        candidate = candidate if candidate.is_absolute() else (project_root / candidate).resolve()
        candidate = candidate.resolve()
    except Exception:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def load_env_template(project_root: Path) -> tuple[list[str], dict[str, str]]:
    example_path = project_root / ENV_EXAMPLE_FILE_NAME
    env_path = project_root / ENV_FILE_NAME
    template_values = parse_env_file(example_path)
    current_values = parse_env_file(env_path)
    merged = dict(template_values)
    merged.update(current_values)

    order: list[str] = []
    if example_path.is_file():
        for raw_line in example_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if key and key not in order:
                order.append(key)

    for key in current_values:
        if key not in order:
            order.append(key)

    return order, merged


def project_root() -> Path:
    return Path(__file__).resolve().parent


def python_command() -> list[str]:
    candidates = [[sys.executable], ["python3"], ["python"]]
    for candidate in candidates:
        try:
            completed = subprocess.run(
                [*candidate, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except OSError:
            continue
        if completed.returncode == 0:
            return candidate
    raise RuntimeError("Не удалось найти установленный Python 3.")


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def run_capture_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = UPDATE_CHECK_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def is_module_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def pip_installed_packages() -> set[str]:
    python = python_command()
    try:
        completed = subprocess.run(
            [*python, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout or "[]")
        return {
            normalize_package_name(str(item.get("name", "")))
            for item in payload
            if isinstance(item, dict) and item.get("name")
        }
    except Exception:
        return set()


def ensure_python_dependencies(project_root: Path) -> None:
    required = {
        "aiogram": "aiogram>=3.0,<4.0",
        "aiohttp": "aiohttp>=3.9,<4.0",
        "flask": "Flask>=3.0,<4.0",
    }
    installed_packages = pip_installed_packages()
    missing = [
        package
        for module, package in required.items()
        if normalize_package_name(module) not in installed_packages and not is_module_installed(module)
    ]
    if not missing:
        return

    print_block(
        """
        Я обнаружил что нету нужных для работы библиотек, сейчас все сделаю.....
        """
    )
    run_command(
        [*python_command(), "-m", "pip", "install", "--disable-pip-version-check", *missing],
        cwd=project_root,
    )


def git_available() -> bool:
    return shutil.which("git") is not None


def is_git_repository(project_root: Path) -> bool:
    return (project_root / ".git").exists()


def git_output(project_root: Path, *args: str, timeout: int = UPDATE_CHECK_TIMEOUT_SECONDS) -> str:
    completed = run_capture_command(["git", *args], cwd=project_root, timeout=timeout)
    if completed is None or completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def git_output_lines(project_root: Path, *args: str, timeout: int = UPDATE_CHECK_TIMEOUT_SECONDS) -> list[str]:
    return [line.strip() for line in git_output(project_root, *args, timeout=timeout).splitlines() if line.strip()]


def current_git_branch(project_root: Path) -> str:
    return git_output(project_root, "symbolic-ref", "--short", "HEAD") or ""


def current_git_head_sha(project_root: Path, ref: str = "HEAD") -> str:
    return git_output(project_root, "rev-parse", ref) or ""


def git_has_local_changes(project_root: Path) -> bool:
    completed = run_capture_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=project_root,
    )
    if completed is None:
        return False
    return bool((completed.stdout or "").strip())


def git_remote_can_fast_forward_current_branch(project_root: Path, branch: str) -> bool | None:
    completed = run_capture_command(
        ["git", "merge-base", "--is-ancestor", "HEAD", f"{GIT_REMOTE_NAME}/{branch}"],
        cwd=project_root,
    )
    if completed is None:
        return None
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    return None


def realign_current_branch_to_remote_snapshot(project_root: Path, branch: str) -> subprocess.CompletedProcess[str] | None:
    return run_capture_command(
        ["git", "checkout", "-B", branch, f"{GIT_REMOTE_NAME}/{branch}"],
        cwd=project_root,
        timeout=60,
    )


def git_remote_url(project_root: Path, remote_name: str = GIT_REMOTE_NAME) -> str:
    return git_output(project_root, "remote", "get-url", remote_name) or ""


def normalize_version_label(raw_tag: str) -> str:
    tag = raw_tag.strip()
    if not tag:
        return ""
    if tag.startswith("refs/tags/"):
        tag = tag.split("/", 2)[-1]
    if tag.endswith("^{}"):
        tag = tag[:-3]
    if tag.lower().startswith("v") and len(tag) > 1 and tag[1].isdigit():
        tag = tag[1:]
    if not tag or not tag[0].isdigit():
        return ""
    if not re.fullmatch(r"[0-9A-Za-z._-]+", tag):
        return ""
    return tag.replace("_", ".")


def list_merged_version_tags(project_root: Path, ref: str = "HEAD") -> list[str]:
    tags: list[str] = []
    for line in git_output_lines(project_root, "tag", "--merged", ref, "--sort=-v:refname"):
        normalized = normalize_version_label(line)
        if normalized and normalized not in tags:
            tags.append(normalized)
    return tags


def version_tag_points_at_ref(project_root: Path, ref: str = "HEAD") -> str:
    for line in git_output_lines(project_root, "tag", "--points-at", ref, "--sort=-v:refname"):
        normalized = normalize_version_label(line)
        if normalized:
            return normalized
    return ""


def remote_version_tags_by_sha(project_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in git_output_lines(project_root, "ls-remote", "--tags", GIT_REMOTE_NAME, timeout=UPDATE_CHECK_TIMEOUT_SECONDS):
        parts = line.split()
        if len(parts) != 2:
            continue
        sha, ref_name = parts
        normalized = normalize_version_label(ref_name)
        if normalized and sha not in mapping:
            mapping[sha] = normalized
    return mapping


def build_version_label(sha: str, *, exact_version: str = "", base_version: str = "") -> str:
    short_sha = sha[:7] if sha else "unknown"
    if exact_version:
        return exact_version
    if base_version:
        return f"{base_version}+{short_sha}"
    return short_sha


def parse_github_repo_slug(remote_url: str) -> tuple[str, str] | None:
    raw = remote_url.strip()
    if not raw:
        return None

    https_match = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?$", raw)
    if https_match:
        return https_match.group(1), https_match.group(2)
    return None


def human_readable_download_size(size_bytes: int | None) -> str:
    if size_bytes is None or size_bytes <= 0:
        return "размер не удалось определить"
    value = float(size_bytes)
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ"]
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def estimate_remote_archive_size_bytes(project_root: Path, branch: str) -> int | None:
    repo_slug = parse_github_repo_slug(git_remote_url(project_root))
    if repo_slug is None:
        repo_slug = parse_github_repo_slug(REPO_PAGE_URL)
    if repo_slug is None:
        return None

    owner, repo = repo_slug
    archive_url = (
        f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/"
        f"{urllib.parse.quote(branch, safe='')}"
    )

    request = urllib.request.Request(
        archive_url,
        method="HEAD",
        headers={"User-Agent": "HeyMateLauncher/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            size = response.headers.get("Content-Length", "").strip()
            return int(size) if size.isdigit() else None
    except Exception:
        return None


def collect_project_update_status(project_root: Path) -> dict[str, Any]:
    if not git_available():
        return {"status": "unavailable", "reason": "git не найден."}
    if not is_git_repository(project_root):
        return {"status": "unavailable", "reason": "Текущий проект не является git-клоном."}

    local_exact_version = version_tag_points_at_ref(project_root, "HEAD")
    base_versions = list_merged_version_tags(project_root, "HEAD")
    base_version = base_versions[0] if base_versions else ""
    branch = current_git_branch(project_root)
    local_sha = current_git_head_sha(project_root)
    if not branch or not local_sha:
        return {"status": "unavailable", "reason": "Не удалось определить текущую ветку или commit."}

    local_version = build_version_label(
        local_sha,
        exact_version=local_exact_version,
        base_version=base_version,
    )
    common = {
        "branch": branch,
        "local_sha": local_sha,
        "local_version": local_version,
        "has_local_changes": git_has_local_changes(project_root),
        "impact_warnings": build_update_impact_warnings(project_root),
    }

    remote_line = git_output(
        project_root,
        "ls-remote",
        "--heads",
        GIT_REMOTE_NAME,
        branch,
    )
    if not remote_line:
        return {
            **common,
            "status": "unavailable",
            "reason": "Не удалось получить состояние удалённой ветки.",
        }

    remote_sha = remote_line.split()[0].strip()
    if not remote_sha:
        return {
            **common,
            "status": "unavailable",
            "reason": "Удалённая ветка вернула пустой SHA.",
        }

    remote_exact_version = remote_version_tags_by_sha(project_root).get(remote_sha, "")
    remote_version = build_version_label(
        remote_sha,
        exact_version=remote_exact_version,
        base_version=base_version,
    )

    if remote_sha == local_sha:
        return {
            **common,
            "status": "up_to_date",
            "remote_sha": remote_sha,
            "remote_version": remote_version,
            "download_size_bytes": 0,
        }

    return {
        **common,
        "status": "update_available",
        "remote_sha": remote_sha,
        "remote_version": remote_version,
        "download_size_bytes": estimate_remote_archive_size_bytes(project_root, branch),
    }


def collect_project_update_info(project_root: Path) -> dict[str, Any] | None:
    status = collect_project_update_status(project_root)
    if status.get("status") != "update_available":
        return None
    return {**status}


def restart_launcher(project_root: Path) -> None:
    launcher_path = project_root / "launcher_cli.py"
    os.execv(sys.executable, [sys.executable, str(launcher_path), *sys.argv[1:]])


def install_project_update(project_root: Path, update_info: dict[str, Any]) -> bool:
    branch = str(update_info["branch"])
    print_live_banner(
        "Идет загрузка",
        f"Обновляю ветку {branch}. Скачалось: подготовка...",
        progress=0.15,
    )
    fetch_completed = run_capture_command(
        ["git", "fetch", "--tags", GIT_REMOTE_NAME, branch],
        cwd=project_root,
        timeout=60,
    )
    if fetch_completed is None or fetch_completed.returncode != 0:
        print_live_banner(
            "Идет загрузка",
            f"Обновляю ветку {branch}. Скачалось: fetch не удался.",
            progress=1.0,
            final_newline=True,
        )
        return False

    can_fast_forward = git_remote_can_fast_forward_current_branch(project_root, branch)
    if can_fast_forward is False:
        if git_has_local_changes(project_root):
            print_live_banner(
                "Идет загрузка",
                f"Ветка {branch} была переписана, но локальное git-дерево не чистое.",
                progress=1.0,
                final_newline=True,
            )
            return False
        print_live_banner(
            "Идет загрузка",
            f"Ветка {branch} была переписана. Пересинхронизирую snapshot...",
            progress=0.82,
        )
        reset_completed = realign_current_branch_to_remote_snapshot(project_root, branch)
        if reset_completed is None or reset_completed.returncode != 0:
            print_live_banner(
                "Идет загрузка",
                f"Обновляю ветку {branch}. Скачалось: snapshot-sync не удался.",
                progress=1.0,
                final_newline=True,
            )
            return False
        print_live_banner(
            "Идет загрузка",
            f"Ветка {branch} пересинхронизирована по remote snapshot.",
            done=True,
            final_newline=True,
        )
        return True

    print_live_banner(
        "Идет загрузка",
        f"Обновляю ветку {branch}. Скачалось: fetch готов, применяю update...",
        progress=0.75,
    )
    merge_completed = run_capture_command(
        ["git", "merge", "--ff-only", f"{GIT_REMOTE_NAME}/{branch}"],
        cwd=project_root,
        timeout=60,
    )
    if merge_completed is None or merge_completed.returncode != 0:
        print_live_banner(
            "Идет загрузка",
            f"Обновляю ветку {branch}. Скачалось: merge не удался.",
            progress=1.0,
            final_newline=True,
        )
        return False
    print_live_banner(
        "Идет загрузка",
        f"Обновление ветки {branch} применено.",
        done=True,
        final_newline=True,
    )
    return True


def handle_project_update_status(
    project_root: Path,
    update_status: dict[str, Any],
    *,
    show_if_latest: bool,
    show_unavailable: bool,
    show_intro_banner: bool,
) -> None:
    if show_intro_banner:
        print_live_banner(
            "Идет загрузка",
            "Проверяю обновление лаунчера...",
            progress=0.1,
            final_newline=True,
        )
    status = str(update_status.get("status") or "")

    if status == "unavailable":
        if show_unavailable:
            print(
                update_status.get("reason")
                or "Не удалось проверить обновления.",
                flush=True,
            )
            print("", flush=True)
        return

    if status == "up_to_date":
        if show_if_latest:
            print(
                render_live_banner(
                    "Все готово :3",
                    "Установлена самая-самая последняя версия :3. Мяу",
                    done=True,
                ),
                flush=True,
            )
            print("", flush=True)
        return

    local_version = str(update_status["local_version"])
    remote_version = str(update_status["remote_version"])
    size_text = human_readable_download_size(update_status.get("download_size_bytes"))
    impact_warnings = [
        str(item).strip()
        for item in (update_status.get("impact_warnings") or [])
        if str(item).strip()
    ]

    if update_status.get("has_local_changes"):
        print(
            render_live_banner(
                "Идет загрузка",
                f"Доступна версия {remote_version}, но есть локальные изменения. Вес: {size_text}.",
                progress=1.0,
            ),
            flush=True,
        )
        print(
            "Локальные изменения найдены, так что автоматически поверх них я не полезу.\n",
            flush=True,
        )
        if impact_warnings:
            print("Дополнительно учти:\n", flush=True)
            for warning in impact_warnings:
                print(f"- {warning}", flush=True)
            print("", flush=True)
        return

    print(
        f"Доступно обновление {remote_version}!\n"
        f"Текущая версия: {local_version}\n"
        f"Весит обновление: {size_text}\n",
        flush=True,
    )
    if impact_warnings:
        print("Что важно перед установкой:\n", flush=True)
        for warning in impact_warnings:
            print(f"- {warning}", flush=True)
        print("", flush=True)

    if not prompt_yes_no("Устанавливаю?", default=True):
        print("Ок, пока останешься на текущей версии.\n", flush=True)
        return

    if not install_project_update(project_root, update_status):
        print(
            "Не удалось установить обновление. "
            "Проверь сеть, права и чистоту git-дерева. "
            "Если ветку переписывали или перезаливали заново, локальных изменений быть не должно вообще.\n",
            flush=True,
        )
        return

    record_installed_project_update(project_root, update_status)
    print("Обновление поставил. Перезапускаю лаунчер...\n", flush=True)
    time.sleep(1)
    restart_launcher(project_root)


def check_for_project_update(
    project_root: Path,
    *,
    show_if_latest: bool,
) -> None:
    handle_project_update_status(
        project_root,
        collect_project_update_status(project_root),
        show_if_latest=show_if_latest,
        show_unavailable=True,
        show_intro_banner=True,
    )


def maybe_offer_project_update(project_root: Path) -> None:
    update_status = collect_project_update_status(project_root)
    if update_status.get("status") != "update_available":
        return
    handle_project_update_status(
        project_root,
        update_status,
        show_if_latest=False,
        show_unavailable=False,
        show_intro_banner=False,
    )


def append_update_history_record(project_root: Path, record: dict[str, Any]) -> None:
    state_path = project_root / STATE_FILE_NAME
    state = load_json(state_path)
    history = state.get("update_history")
    if not isinstance(history, list):
        history = []
    history.append(record)
    state["update_history"] = history[-UPDATE_HISTORY_LIMIT:]
    save_json(state_path, state)


def list_update_history_records(project_root: Path) -> list[dict[str, Any]]:
    state = load_json(project_root / STATE_FILE_NAME)
    history = state.get("update_history")
    if not isinstance(history, list):
        return []
    return [item for item in history if isinstance(item, dict)]


def record_installed_project_update(project_root: Path, update_status: dict[str, Any]) -> None:
    append_update_history_record(
        project_root,
        {
            "type": "update",
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "branch": str(update_status.get("branch") or current_git_branch(project_root)),
            "from_sha": str(update_status.get("local_sha") or ""),
            "to_sha": str(update_status.get("remote_sha") or current_git_head_sha(project_root)),
            "from_version": str(update_status.get("local_version") or ""),
            "to_version": str(update_status.get("remote_version") or current_git_head_sha(project_root)[:7]),
        },
    )


def backups_root(project_root: Path) -> Path:
    return project_root / BACKUPS_DIR_NAME


def list_project_backups(project_root: Path) -> list[Path]:
    root = backups_root(project_root)
    if not root.is_dir():
        return []
    return sorted((path for path in root.glob("*.tar.gz") if path.is_file()), key=lambda item: item.name, reverse=True)


def create_project_backup_archive(project_root: Path, *, label: str = "manual") -> Path:
    root = backups_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip() or "manual").strip("-") or "manual"
    archive_path = root / f"heymate_backup_{stamp}_{safe_label}.tar.gz"
    include_paths = [
        project_root / ENV_FILE_NAME,
        project_root / STATE_FILE_NAME,
        project_root / "web_panel_runtime",
        project_root / "terminal_sessions",
        project_root / "deep_think_jobs",
        project_root / "local_kb",
        project_root / "project_contexts",
        project_root / "task_queue",
        project_root / "bot_logs",
    ]
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in include_paths:
            if not path.exists():
                continue
            archive.add(path, arcname=str(path.relative_to(project_root)))
    return archive_path


def print_backups_overview(project_root: Path) -> list[Path]:
    backups = list_project_backups(project_root)
    if not backups:
        print("Бэкапов пока нет. То есть живёшь дерзко и без сетки снизу.\n", flush=True)
        return []
    print("Последние бэкапы:", flush=True)
    for index, path in enumerate(backups, start=1):
        print(
            f"{index}. {path.name} | {human_readable_download_size(path.stat().st_size)} | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime))}",
            flush=True,
        )
    print("", flush=True)
    return backups


def create_project_backup_from_launcher(project_root: Path) -> None:
    cls()
    backup_path = create_project_backup_archive(project_root)
    print(
        "Бэкап собрал.\n"
        f"- Файл: {backup_path.name}\n"
        f"- Путь: {backup_path}\n"
        f"- Размер: {human_readable_download_size(backup_path.stat().st_size)}\n",
        flush=True,
    )
    pause()


def show_project_backups(project_root: Path) -> None:
    cls()
    print_backups_overview(project_root)
    pause()


def rollback_last_project_update(project_root: Path) -> None:
    cls()
    history = [item for item in list_update_history_records(project_root) if item.get("type") == "update"]
    if not history:
        print("Истории обновлений пока нет. Откатывать просто не на что.\n", flush=True)
        pause()
        return
    if git_has_local_changes(project_root):
        print(
            "Git-дерево не чистое. Откат на грязном дереве делать не буду, а то потом сам будешь проклинать этот момент.\n",
            flush=True,
        )
        pause()
        return
    record = history[-1]
    target_sha = str(record.get("from_sha") or "").strip()
    if not target_sha:
        print("В истории обновлений нет SHA для отката. Тут уже магия закончилась.\n", flush=True)
        pause()
        return
    print(
        "Последнее обновление:\n"
        f"- Было: {record.get('from_version') or target_sha[:7]}\n"
        f"- Стало: {record.get('to_version') or str(record.get('to_sha') or '')[:7]}\n"
        f"- Откатываю на: {target_sha[:7]}\n",
        flush=True,
    )
    if not prompt_yes_no("Перед откатом сделать бэкап?", default=True):
        backup_path = None
    else:
        backup_path = create_project_backup_archive(project_root, label="before-rollback")
        print(f"Бэкап перед откатом собрал: {backup_path}\n", flush=True)
    if not prompt_yes_no("Точно откатываю последнее обновление?", default=False):
        print("Ок, откат отменил.\n", flush=True)
        pause()
        return
    completed = run_capture_command(["git", "reset", "--hard", target_sha], cwd=project_root, timeout=60)
    if completed is None or completed.returncode != 0:
        print("Откат не удался. Git решил показать характер.\n", flush=True)
        if completed is not None and (completed.stderr or completed.stdout):
            print((completed.stderr or completed.stdout).strip() + "\n", flush=True)
        pause()
        return
    append_update_history_record(
        project_root,
        {
            "type": "rollback",
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "target_sha": target_sha,
            "target_version": str(record.get("from_version") or target_sha[:7]),
            "backup_path": str(backup_path) if backup_path is not None else "",
        },
    )
    print("Откат применил. Перезапускаю лаунчер...\n", flush=True)
    time.sleep(1)
    restart_launcher(project_root)


def download_file(url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "HeyMateLinux/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0") or "0")
        written = 0
        chunk_size = 1024 * 1024
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if total > 0:
                    print_live_banner(
                        "Идет загрузка",
                        (
                            f"{label}. Скачалось: {written * 100 / total:.1f}% "
                            f"({human_readable_download_size(written)} / {human_readable_download_size(total)})"
                        ),
                        progress=written / total,
                    )
                else:
                    print_live_banner(
                        "Идет загрузка",
                        f"{label}. Скачалось: {human_readable_download_size(written)}",
                        progress=None,
                    )
    print_live_banner(
        "Идет загрузка",
        f"Скачалось: {label}",
        done=True,
        final_newline=True,
    )


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(target_dir)
        return
    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix.lower() == ".tgz":
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(target_dir)
        return
    raise RuntimeError(f"Не знаю как распаковать архив: {archive_path.name}")


def iter_common_model_roots(project_root: Path) -> list[Path]:
    invoke_cwd = normalize_path_input(os.getenv("HEYMATE_INVOKE_CWD", ""))
    invoke_dir = Path(invoke_cwd).expanduser() if invoke_cwd else None
    roots = [
        project_root / "models",
        project_root / "model",
        project_root,
        project_root.parent / "models",
        project_root.parent / "model",
        Path.home() / "models",
        Path.home() / "model",
        Path.home() / "Downloads",
        Path.home() / ".cache" / "huggingface",
        Path("/opt/models"),
        Path("/opt/model"),
        Path("/srv/models"),
        Path("/srv/model"),
        Path("/var/lib/models"),
        Path("/var/lib/model"),
        Path("/usr/local/share/models"),
        Path("/usr/local/share/model"),
    ]
    for base in [project_root, *project_root.parents[:4]]:
        roots.extend(
            [
                base / "models",
                base / "model",
            ]
        )
    if invoke_dir is not None:
        roots = [
            invoke_dir / "models",
            invoke_dir / "model",
            invoke_dir,
            invoke_dir.parent / "models",
            invoke_dir.parent / "model",
            *roots,
        ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except Exception:
            continue
        if not resolved.exists():
            continue
        marker = str(resolved).lower()
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(resolved)
    return unique


def iter_system_model_roots(project_root: Path) -> list[Path]:
    roots: list[Path] = []
    if os.name == "nt":
        anchor = Path.home().anchor
        if anchor:
            roots.append(Path(anchor))
        roots.append(project_root.anchor and Path(project_root.anchor) or project_root)
    else:
        roots.append(Path("/"))

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            continue
        marker = str(resolved).lower()
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(resolved)
    return unique


def should_skip_model_scan_dir(path: Path, *, system_wide: bool) -> bool:
    name = path.name.lower()
    if name in MODEL_SCAN_SKIP_DIR_NAMES:
        return True
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True
    if not system_wide:
        return False
    path_text = str(path)
    if os.name != "nt":
        if any(path_text == prefix or path_text.startswith(prefix + os.sep) for prefix in MODEL_SCAN_SKIP_ABSOLUTE_PREFIXES):
            return True
    return False


def collect_gguf_candidates_under_root(root: Path, *, limit: int, system_wide: bool) -> list[Path]:
    if root.is_file():
        return [root.resolve()] if root.suffix.lower() == ".gguf" else []
    if not root.is_dir():
        return []

    matches: list[Path] = []
    try:
        walk_root = root.resolve()
    except Exception:
        walk_root = root

    try:
        for current_dir, dirnames, filenames in os.walk(walk_root, topdown=True, onerror=lambda _exc: None, followlinks=False):
            current_path = Path(current_dir)
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not should_skip_model_scan_dir(current_path / dirname, system_wide=system_wide)
            ]
            for filename in filenames:
                if not filename.lower().endswith(".gguf"):
                    continue
                candidate = current_path / filename
                try:
                    resolved = candidate.resolve()
                except Exception:
                    resolved = candidate
                matches.append(resolved)
                if len(matches) >= limit:
                    return matches
    except Exception:
        return matches
    return matches


def score_model_candidate(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    score = 0
    if path.name == RECOMMENDED_MODEL_FILE:
        score += 1000
    if "qwen3.5" in name:
        score += 300
    elif "qwen" in name:
        score += 150
    if "35b" in name:
        score += 120
    if "a3b" in name:
        score += 80
    if "uncensored" in name:
        score += 40
    if "q5_k_m" in name:
        score += 90
    elif "q4_k_m" in name:
        score += 70
    elif "bf16" in name:
        score -= 25
    return (score, -len(str(path)), str(path).lower())


def model_identity_key(path: Path) -> str:
    try:
        return str(path.resolve()).lower()
    except Exception:
        return str(path).lower()


def find_external_model_paths(project_root: Path) -> list[Path]:
    env_candidate = resolve_existing_file_path(os.getenv("MODEL_PATH", "").strip(), project_root, suffix=".gguf")
    matches: list[tuple[tuple[int, int, str], Path]] = []
    seen: set[str] = set()
    if env_candidate is not None:
        seen.add(model_identity_key(env_candidate))
        matches.append((score_model_candidate(env_candidate), env_candidate))

    for root in iter_common_model_roots(project_root):
        for candidate in collect_gguf_candidates_under_root(
            root,
            limit=MODEL_SCAN_COMMON_ROOT_LIMIT,
            system_wide=False,
        ):
            marker = model_identity_key(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            matches.append((score_model_candidate(candidate), candidate))

    for root in iter_system_model_roots(project_root):
        for candidate in collect_gguf_candidates_under_root(
            root,
            limit=MODEL_SCAN_SYSTEM_ROOT_LIMIT,
            system_wide=True,
        ):
            marker = model_identity_key(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            matches.append((score_model_candidate(candidate), candidate))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in matches]


def resolve_model_input_path(raw_value: Any, project_root: Path) -> Path | None:
    normalized_value = normalize_path_input(raw_value)
    direct_file = resolve_existing_file_path(raw_value, project_root, suffix=".gguf")
    if direct_file is not None:
        return direct_file

    directory = resolve_existing_dir_path(raw_value, project_root)
    if directory is None:
        return None

    candidates = sorted(directory.glob("*.gguf"))
    if len(candidates) == 1:
        return candidates[0].resolve()

    file_name = Path(normalized_value).name.lower() if normalized_value else ""
    if file_name:
        name_matches: list[Path] = []
        seen: set[str] = set()
        for candidate in find_external_model_paths(project_root):
            if candidate.name.lower() != file_name:
                continue
            marker = model_identity_key(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            name_matches.append(candidate)
        if len(name_matches) == 1:
            return name_matches[0]
    return None


def format_model_choice(path: Path) -> str:
    try:
        size_gib = path.stat().st_size / (1024 ** 3)
    except OSError:
        size_gib = 0.0
    return f"{path.name} | {size_gib:.1f} GiB | {path}"


def format_model_size(path: Path) -> str:
    try:
        size_bytes = path.stat().st_size
    except OSError:
        return "размер неизвестен"
    gib = size_bytes / (1024 ** 3)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    mib = size_bytes / (1024 ** 2)
    return f"{mib:.1f} MiB"


def resolve_current_model_path(project_root: Path, state: dict[str, Any]) -> Path | None:
    env_values = parse_env_file(project_root / ENV_FILE_NAME)
    return (
        resolve_existing_file_path(env_values.get("MODEL_PATH", "").strip(), project_root, suffix=".gguf")
        or resolve_existing_file_path(state.get("model_path", ""), project_root, suffix=".gguf")
    )


def clear_current_model_reference(project_root: Path, state: dict[str, Any]) -> None:
    order, values = load_env_template(project_root)
    values["MODEL_PATH"] = ""
    write_env_file(project_root, values, order)
    state["model_path"] = ""
    state["configured"] = False
    state["env_review_required"] = True
    save_json(project_root / STATE_FILE_NAME, state)


def delete_current_model(project_root: Path, state: dict[str, Any]) -> None:
    cls()
    current_model = resolve_current_model_path(project_root, state)
    if current_model is None:
        print("Текущая модель не найдена. Удалять тут пока нечего.\n", flush=True)
        pause()
        return

    print(
        "Текущая модель:\n"
        f"- Имя: {current_model.name}\n"
        f"- Размер: {format_model_size(current_model)}\n"
        f"- Путь: {current_model}\n",
        flush=True,
    )
    if not prompt_yes_no("Удаляю этот файл модели?", default=False):
        print("Ок, модель пока оставил в живых.\n", flush=True)
        pause()
        return

    try:
        current_model.unlink()
    except OSError as exc:
        print(f"Не удалось удалить модель: {exc}\n", flush=True)
        pause()
        return

    clear_current_model_reference(project_root, state)
    print(
        "Модель удалил.\n"
        "MODEL_PATH в env очистил, так что дальше надо будет выбрать новую модель.\n",
        flush=True,
    )
    pause()


def huggingface_url(repo_id: str, filename: str) -> str:
    safe_filename = urllib.parse.quote(filename, safe="/")
    return f"https://huggingface.co/{repo_id}/resolve/main/{safe_filename}?download=true"


def download_model(repo_id: str, filename: str, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / Path(filename).name
    if destination.is_file():
        return destination
    download_file(huggingface_url(repo_id, filename), destination, "Качаю модель")
    return destination


def choose_model_path(project_root: Path) -> Path:
    models_dir = project_root / "models"
    choice = prompt_choice(
        "Теперь надо поставить .gguf модель. Что делаем?",
        [
            "Скачать свою модель с Hugging Face",
            "Скачать рекомендуемую модель разработчика",
            "Выбрать готовую модель из найденных или вбить путь вручную",
        ],
    )

    if choice == 1:
        repo_id = prompt_text("Введи repo_id модели на Hugging Face")
        filename = prompt_text("Введи точное имя GGUF-файла")
        return download_model(repo_id, filename, models_dir)
    if choice == 2:
        return download_model(RECOMMENDED_MODEL_REPO, RECOMMENDED_MODEL_FILE, models_dir)

    detected_models = find_external_model_paths(project_root)
    if detected_models:
        print_block(
            """
            Я сам прошерстил систему и нашёл готовые .gguf модели.
            Выбирай, что ставим.
            """
        )
        options = [format_model_choice(model_path) for model_path in detected_models]
        options.append("Вбить путь вручную")
        detected_choice = prompt_choice("Какую модель ставим?", options)
        if detected_choice <= len(detected_models):
            return detected_models[detected_choice - 1]

    while True:
        raw_path = prompt_text("Введи полный путь к .gguf модели")
        model_path = resolve_model_input_path(raw_path, project_root)
        if model_path is not None:
            return model_path
        nearby = detected_models[:5]
        print("Не вижу .gguf по этому пути. Можно вставить путь к файлу или к каталогу, где лежит одна модель.", flush=True)
        if nearby:
            print("Вот что я сейчас вижу рядом:", flush=True)
            for candidate in nearby:
                print(f"- {candidate}", flush=True)
        print("", flush=True)


def model_supports_fast_reply(model_path: Path) -> bool:
    name = model_path.name.lower()
    positive_markers = (
        "instruct",
        "chat",
        "assistant",
        "it",
        "coder",
        "qwen",
        "llama",
        "mistral",
        "mixtral",
        "deepseek",
        "gemma",
        "phi",
        "yi",
        "hermes",
        "zephyr",
        "dolphin",
        "nemotron",
    )
    negative_markers = ("base", "pretrain", "embedding", "rerank", "reranker")
    return any(marker in name for marker in positive_markers) or not any(marker in name for marker in negative_markers)


def model_profile_for_path(model_path: Path) -> dict[str, str]:
    name = model_path.name.lower()
    if "qwen" in name:
        chat_format = "qwen"
    elif any(marker in name for marker in ("mistral", "mixtral", "zephyr", "hermes", "openchat", "chatml")):
        chat_format = "chatml"
    else:
        chat_format = ""

    if any(marker in name for marker in ("coder", "code")):
        max_tokens = "4096"
        temperature = "0.35"
    elif any(marker in name for marker in ("instruct", "chat", "assistant", "it", "qwen", "llama", "mistral", "mixtral", "deepseek", "gemma", "phi")):
        max_tokens = "3072"
        temperature = "0.5"
    else:
        max_tokens = "2048"
        temperature = "0.6"

    if any(marker in name for marker in ("70b", "72b", "35b", "34b", "32b", "30b", "27b")):
        n_ctx = "32768"
    else:
        n_ctx = "16384"

    return {
        "CHAT_FORMAT": chat_format,
        "N_CTX": n_ctx,
        "MAX_TOKENS": max_tokens,
        "BRIEF_MAX_TOKENS": "240",
        "TEMPERATURE": temperature,
        "TOP_P": "0.95",
        "TOP_K": "20",
        "REPEAT_PENALTY": "1.1",
    }


def read_linux_mem_total_gib() -> float | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.is_file():
        return None
    try:
        for raw_line in meminfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if raw_line.startswith("MemTotal:"):
                parts = raw_line.split()
                if len(parts) >= 2:
                    mem_kib = int(parts[1])
                    return mem_kib / (1024 ** 2)
    except Exception:
        return None
    return None


def detect_nvidia_gpu() -> dict[str, Any] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None

    best_gpu: dict[str, Any] | None = None
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vram_gib = float(parts[-1]) / 1024.0
        except ValueError:
            continue
        candidate = {
            "vendor": "nvidia",
            "name": parts[0],
            "vram_gib": vram_gib,
        }
        if best_gpu is None or candidate["vram_gib"] > best_gpu["vram_gib"]:
            best_gpu = candidate
    return best_gpu


def detect_amd_gpu() -> dict[str, Any] | None:
    drm_root = Path("/sys/class/drm")
    if not drm_root.is_dir():
        return None

    best_gpu: dict[str, Any] | None = None
    for vram_path in drm_root.glob("card*/device/mem_info_vram_total"):
        try:
            total_bytes = int(vram_path.read_text(encoding="utf-8", errors="ignore").strip())
        except Exception:
            continue
        if total_bytes <= 0:
            continue
        vram_gib = total_bytes / (1024 ** 3)
        candidate = {
            "vendor": "amd",
            "name": vram_path.parents[1].name.upper(),
            "vram_gib": vram_gib,
        }
        if best_gpu is None or candidate["vram_gib"] > best_gpu["vram_gib"]:
            best_gpu = candidate
    return best_gpu


def detect_linux_gpu() -> dict[str, Any] | None:
    return detect_nvidia_gpu() or detect_amd_gpu()


def detect_llama_gpu_backend(llama_dir: Path, llama_server_exe: Path) -> str | None:
    search_roots: list[Path] = []
    for candidate in (llama_server_exe.parent, llama_dir):
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.exists() and resolved not in search_roots:
            search_roots.append(resolved)

    backend_tokens = {
        "cuda": ("cuda", "cublas"),
        "hip": ("hip", "rocm"),
        "vulkan": ("vulkan",),
        "opencl": ("opencl",),
        "sycl": ("sycl",),
    }

    for root in search_roots:
        root_name = root.name.lower()
        for backend, tokens in backend_tokens.items():
            if any(token in root_name for token in tokens):
                return backend

        for candidate in itertools.islice(root.rglob("*"), 400):
            if not candidate.is_file():
                continue
            file_name = candidate.name.lower()
            for backend, tokens in backend_tokens.items():
                if any(token in file_name for token in tokens):
                    return backend
    return None


def detect_linux_hardware_profile(
    model_path: Path,
    llama_dir: Path,
    llama_server_exe: Path,
    model_profile: dict[str, str],
) -> dict[str, Any]:
    logical_cores = max(1, os.cpu_count() or 1)
    ram_gib = read_linux_mem_total_gib() or 16.0
    try:
        model_size_gib = model_path.stat().st_size / (1024 ** 3)
    except OSError:
        model_size_gib = 0.0

    gpu = detect_linux_gpu()
    llama_backend = detect_llama_gpu_backend(llama_dir, llama_server_exe)

    if ram_gib < 8:
        ctx_cap = 4096
        batch_size = 64
        token_cap = 768
        brief_token_cap = 96
        history_messages = 4
    elif ram_gib < 12:
        ctx_cap = 8192
        batch_size = 128
        token_cap = 1024
        brief_token_cap = 128
        history_messages = 5
    elif ram_gib < 24:
        ctx_cap = 16384
        batch_size = 256
        token_cap = 1536
        brief_token_cap = 160
        history_messages = 6
    elif ram_gib < 48:
        ctx_cap = 16384
        batch_size = 384
        token_cap = 2048
        brief_token_cap = 200
        history_messages = 8
    else:
        ctx_cap = 32768
        batch_size = 512
        token_cap = int(model_profile["MAX_TOKENS"])
        brief_token_cap = int(model_profile["BRIEF_MAX_TOKENS"])
        history_messages = 10

    n_ctx = min(int(model_profile["N_CTX"]), ctx_cap)
    if model_size_gib >= 20 and ram_gib < 16:
        n_ctx = min(n_ctx, 4096)
    elif model_size_gib >= 20 and ram_gib < 24:
        n_ctx = min(n_ctx, 8192)

    max_tokens = min(int(model_profile["MAX_TOKENS"]), token_cap)
    brief_max_tokens = min(int(model_profile["BRIEF_MAX_TOKENS"]), brief_token_cap)
    n_threads = max(1, min(32, logical_cores - 1 if logical_cores > 1 else 1))

    start_timeout = 180
    if model_size_gib >= 20 and logical_cores < 8:
        start_timeout = 420
    elif model_size_gib >= 20 or ram_gib < 16:
        start_timeout = 300
    elif logical_cores < 8:
        start_timeout = 240

    n_gpu_layers = 0
    if gpu is not None and llama_backend is not None:
        vram_gib = float(gpu["vram_gib"])
        if llama_backend in ("cuda", "hip"):
            if vram_gib >= 24:
                n_gpu_layers = -1
            elif vram_gib >= 16:
                n_gpu_layers = 80
            elif vram_gib >= 12:
                n_gpu_layers = 40
            elif vram_gib >= 8:
                n_gpu_layers = 20
        elif llama_backend in ("vulkan", "opencl", "sycl"):
            if vram_gib >= 16:
                n_gpu_layers = 40
            elif vram_gib >= 12:
                n_gpu_layers = 24
            elif vram_gib >= 8:
                n_gpu_layers = 12

        if model_size_gib >= 20 and vram_gib < 12:
            n_gpu_layers = 0

    gpu_summary = "CPU-only"
    if gpu is not None:
        backend_label = llama_backend or "no llama.cpp GPU backend found"
        gpu_summary = f"{gpu['name']} | {gpu['vendor']} | {gpu['vram_gib']:.1f} GiB | backend: {backend_label}"

    summary_lines = [
        f"CPU threads detected: {logical_cores}",
        f"RAM detected: {ram_gib:.1f} GiB",
        f"Model size: {model_size_gib:.1f} GiB",
        f"GPU: {gpu_summary}",
        f"N_THREADS={n_threads}",
        f"N_CTX={n_ctx}",
        f"N_BATCH={batch_size}",
        f"N_GPU_LAYERS={n_gpu_layers}",
        f"MAX_TOKENS={max_tokens}",
        f"BRIEF_MAX_TOKENS={brief_max_tokens}",
        f"MAX_HISTORY_MESSAGES={history_messages}",
        f"LLAMA_SERVER_START_TIMEOUT={start_timeout}",
    ]

    return {
        "summary_lines": summary_lines,
        "overrides": {
            "N_THREADS": str(n_threads),
            "N_CTX": str(n_ctx),
            "N_BATCH": str(batch_size),
            "N_GPU_LAYERS": str(n_gpu_layers),
            "MAX_TOKENS": str(max_tokens),
            "BRIEF_MAX_TOKENS": str(brief_max_tokens),
            "MAX_HISTORY_MESSAGES": str(history_messages),
            "LLAMA_SERVER_START_TIMEOUT": str(start_timeout),
        },
    }


def print_linux_hardware_profile(profile: dict[str, Any]) -> None:
    summary_lines = profile.get("summary_lines") or []
    if not summary_lines:
        return
    print_block(
        "Оценил железо Linux и подстроил env под эту машину:\n"
        + "\n".join(f"- {line}" for line in summary_lines)
    )


def measure_file_read_speed_mib(path: Path, sample_bytes: int = 64 * 1024 * 1024) -> float | None:
    if not path.is_file():
        return None
    try:
        total_to_read = min(sample_bytes, max(1, path.stat().st_size))
    except OSError:
        return None
    chunk_size = 1024 * 1024
    started_at = time.perf_counter()
    read_bytes = 0
    try:
        with path.open("rb") as handle:
            while read_bytes < total_to_read:
                chunk = handle.read(min(chunk_size, total_to_read - read_bytes))
                if not chunk:
                    break
                read_bytes += len(chunk)
    except OSError:
        return None
    elapsed = time.perf_counter() - started_at
    if elapsed <= 0 or read_bytes <= 0:
        return None
    return round((read_bytes / (1024 ** 2)) / elapsed, 2)


def guess_model_use_case(path: Path) -> str:
    name = path.name.lower()
    if any(marker in name for marker in ("coder", "code", "codestral", "deepseek-coder")):
        return "код / рефактор / патчи"
    if any(marker in name for marker in ("instruct", "chat", "assistant", "it")):
        return "общий чат / инструкции"
    if any(marker in name for marker in ("reason", "thinking", "r1")):
        return "длинные рассуждения / сложные задачи"
    return "универсальный режим без явной специализации"


def build_model_fit_report(project_root: Path, state: dict[str, Any], model_path: Path) -> str:
    model_profile = model_profile_for_path(model_path)
    llama_server_exe = (
        resolve_existing_file_path(state.get("llama_server_exe"), project_root)
        or find_external_llama_server_exe(project_root)
    )
    llama_cpp_dir = (
        resolve_existing_dir_path(state.get("llama_cpp_dir"), project_root)
        or (llama_server_exe.parent if llama_server_exe is not None else None)
    )
    hardware_profile = (
        detect_linux_hardware_profile(model_path, llama_cpp_dir, llama_server_exe, model_profile)
        if llama_server_exe is not None and llama_cpp_dir is not None
        else {"summary_lines": [], "overrides": {}}
    )
    ram_gib = read_linux_mem_total_gib() or 0.0
    gpu = detect_linux_gpu()
    try:
        model_size_gib = model_path.stat().st_size / (1024 ** 3)
    except OSError:
        model_size_gib = 0.0
    read_speed = measure_file_read_speed_mib(model_path)
    n_gpu_layers = int((hardware_profile.get("overrides") or {}).get("N_GPU_LAYERS", "0") or 0)
    verdict = "Влезает нормально"
    if gpu is None and model_size_gib >= max(12.0, ram_gib * 0.9):
        verdict = "Очень тяжёлая для CPU-only этой машины"
    elif gpu is None and model_size_gib >= max(8.0, ram_gib * 0.6):
        verdict = "Жить сможет, но готовься к тяжёлому CPU-only режиму"
    elif gpu is not None and n_gpu_layers <= 0 and model_size_gib >= gpu.get("vram_gib", 0.0):
        verdict = "GPU есть, но эта модель в основном ляжет на RAM/CPU"
    elif gpu is not None and n_gpu_layers > 0:
        verdict = "Есть шанс нормально разгрузить часть слоёв на GPU"
    speed_line = (
        f"- Быстрый диск-прогон: ~{read_speed:.1f} MiB/s\n"
        if read_speed is not None
        else "- Быстрый диск-прогон: не удалось измерить\n"
    )
    lines = [
        "Fit-check модели:",
        f"- Модель: {model_path.name}",
        f"- Размер: {format_model_size(model_path)}",
        f"- Предполагаемый профиль: {guess_model_use_case(model_path)}",
        f"- Вердикт: {verdict}",
        f"- RAM машины: {ram_gib:.1f} GiB",
        (
            f"- GPU: {gpu['name']} ({gpu['vram_gib']:.1f} GiB)"
            if gpu is not None
            else "- GPU: N/A"
        ),
        speed_line.rstrip(),
    ]
    summary_lines = hardware_profile.get("summary_lines") or []
    if summary_lines:
        lines.extend(f"- {line}" for line in summary_lines)
    else:
        lines.append("- llama-runtime ещё не найден, поэтому рекомендации по N_CTX/N_GPU_LAYERS урезаны.")
    return "\n".join(lines)


def run_model_fit_check_from_launcher(project_root: Path, state: dict[str, Any]) -> None:
    cls()
    current_model = resolve_current_model_path(project_root, state)
    models = find_external_model_paths(project_root)
    target_model = current_model
    if target_model is None and models:
        options = [format_model_choice(model_path) for model_path in models[:12]]
        options.append("Назад")
        choice = prompt_choice("Какую модель оценивать", options)
        if choice == len(options):
            return
        target_model = models[choice - 1]
    if target_model is None:
        print("Модель для fit-check не нашёл. Тут даже анализировать пока нечего.\n", flush=True)
        pause()
        return
    print(build_model_fit_report(project_root, state, target_model) + "\n", flush=True)
    pause()


def handle_model_support(project_root: Path, current_model: Path) -> Path:
    if model_supports_fast_reply(current_model):
        print_block(
            """
            Отлично! Эта модель выглядит адекватно для быстрого ответа.
            """
        )
        return current_model

    choice = prompt_choice(
        "Похоже, эта модель может полезть в размышления и сломать быстрый ответ. Что делаем?",
        [
            "Оставляем текущую модель",
            "Качаем рекомендуемую модель",
        ],
    )
    if choice == 1:
        return current_model
    return download_model(RECOMMENDED_MODEL_REPO, RECOMMENDED_MODEL_FILE, project_root / "models")


def find_llama_server_exe(root: Path) -> Path | None:
    try:
        direct_candidate = root.resolve() / "llama-server"
    except Exception:
        return None
    if direct_candidate.is_file():
        return direct_candidate.resolve()
    try:
        for current_dir, dirnames, filenames in os.walk(root, onerror=lambda _exc: None):
            current_path = Path(current_dir)
            try:
                depth = len(current_path.relative_to(root).parts)
            except Exception:
                depth = 0
            if depth >= 5:
                dirnames[:] = []
            if "llama-server" not in filenames:
                continue
            candidate = current_path / "llama-server"
            if candidate.is_file():
                return candidate.resolve()
    except Exception:
        return None
    return None


def iter_common_llama_roots(project_root: Path) -> list[Path]:
    roots = [
        project_root,
        project_root / "llama.cpp",
        project_root.parent / "llama.cpp",
        Path.home() / "llama.cpp",
        Path.home() / "tools" / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("/opt/llama.cpp/build/bin"),
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/usr/local/llama.cpp"),
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except Exception:
            continue
        if not resolved.exists():
            continue
        marker = str(resolved).lower()
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(resolved)
    return unique


def find_external_llama_server_exe(project_root: Path) -> Path | None:
    env_server = os.getenv("LLAMA_SERVER_EXE", "").strip()
    if env_server:
        candidate = Path(env_server).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    env_dir = os.getenv("LLAMA_CPP_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir).expanduser() / "llama-server"
        if candidate.is_file():
            return candidate.resolve()

    command = shutil.which("llama-server")
    if command:
        candidate = Path(command)
        if candidate.is_file():
            return candidate.resolve()

    for root in iter_common_llama_roots(project_root):
        found = find_llama_server_exe(root)
        if found is not None:
            return found
    return None


def choose_llama_asset(assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [asset for asset in assets if isinstance(asset.get("name"), str)]
    forbidden_tokens = (
        "openvino",
        "vulkan",
        "rocm",
        "cuda",
        "hip",
        "metal",
        "sycl",
        "arm64",
        "aarch64",
        "s390x",
        "ppc64",
    )
    preferred_patterns = [
        ("ubuntu", "x64", ".tar.gz"),
        ("ubuntu", "amd64", ".tar.gz"),
        ("linux", "x64", ".tar.gz"),
        ("linux", "amd64", ".tar.gz"),
        ("ubuntu", "x64", ".zip"),
        ("ubuntu", "amd64", ".zip"),
        ("linux", "x64", ".zip"),
        ("linux", "amd64", ".zip"),
    ]
    for patterns in preferred_patterns:
        for asset in candidates:
            name = asset["name"].lower()
            if any(token in name for token in forbidden_tokens):
                continue
            if all(pattern in name for pattern in patterns):
                return asset
    for asset in candidates:
        name = asset["name"].lower()
        if any(token in name for token in forbidden_tokens):
            continue
        if "linux" in name and (name.endswith(".zip") or name.endswith(".tar.gz") or name.endswith(".tgz")):
            return asset
    return None


def ensure_llama_runtime(project_root: Path) -> tuple[Path, Path]:
    external = find_external_llama_server_exe(project_root)
    if external is not None:
        validate_llama_runtime(external.parent, external)
        print_block(
            f"""
            Нашел уже установленный llama.cpp.
            Использую вот этот llama-server:
            {external}
            """
        )
        return external.parent, external

    llama_root = project_root / "llama.cpp"
    llama_root.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        LLAMA_RELEASE_API_URL,
        headers={"User-Agent": "HeyMateLinux/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)

    asset = choose_llama_asset(release.get("assets") or [])
    if asset is None:
        raise RuntimeError("Не удалось найти Linux-сборку llama.cpp в latest release.")

    asset_name = asset["name"]
    asset_url = asset["browser_download_url"]
    if llama_root.exists():
        shutil.rmtree(llama_root, ignore_errors=True)
    llama_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / asset_name
        download_file(asset_url, archive_path, "Качаю llama.cpp")
        extract_archive(archive_path, llama_root)

    executable = find_llama_server_exe(llama_root)
    if executable is None:
        raise RuntimeError("Скачал llama.cpp, но не нашел llama-server после распаковки.")
    try:
        validate_llama_runtime(executable.parent, executable)
    except RuntimeError as exc:
        raise RuntimeError(f"Скачанная сборка llama.cpp '{asset_name}' не подходит этой системе. {exc}") from exc
    return executable.parent, executable


def build_default_env(
    project_root: Path,
    model_path: Path,
    llama_dir: Path,
    llama_server_exe: Path,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    order, values = load_env_template(project_root)
    profile = model_profile_for_path(model_path)
    hardware_profile = detect_linux_hardware_profile(model_path, llama_dir, llama_server_exe, profile)
    values["BOT_TOKEN"] = values.get("BOT_TOKEN", "")
    values["MODEL_PATH"] = str(model_path)
    values["LLAMA_CPP_DIR"] = str(llama_dir)
    values["LLAMA_SERVER_EXE"] = str(llama_server_exe)
    values["SOURCE_URL"] = REPO_PAGE_URL
    for key, value in profile.items():
        values[key] = value
    values.setdefault("AI_ENABLED", "true")
    values.update(hardware_profile.get("overrides") or {})
    for key in (
        "CHAT_FORMAT",
        "N_THREADS",
        "N_CTX",
        "N_BATCH",
        "N_GPU_LAYERS",
        "MAX_TOKENS",
        "BRIEF_MAX_TOKENS",
        "MAX_HISTORY_MESSAGES",
        "LLAMA_SERVER_START_TIMEOUT",
        "TEMPERATURE",
        "TOP_P",
        "TOP_K",
        "REPEAT_PENALTY",
        "AI_ENABLED",
    ):
        if key not in order:
            order.append(key)
    return order, values, hardware_profile


def validate_history_limit(raw_value: str) -> str:
    if raw_value == "-1":
        return raw_value
    if raw_value.isdigit():
        return str(int(raw_value))
    raise ValueError("Нужно положительное целое число или -1.")


def validate_existing_env(project_root: Path, state: dict[str, Any]) -> tuple[bool, dict[str, str]]:
    values = parse_env_file(project_root / ENV_FILE_NAME)
    if not values:
        return False, values

    if not values.get("BOT_TOKEN", "").strip():
        return False, values

    model_path = resolve_existing_file_path(values.get("MODEL_PATH", "").strip(), project_root, suffix=".gguf")
    if model_path is None:
        return False, values
    values["MODEL_PATH"] = str(model_path)

    try:
        values["MAX_HISTORY_MESSAGES"] = validate_history_limit(values.get("MAX_HISTORY_MESSAGES", "10").strip())
    except ValueError:
        return False, values

    llama_server_exe = resolve_existing_file_path(values.get("LLAMA_SERVER_EXE", "").strip(), project_root)
    if llama_server_exe is None and state.get("llama_server_exe"):
        llama_server_exe = resolve_existing_file_path(state["llama_server_exe"], project_root)
    if llama_server_exe is None:
        llama_server_exe = find_external_llama_server_exe(project_root)
    if llama_server_exe is None:
        return False, values
    values["LLAMA_SERVER_EXE"] = str(llama_server_exe)

    llama_cpp_dir = resolve_existing_dir_path(values.get("LLAMA_CPP_DIR", "").strip(), project_root)
    if llama_cpp_dir is None:
        llama_cpp_dir = llama_server_exe.parent
    values["LLAMA_CPP_DIR"] = str(llama_cpp_dir)
    return True, values


def mark_state_configured_from_env(project_root: Path, state: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    updated_state = dict(state)
    updated_state["configured"] = True
    updated_state["env_review_required"] = False
    updated_state["model_path"] = values.get("MODEL_PATH", updated_state.get("model_path", ""))
    updated_state["llama_server_exe"] = values.get("LLAMA_SERVER_EXE", updated_state.get("llama_server_exe", ""))
    updated_state["llama_cpp_dir"] = values.get("LLAMA_CPP_DIR", updated_state.get("llama_cpp_dir", ""))
    save_json(project_root / STATE_FILE_NAME, updated_state)
    return updated_state


def env_summary_lines(values: dict[str, str], ordered_keys: list[str]) -> list[str]:
    lines: list[str] = []
    for index, key in enumerate(ordered_keys, start=1):
        display_value = values.get(key, "")
        if key == "BOT_TOKEN" and display_value:
            display_value = display_value[:8] + "..." + display_value[-4:]
        lines.append(f"{index}. {key} = {display_value}")
    return lines


def configure_env(project_root: Path, state: dict[str, Any]) -> None:
    cls()
    print_block(
        """
        Добро пожаловать! На Linux я тоже могу провести тебя по env.
        Это нужно для работы модели и Telegram-бота.
        """
    )

    current_env_values = parse_env_file(project_root / ENV_FILE_NAME)
    model_path = (
        resolve_existing_file_path(current_env_values.get("MODEL_PATH"), project_root, suffix=".gguf")
        or resolve_existing_file_path(state.get("model_path"), project_root, suffix=".gguf")
        or next(iter(find_external_model_paths(project_root)), None)
    )
    llama_server_exe = (
        resolve_existing_file_path(current_env_values.get("LLAMA_SERVER_EXE"), project_root)
        or resolve_existing_file_path(state.get("llama_server_exe"), project_root)
        or find_external_llama_server_exe(project_root)
    )
    llama_dir = (
        resolve_existing_dir_path(current_env_values.get("LLAMA_CPP_DIR"), project_root)
        or resolve_existing_dir_path(state.get("llama_cpp_dir"), project_root)
        or (llama_server_exe.parent if llama_server_exe is not None else None)
    )

    if model_path is None:
        model_path = choose_model_path(project_root)
    state["model_path"] = str(model_path)

    if llama_server_exe is None:
        llama_dir, llama_server_exe = ensure_llama_runtime(project_root)
    if llama_dir is None:
        llama_dir = llama_server_exe.parent

    state["llama_server_exe"] = str(llama_server_exe)
    state["llama_cpp_dir"] = str(llama_dir)
    order, values, hardware_profile = build_default_env(project_root, model_path, llama_dir, llama_server_exe)
    print_linux_hardware_profile(hardware_profile)
    prompted_keys = ["BOT_TOKEN", "MODEL_PATH", "MAX_HISTORY_MESSAGES"]

    explanations = {
        "BOT_TOKEN": "Это токен Telegram-бота. Просто скопируй его из BotFather и вставь.",
        "MODEL_PATH": "Это путь к .gguf модели. Если все окей, просто жми Enter и оставляй как есть.",
        "MAX_HISTORY_MESSAGES": "Это размер памяти по сообщениям. Рекомендуемое значение - 10. Безлимитный режим -1.",
    }
    validators = {
        "BOT_TOKEN": lambda value: value if value else (_ for _ in ()).throw(ValueError("Токен не должен быть пустым.")),
        "MODEL_PATH": lambda value: value if Path(value).is_file() and Path(value).suffix.lower() == ".gguf" else (_ for _ in ()).throw(ValueError("Нужен путь к существующему .gguf-файлу.")),
        "MAX_HISTORY_MESSAGES": validate_history_limit,
    }

    while True:
        for key in prompted_keys:
            while True:
                print(f"{key} | {explanations[key]}", flush=True)
                try:
                    values[key] = validators[key](prompt_text("Введи значение", default=values.get(key, "")))
                except ValueError as exc:
                    print(f"{exc}\n", flush=True)
                    continue
                break
            print("", flush=True)

        while True:
            cls()
            print("Проверь параметры, все ли верно.\n", flush=True)
            summary = env_summary_lines(values, prompted_keys)
            print("\n".join(summary), flush=True)
            print("", flush=True)
            choice = prompt_choice("Все норм?", ["Да, все верно", "Нет, надо подкорректировать"])
            if choice == 1:
                write_env_file(project_root, values, order)
                state["configured"] = True
                state["env_review_required"] = False
                save_json(project_root / STATE_FILE_NAME, state)
                print_block("Отлично. Env сохранил, дальше сам выберешь что запускать из главного меню.")
                return
            correction_choice = prompt_choice("Что правим?", summary)
            key = prompted_keys[correction_choice - 1]
            while True:
                try:
                    values[key] = validators[key](prompt_text("Новое значение", default=values.get(key, "")))
                except ValueError as exc:
                    print(f"{exc}\n", flush=True)
                    continue
                break


def launch_bot(project_root: Path) -> None:
    cls()
    print("Запускаю бота...\n", flush=True)
    subprocess.run([*python_command(), BOT_ENTRYPOINT], cwd=str(project_root), check=False)


def launch_terminal_mode(project_root: Path, session_number: int | None = None) -> None:
    cls()
    if session_number is None:
        print("Запускаю терминальный режим и создаю новую сессию...\n", flush=True)
    else:
        print(f"Открываю terminal-сессию #{session_number}...\n", flush=True)
    command = [*python_command(), BOT_ENTRYPOINT, "--terminal-worker"]
    if session_number is not None:
        command.extend(["--session-number", str(session_number)])
    subprocess.run(
        command,
        cwd=str(project_root),
        check=False,
    )


def open_terminal_session_from_launcher(project_root: Path) -> None:
    cls()
    sessions = print_terminal_sessions_overview(project_root)
    if not sessions:
        pause()
        return

    available_numbers = {int(session["session_number"]) for session in sessions}
    print("0. Назад\n", flush=True)
    while True:
        raw_value = prompt_text(
            "Войти в сессию под номером",
            allow_empty=True,
        ).strip()
        if not raw_value or raw_value == "0":
            return
        session_number = parse_positive_int(raw_value)
        if session_number in available_numbers:
            launch_terminal_mode(project_root, session_number=session_number)
            return
        print("Такой сессии нет. Смотри на номера, а не на звёзды в небе.\n", flush=True)


def delete_terminal_session_from_launcher(project_root: Path) -> None:
    cls()
    sessions = print_terminal_sessions_overview(project_root)
    if not sessions:
        pause()
        return

    session_paths = {
        int(session["session_number"]): Path(session["path"])
        for session in sessions
    }
    print("0. Назад\n", flush=True)
    while True:
        raw_value = prompt_text(
            "Удалить сессию под номером",
            allow_empty=True,
        ).strip()
        if not raw_value or raw_value == "0":
            return
        session_number = parse_positive_int(raw_value)
        if session_number not in session_paths:
            print("Такой сессии нет. Удалять призраков пока не научились.\n", flush=True)
            continue
        session_paths[session_number].unlink(missing_ok=True)
        print(f"Сессию #{session_number} удалил.\n", flush=True)
        pause()
        return


def search_terminal_sessions_from_launcher(project_root: Path) -> None:
    cls()
    query = prompt_text("Поиск по названию/тегу/модели", allow_empty=True).strip()
    if not query:
        return
    matches = search_terminal_sessions(project_root, query)
    if not matches:
        print("По такому запросу сессий не нашёл. Видимо, память у тебя лучше, чем у этого поиска.\n", flush=True)
        pause()
        return
    print("Совпавшие сессии:", flush=True)
    for session in matches:
        tags = f" | теги: {', '.join(session['tags'])}" if session.get("tags") else ""
        print(
            f"#{session['session_number']} | {session['title']} | режим: {session.get('response_mode') or 'chat'}{tags}",
            flush=True,
        )
    print("", flush=True)
    pause()


def sessions_menu(project_root: Path) -> None:
    while True:
        cls()
        print_block(
            """
            Здесь лежат сохранённые terminal-сессии.
            Можно вернуться в старую переписку, поискать её по тегам/названию или снести к чёрту по номеру.
            """
        )
        choice = prompt_choice(
            "Сессии",
            [
                "Посмотреть сессии",
                "Поиск по сессиям",
                "Удалить сессию",
                "Назад",
            ],
        )
        if choice == 1:
            open_terminal_session_from_launcher(project_root)
        elif choice == 2:
            search_terminal_sessions_from_launcher(project_root)
        elif choice == 3:
            delete_terminal_session_from_launcher(project_root)
        else:
            return


def backups_menu(project_root: Path) -> None:
    while True:
        cls()
        history = list_update_history_records(project_root)
        updates = [item for item in history if item.get("type") == "update"]
        last_update = updates[-1] if updates else None
        print_block(
            f"""
            Бэкапы и откат.
            Последний апдейт: {(last_update.get('to_version') if last_update else 'ещё не записывался')}.
            Перед опасными движениями лучше иметь архив, а не только веру в лучшее.
            """
        )
        choice = prompt_choice(
            "Бэкапы и откат",
            [
                "Создать бэкап сейчас",
                "Посмотреть список бэкапов",
                "Откатить последнее обновление",
                "Назад",
            ],
        )
        if choice == 1:
            create_project_backup_from_launcher(project_root)
        elif choice == 2:
            show_project_backups(project_root)
        elif choice == 3:
            rollback_last_project_update(project_root)
        else:
            return


def resolve_local_launcher_path(project_root: Path, raw_value: Any) -> Path | None:
    normalized = normalize_path_input(raw_value)
    if not normalized:
        return None
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        return candidate.resolve()
    except Exception:
        return candidate


def prompt_existing_launcher_path(
    project_root: Path,
    prompt: str,
    *,
    require_directory: bool | None = None,
) -> Path | None:
    print("0. Назад\n", flush=True)
    while True:
        raw_value = prompt_text(prompt, allow_empty=True).strip()
        if not raw_value or raw_value == "0":
            return None
        path = resolve_local_launcher_path(project_root, raw_value)
        if path is None or not path.exists():
            print("Такого пути нет. Либо опечатался, либо пытаешься индексировать воздух.\n", flush=True)
            continue
        if require_directory is True and not path.is_dir():
            print("Нужна именно папка, а не очередной хитрый файл.\n", flush=True)
            continue
        if require_directory is False and not path.is_file():
            print("Тут нужен файл. Папка сейчас не прокатит.\n", flush=True)
            continue
        return path


def refresh_runtime_background_tasks(module: Any) -> None:
    try:
        module.background_tasks.clear()
    except Exception:
        pass
    try:
        module.background_task_order.clear()
    except Exception:
        pass
    try:
        module.load_background_tasks_from_disk()
    except Exception:
        return


def search_in_knowledge_base(project_root: Path) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    docs = module.list_knowledge_docs()
    if not docs:
        print("Локальная БЗ пока пустая. Искать в пустоте — сильный жанр, но бесполезный.\n", flush=True)
        pause()
        return
    query = prompt_text("Что ищем в локальной БЗ", allow_empty=True).strip()
    if not query:
        return
    matches = module.search_chunks_in_payloads(docs, query, limit=8)
    cls()
    if not matches:
        print("Совпадений не нашёл.\n", flush=True)
    else:
        print(module.render_context_matches("Совпадения по локальной БЗ:", matches, max_chars=6000), flush=True)
        print("", flush=True)
    pause()


def index_knowledge_source_now(project_root: Path) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    source_path = prompt_existing_launcher_path(project_root, "Путь к файлу или папке для БЗ")
    if source_path is None:
        return
    try:
        module.ensure_feature_roots()
        print_live_banner("Локальная БЗ", f"Индексирую {source_path.name}", progress=0.5)
        doc = module.build_knowledge_doc_payload(source_path)
        module.atomic_write_json(module.knowledge_doc_file_path(str(doc["doc_id"])), doc)
        print_live_banner("Локальная БЗ", f"{doc['title']} | чанков: {doc['chunk_count']}", progress=1.0, done=True, final_newline=True)
        print(
            "Источник закинул в локальную БЗ.\n"
            f"- ID: {str(doc['doc_id'])[:8]}\n"
            f"- Файлов: {doc['file_count']}\n"
            f"- Чанков: {doc['chunk_count']}\n",
            flush=True,
        )
    except Exception as exc:
        print_live_banner("Локальная БЗ", "Упало с ошибкой", progress=1.0, final_newline=True)
        print(f"Не удалось собрать БЗ: {exc}\n", flush=True)
    pause()


def queue_knowledge_source(project_root: Path) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    source_path = prompt_existing_launcher_path(project_root, "Путь к файлу или папке для очереди БЗ")
    if source_path is None:
        return
    try:
        task = module.create_background_task(
            kind="kb_ingest",
            owner_key="launcher",
            description=f"Импорт в локальную БЗ: {source_path.name}",
            payload={"source_path": str(source_path)},
        )
    except Exception as exc:
        print(f"Не удалось поставить задачу в очередь: {exc}\n", flush=True)
        pause()
        return
    print(
        "Задачу в очередь поставил.\n"
        f"- Task ID: {str(task.get('task_id') or '')[:8]}\n"
        "Подхватится, когда запустится бот или terminal worker. Сам лаунчер, сюрприз, задачи не исполняет.\n",
        flush=True,
    )
    pause()


def remove_knowledge_doc_from_launcher(project_root: Path) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    print(module.render_knowledge_docs_text() + "\n", flush=True)
    print("0. Назад\n", flush=True)
    query = prompt_text("Удалить документ по номеру, ID или куску пути", allow_empty=True).strip()
    if not query or query == "0":
        return
    doc = module.resolve_knowledge_doc(query)
    if doc is None:
        print("Такой документ не нашёл. БЗ пока ещё не научилась телепатии.\n", flush=True)
        pause()
        return
    doc_title = doc.get("title") or "без названия"
    if not prompt_yes_no(f"Удаляю '{doc_title}'?", default=False):
        print("Ок, документ пока жив.\n", flush=True)
        pause()
        return
    if module.delete_knowledge_doc(str(doc.get("doc_id") or "")):
        print("Документ удалил.\n", flush=True)
    else:
        print("Не удалось удалить документ. Возможно, он уже успел испариться.\n", flush=True)
    pause()


def clear_knowledge_base_from_launcher(project_root: Path) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    docs = module.list_knowledge_docs()
    if not docs:
        print("Локальная БЗ уже пустая.\n", flush=True)
        pause()
        return
    if not prompt_yes_no(f"Снести все {len(docs)} документов из локальной БЗ?", default=False):
        print("Ок, массовую зачистку отменил.\n", flush=True)
        pause()
        return
    removed = 0
    for doc in docs:
        if module.delete_knowledge_doc(str(doc.get("doc_id") or "")):
            removed += 1
    print(f"Очистил локальную БЗ. Улетело документов: {removed}.\n", flush=True)
    pause()


def knowledge_base_menu(project_root: Path) -> None:
    while True:
        cls()
        print_block(
            """
            Локальная БЗ живёт отдельно и не липнет к ответам, пока ты сам её не включишь через /kb on.
            То есть можно и с ней, и без неё — как нормальный человек, а не заложник фичи.
            """
        )
        choice = prompt_choice(
            "Локальная БЗ",
            [
                "Посмотреть документы",
                "Индексировать источник прямо сейчас",
                "Поставить импорт в очередь",
                "Поиск по локальной БЗ",
                "Удалить документ",
                "Очистить локальную БЗ",
                "Назад",
            ],
        )
        if choice == 1:
            cls()
            module = require_bot_runtime_module(project_root)
            if module is None:
                continue
            print(module.render_knowledge_docs_text() + "\n", flush=True)
            pause()
        elif choice == 2:
            index_knowledge_source_now(project_root)
        elif choice == 3:
            queue_knowledge_source(project_root)
        elif choice == 4:
            search_in_knowledge_base(project_root)
        elif choice == 5:
            remove_knowledge_doc_from_launcher(project_root)
        elif choice == 6:
            clear_knowledge_base_from_launcher(project_root)
        else:
            return


def search_in_projects(project_root: Path) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    projects = module.list_projects()
    if not projects:
        print("Индексов проектов пока нет.\n", flush=True)
        pause()
        return
    query = prompt_text("Что ищем по индексам проектов", allow_empty=True).strip()
    if not query:
        return
    matches = module.search_chunks_in_payloads(projects, query, limit=10)
    cls()
    if not matches:
        print("По проектам совпадений не нашёл.\n", flush=True)
    else:
        print(module.render_context_matches("Совпадения по проектам:", matches, max_chars=7000), flush=True)
        print("", flush=True)
    pause()


def scan_project_now(project_root: Path, *, existing_project: dict[str, Any] | None = None) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    default_path = str(existing_project.get("project_path") or "") if existing_project else None
    project_path = prompt_existing_launcher_path(
        project_root,
        "Путь к папке проекта",
        require_directory=True,
    )
    if project_path is None:
        return
    try:
        module.ensure_feature_roots()
        print_live_banner("Проекты", f"Сканирую {project_path.name}", progress=0.5)
        record = module.build_project_record_payload(project_path)
        if existing_project is not None:
            project_id = str(existing_project.get("project_id") or "")
            if project_id:
                record["project_id"] = project_id
            created_at = existing_project.get("created_at")
            if created_at:
                record["created_at"] = created_at
            title = str(existing_project.get("title") or "").strip()
            if title:
                record["title"] = title
            if hasattr(module, "iso_now"):
                record["updated_at"] = module.iso_now()
        module.atomic_write_json(module.project_record_file_path(str(record["project_id"])), record)
        print_live_banner("Проекты", f"{record['title']} | файлов: {record['file_count']}", progress=1.0, done=True, final_newline=True)
        print(
            "Индекс проекта готов.\n"
            f"- ID: {str(record['project_id'])[:8]}\n"
            f"- Файлов: {record['file_count']}\n"
            f"- Символов: {record['total_chars']}\n",
            flush=True,
        )
    except Exception as exc:
        print_live_banner("Проекты", "Скан упал", progress=1.0, final_newline=True)
        if default_path:
            print(f"Не удалось пересканировать проект {default_path}: {exc}\n", flush=True)
        else:
            print(f"Не удалось просканировать проект: {exc}\n", flush=True)
    pause()


def queue_project_scan(project_root: Path, *, existing_project: dict[str, Any] | None = None) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    project_path = prompt_existing_launcher_path(
        project_root,
        "Путь к папке проекта для очереди",
        require_directory=True,
    )
    if project_path is None:
        return
    payload = {"project_path": str(project_path)}
    if existing_project is not None:
        payload["replace_project_id"] = str(existing_project.get("project_id") or "")
    try:
        task = module.create_background_task(
            kind="project_scan",
            owner_key="launcher",
            description=f"Скан проекта: {project_path.name}",
            payload=payload,
        )
    except Exception as exc:
        print(f"Не удалось поставить скан проекта в очередь: {exc}\n", flush=True)
        pause()
        return
    print(
        "Скан проекта поставил в очередь.\n"
        f"- Task ID: {str(task.get('task_id') or '')[:8]}\n"
        "Запустится, когда поднимется живой bot/terminal worker. Лаунчер тут не грузчик, он только оформляет заявку.\n",
        flush=True,
    )
    pause()


def show_project_details(project_root: Path) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    print(module.render_projects_text() + "\n", flush=True)
    print("0. Назад\n", flush=True)
    query = prompt_text("Показать проект по номеру, ID или пути", allow_empty=True).strip()
    if not query or query == "0":
        return
    project = module.resolve_project_record(query)
    if project is None:
        print("Такой проект не нашёл.\n", flush=True)
        pause()
        return
    cls()
    print(module.render_project_record_detail(project) + "\n", flush=True)
    pause()


def delete_project_from_launcher(project_root: Path) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    print(module.render_projects_text() + "\n", flush=True)
    print("0. Назад\n", flush=True)
    query = prompt_text("Удалить проект по номеру, ID или пути", allow_empty=True).strip()
    if not query or query == "0":
        return
    project = module.resolve_project_record(query)
    if project is None:
        print("Такой проект не нашёл.\n", flush=True)
        pause()
        return
    project_title = project.get("title") or "без названия"
    if not prompt_yes_no(f"Удаляю индекс проекта '{project_title}'?", default=False):
        print("Ок, проектовый индекс пока оставил.\n", flush=True)
        pause()
        return
    if module.delete_project_record(str(project.get("project_id") or "")):
        print("Индекс проекта удалил.\n", flush=True)
    else:
        print("Не удалось удалить индекс проекта.\n", flush=True)
    pause()


def projects_menu(project_root: Path) -> None:
    while True:
        cls()
        print_block(
            """
            Здесь живут индексы проектов для режима /project и code mode.
            Сам исходник не трогаю, только собираю карту файлов и чанки контекста.
            """
        )
        choice = prompt_choice(
            "Проекты",
            [
                "Посмотреть индексы проектов",
                "Просканировать проект прямо сейчас",
                "Поставить скан проекта в очередь",
                "Поиск по индексам проектов",
                "Подробности проекта",
                "Пересканировать проект сейчас",
                "Удалить индекс проекта",
                "Назад",
            ],
        )
        if choice == 1:
            cls()
            module = require_bot_runtime_module(project_root)
            if module is None:
                continue
            print(module.render_projects_text() + "\n", flush=True)
            pause()
        elif choice == 2:
            scan_project_now(project_root)
        elif choice == 3:
            queue_project_scan(project_root)
        elif choice == 4:
            search_in_projects(project_root)
        elif choice == 5:
            show_project_details(project_root)
        elif choice == 6:
            cls()
            module = require_bot_runtime_module(project_root)
            if module is None:
                continue
            print(module.render_projects_text() + "\n", flush=True)
            print("0. Назад\n", flush=True)
            query = prompt_text("Пересканировать проект по номеру, ID или пути", allow_empty=True).strip()
            if not query or query == "0":
                continue
            project = module.resolve_project_record(query)
            if project is None:
                print("Такой проект не нашёл.\n", flush=True)
                pause()
                continue
            scan_project_now(project_root, existing_project=project)
        elif choice == 7:
            delete_project_from_launcher(project_root)
        else:
            return


def print_background_tasks_overview(project_root: Path) -> None:
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    refresh_runtime_background_tasks(module)
    print(module.render_background_tasks_text() + "\n", flush=True)
    tasks = module.list_background_tasks()
    if any(str(task.get("status") or "") == "queued" for task in tasks):
        print(
            "Queued-задачи оживут, когда будет запущен bot.py или terminal worker.\n",
            flush=True,
        )


def remove_background_task_from_launcher(project_root: Path) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    refresh_runtime_background_tasks(module)
    print(module.render_background_tasks_text() + "\n", flush=True)
    print("0. Назад\n", flush=True)
    query = prompt_text("Удалить задачу из истории по номеру или ID", allow_empty=True).strip()
    if not query or query == "0":
        return
    task = module.resolve_background_task(query)
    if task is None:
        print("Такую задачу не нашёл.\n", flush=True)
        pause()
        return
    status = str(task.get("status") or "")
    if status in {"queued", "running"}:
        print(
            "Живую задачу из лаунчера не выпиливаю. Дождись конца или пользуйся /tasks в самом боте.\n",
            flush=True,
        )
        pause()
        return
    module.delete_background_task(str(task.get("task_id") or ""))
    print("Задачу из истории удалил.\n", flush=True)
    pause()


def clear_finished_background_tasks(project_root: Path) -> None:
    cls()
    module = require_bot_runtime_module(project_root)
    if module is None:
        return
    refresh_runtime_background_tasks(module)
    tasks = module.list_background_tasks()
    removable_statuses = {"completed", "failed", "cancelled", "interrupted"}
    removable = [task for task in tasks if str(task.get("status") or "") in removable_statuses]
    if not removable:
        print("Чистить пока нечего. Очередь и так выглядит прилично.\n", flush=True)
        pause()
        return
    if not prompt_yes_no(f"Удаляю {len(removable)} завершённых/битых задач из истории?", default=False):
        print("Ок, хвосты пока оставил.\n", flush=True)
        pause()
        return
    for task in removable:
        module.delete_background_task(str(task.get("task_id") or ""))
    print(f"Историю подчистил. Улетело задач: {len(removable)}.\n", flush=True)
    pause()


def tasks_menu(project_root: Path) -> None:
    while True:
        cls()
        print_block(
            """
            Тут живёт очередь индексации БЗ и проектов.
            Лаунчер может ставить задачи и чистить историю, но сам worker из себя не строит.
            """
        )
        choice = prompt_choice(
            "Фоновые задачи",
            [
                "Посмотреть очередь и историю",
                "Удалить задачу из истории",
                "Очистить завершённые и битые задачи",
                "Назад",
            ],
        )
        if choice == 1:
            cls()
            print_background_tasks_overview(project_root)
            pause()
        elif choice == 2:
            remove_background_task_from_launcher(project_root)
        elif choice == 3:
            clear_finished_background_tasks(project_root)
        else:
            return


def apply_selected_model(project_root: Path, state: dict[str, Any], model_path: Path) -> None:
    model_path = handle_model_support(project_root, model_path)
    state["model_path"] = str(model_path)
    state["configured"] = False
    state["env_review_required"] = True

    llama_server_exe = resolve_existing_file_path(state.get("llama_server_exe"), project_root)
    llama_cpp_dir = resolve_existing_dir_path(state.get("llama_cpp_dir"), project_root)
    if llama_server_exe is None:
        llama_cpp_dir, llama_server_exe = ensure_llama_runtime(project_root)
        state["llama_server_exe"] = str(llama_server_exe)
    if llama_cpp_dir is None:
        llama_cpp_dir = llama_server_exe.parent
        state["llama_cpp_dir"] = str(llama_cpp_dir)

    save_json(project_root / STATE_FILE_NAME, state)
    order, values, hardware_profile = build_default_env(
        project_root,
        model_path,
        llama_cpp_dir,
        llama_server_exe,
    )
    write_env_file(project_root, values, order)
    print_linux_hardware_profile(hardware_profile)
    print("Модель обновил. На следующем шаге можно править env или сразу запускаться.\n", flush=True)
    pause()


def show_models_from_launcher(project_root: Path, state: dict[str, Any]) -> None:
    cls()
    current_model = resolve_current_model_path(project_root, state)
    if current_model is not None:
        print(
            "Текущая модель:\n"
            f"- {current_model.name}\n"
            f"- {format_model_size(current_model)}\n"
            f"- {current_model}\n",
            flush=True,
        )
    else:
        print("Текущая модель не выбрана или потерялась по пути.\n", flush=True)
    models = find_external_model_paths(project_root)
    if not models:
        print("Локальных `.gguf` моделей не нашёл.\n", flush=True)
        pause()
        return
    current_key = model_identity_key(current_model) if current_model is not None else ""
    print("Найденные модели:", flush=True)
    for index, model_path in enumerate(models, start=1):
        marker = " [текущая]" if model_identity_key(model_path) == current_key else ""
        print(
            f"{index}. {model_path.name}{marker} | {format_model_size(model_path)}\n"
            f"   {model_path}",
            flush=True,
        )
    print("", flush=True)
    pause()


def activate_existing_model_from_launcher(project_root: Path, state: dict[str, Any]) -> None:
    cls()
    models = find_external_model_paths(project_root)
    if not models:
        print("Локальные .gguf не нашёл. Без модели менеджер моделей, конечно, выглядит очень философски.\n", flush=True)
        pause()
        return
    options = [format_model_choice(model_path) for model_path in models]
    options.append("Назад")
    choice = prompt_choice("Выбери найденную модель", options)
    if choice == len(options):
        return
    apply_selected_model(project_root, state, models[choice - 1])


def get_systemd_context(service_name: str = SYSTEMD_SERVICE_NAME) -> tuple[list[str], Path, str] | None:
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return None

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return [systemctl], Path("/etc/systemd/system") / service_name, "system"

    return (
        [systemctl, "--user"],
        Path.home() / ".config" / "systemd" / "user" / service_name,
        "user",
    )


def build_systemd_service_text(project_root: Path, mode: str) -> str:
    python_bin = python_command()[0]
    python_bin = shutil.which(python_bin) or python_bin
    bot_path = project_root / BOT_ENTRYPOINT
    wanted_by = "multi-user.target" if mode == "system" else "default.target"

    return textwrap.dedent(
        f"""
        [Unit]
        Description=HeyMate Telegram bot
        After=network-online.target
        Wants=network-online.target
        StartLimitIntervalSec=300
        StartLimitBurst=20

        [Service]
        Type=simple
        WorkingDirectory={project_root}
        Environment=PYTHONUNBUFFERED=1
        Environment=HEYMATE_SERVICE_MODE=systemd
        ExecStart="{python_bin}" "{bot_path}" --supervisor-worker
        Restart=always
        RestartSec=5
        TimeoutStopSec=45
        KillMode=control-group
        SyslogIdentifier=heymate-bot
        StandardOutput=journal
        StandardError=journal

        [Install]
        WantedBy={wanted_by}
        """
    ).strip() + "\n"


def build_site_dashboard_systemd_service_text(project_root: Path, mode: str) -> str:
    python_bin = python_command()[0]
    python_bin = shutil.which(python_bin) or python_bin
    dashboard_path = project_root / SITE_DASHBOARD_ENTRYPOINT
    wanted_by = "multi-user.target" if mode == "system" else "default.target"
    settings = load_site_dashboard_settings(project_root)

    return textwrap.dedent(
        f"""
        [Unit]
        Description=HeyMate site dashboard
        After=network-online.target
        Wants=network-online.target
        StartLimitIntervalSec=300
        StartLimitBurst=20

        [Service]
        Type=simple
        WorkingDirectory={project_root}
        Environment=PYTHONUNBUFFERED=1
        Environment=SITE_DASHBOARD_HOST={settings['host']}
        Environment=SITE_DASHBOARD_PORT={settings['port']}
        Environment=SITE_DASHBOARD_REFRESH_SECONDS={settings['refresh_seconds']}
        ExecStart="{python_bin}" "{dashboard_path}"
        Restart=always
        RestartSec=5
        TimeoutStopSec=30
        KillMode=control-group
        SyslogIdentifier=heymate-site-dashboard
        StandardOutput=journal
        StandardError=journal

        [Install]
        WantedBy={wanted_by}
        """
    ).strip() + "\n"


def install_systemd_service(project_root: Path) -> None:
    cls()
    context = get_systemd_context()
    if context is None:
        print("systemctl не найден. На этой системе некуда ставить service.\n", flush=True)
        pause()
        return

    command_prefix, service_path, mode = context
    service_path.parent.mkdir(parents=True, exist_ok=True)
    service_path.write_text(
        build_systemd_service_text(project_root, mode),
        encoding="utf-8",
    )

    daemon_reload = subprocess.run([*command_prefix, "daemon-reload"], check=False)
    enable_now = subprocess.run(
        [*command_prefix, "enable", "--now", SYSTEMD_SERVICE_NAME],
        check=False,
    )

    if daemon_reload.returncode != 0 or enable_now.returncode != 0:
        print("Не удалось включить systemd service. Проверь вывод выше.\n", flush=True)
        pause()
        return

    print(f"Service установлен: {service_path}", flush=True)
    if mode == "user":
        print(
            "Если это headless Linux-сервер, может понадобиться loginctl enable-linger $USER.\n",
            flush=True,
        )
    pause()


def show_systemd_service_status() -> None:
    cls()
    context = get_systemd_context()
    if context is None:
        print("systemctl не найден.\n", flush=True)
        pause()
        return

    command_prefix, _, _ = context
    subprocess.run(
        [*command_prefix, "status", SYSTEMD_SERVICE_NAME, "--no-pager", "--full"],
        check=False,
    )
    print("", flush=True)
    pause()


def manage_systemd_service(action: str) -> None:
    cls()
    context = get_systemd_context()
    if context is None:
        print("systemctl не найден.\n", flush=True)
        pause()
        return

    command_prefix, _, _ = context
    completed = subprocess.run(
        [*command_prefix, action, SYSTEMD_SERVICE_NAME],
        check=False,
    )
    if completed.returncode == 0:
        print(f"systemd service action completed: {action}\n", flush=True)
    else:
        print(f"Не удалось выполнить action '{action}' для systemd service.\n", flush=True)
    pause()


def show_systemd_service_logs(project_root: Path) -> None:
    cls()
    context = get_systemd_context()
    journalctl = shutil.which("journalctl")

    if context is not None and journalctl:
        command_prefix, _, mode = context
        journal_command = [journalctl]
        if mode == "user":
            journal_command.append("--user")
        journal_command.extend(
            ["-u", SYSTEMD_SERVICE_NAME, "-n", "200", "--no-pager"]
        )
        subprocess.run(journal_command, check=False)
        print("", flush=True)
        pause()
        return

    log_dir = project_root / "bot_logs"
    runtime_log = log_dir / "runtime.log"
    supervisor_log = log_dir / "systemd_supervisor.log"
    llama_log = log_dir / "llama_server.log"
    printed = False

    if runtime_log.is_file():
        print("=== runtime.log ===", flush=True)
        print(runtime_log.read_text(encoding="utf-8", errors="ignore")[-12000:], flush=True)
        printed = True
    if supervisor_log.is_file():
        if printed:
            print("", flush=True)
        print("=== systemd_supervisor.log ===", flush=True)
        print(supervisor_log.read_text(encoding="utf-8", errors="ignore")[-12000:], flush=True)
        printed = True
    if llama_log.is_file():
        if printed:
            print("", flush=True)
        print("=== llama_server.log ===", flush=True)
        print(llama_log.read_text(encoding="utf-8", errors="ignore")[-12000:], flush=True)
        printed = True
    if not printed:
        print("Логи пока не найдены.\n", flush=True)
    pause()


def remove_systemd_service() -> None:
    cls()
    context = get_systemd_context()
    if context is None:
        print("systemctl не найден.\n", flush=True)
        pause()
        return

    command_prefix, service_path, _ = context
    subprocess.run(
        [*command_prefix, "disable", "--now", SYSTEMD_SERVICE_NAME],
        check=False,
    )
    if service_path.is_file():
        service_path.unlink()
    subprocess.run([*command_prefix, "daemon-reload"], check=False)
    print("systemd service удалён.\n", flush=True)
    pause()


def setup_package(project_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    cls()
    print_block(
        """
        Привет!
        Я Linux-установщик. На серверных Linux тоже можно жить нормально.
        """
    )
    ensure_python_dependencies(project_root)
    llama_dir, llama_server_exe = ensure_llama_runtime(project_root)
    model_path = choose_model_path(project_root)
    model_path = handle_model_support(project_root, model_path)
    state["configured"] = False
    state["env_review_required"] = True
    state["setup_done"] = True
    state["model_path"] = str(model_path)
    state["llama_cpp_dir"] = str(llama_dir)
    state["llama_server_exe"] = str(llama_server_exe)
    save_json(project_root / STATE_FILE_NAME, state)
    order, values, hardware_profile = build_default_env(project_root, model_path, llama_dir, llama_server_exe)
    write_env_file(project_root, values, order)
    print_linux_hardware_profile(hardware_profile)
    return state


def show_env(project_root: Path) -> None:
    cls()
    env_path = project_root / ENV_FILE_NAME
    if not env_path.is_file():
        print("Файл .env пока не найден.\n", flush=True)
    else:
        print(env_path.read_text(encoding="utf-8", errors="ignore"), flush=True)
    pause()


def edit_env(project_root: Path) -> None:
    state = load_json(project_root / STATE_FILE_NAME)
    configure_env(project_root, state)
    pause()


def model_menu(project_root: Path, state: dict[str, Any]) -> None:
    while True:
        cls()
        print_block(
            """
            Здесь живут локальные .gguf модели.
            Можно посмотреть найденные файлы, быстро активировать готовую модель или снести текущую к чёрту.
            """
        )
        choice = prompt_choice(
            "Менеджер моделей",
            [
                "Показать текущую и найденные модели",
                "Быстро выбрать найденную модель",
                "Сменить или скачать модель",
                "Fit-check и быстрый прогон модели",
                "Удалить текущую модель",
                "Назад",
            ],
        )
        if choice == 1:
            show_models_from_launcher(project_root, state)
        elif choice == 2:
            activate_existing_model_from_launcher(project_root, state)
        elif choice == 3:
            model_path = choose_model_path(project_root)
            apply_selected_model(project_root, state, model_path)
        elif choice == 4:
            run_model_fit_check_from_launcher(project_root, state)
        elif choice == 5:
            delete_current_model(project_root, state)
        else:
            return


def site_dashboard_menu(project_root: Path) -> None:
    while True:
        cls()
        status = get_site_dashboard_status(project_root)
        access_urls = status.get("access_urls") or [status["url"]]
        print_block(
            f"""
            Веб-панель управления всем этим хозяйством.
            Сейчас она {'работает' if status['running'] else 'остановлена'}.
            Слушает: {status['listen_url']}
            Адрес входа: {access_urls[0]}
            PID: {status.get('pid') or '-'}
            """
        )
        choice = prompt_choice(
            "Веб-панель",
            [
                "Запустить веб-панель в фоне",
                "Запустить и открыть в браузере",
                "Показать статус и адрес",
                "Показать лог веб-панели",
                "Сеть, HTTPS и reverse proxy",
                "Управление systemd service веб-панели",
                "Остановить веб-панель",
                "Назад",
            ],
        )
        if choice == 1:
            launch_site_dashboard(project_root, open_browser=False)
        elif choice == 2:
            launch_site_dashboard(project_root, open_browser=True)
        elif choice == 3:
            show_site_dashboard_status(project_root)
        elif choice == 4:
            show_site_dashboard_log(project_root)
        elif choice == 5:
            site_dashboard_network_menu(project_root)
        elif choice == 6:
            site_dashboard_systemd_menu(project_root)
        elif choice == 7:
            stop_site_dashboard(project_root)
        else:
            return


def site_dashboard_systemd_menu(project_root: Path) -> None:
    while True:
        cls()
        print_block(
            """
            Отдельный systemd-загон для веб-панели.
            Если хочешь, чтобы Flask-пульт жил сам по себе после выхода из SSH, тебе сюда.
            """
        )
        choice = prompt_choice(
            "systemd: веб-панель",
            [
                "Установить или обновить service веб-панели",
                "Запустить service веб-панели",
                "Остановить service веб-панели",
                "Перезапустить service веб-панели",
                "Посмотреть статус service веб-панели",
                "Посмотреть логи service веб-панели",
                "Удалить service веб-панели",
                "Назад",
            ],
        )
        if choice == 1:
            install_site_dashboard_systemd_service(project_root)
        elif choice == 2:
            manage_site_dashboard_systemd_service("start")
        elif choice == 3:
            manage_site_dashboard_systemd_service("stop")
        elif choice == 4:
            manage_site_dashboard_systemd_service("restart")
        elif choice == 5:
            show_site_dashboard_systemd_status()
        elif choice == 6:
            show_site_dashboard_systemd_logs(project_root)
        elif choice == 7:
            remove_site_dashboard_systemd_service()
        else:
            return


def systemd_menu(project_root: Path) -> None:
    while True:
        cls()
        print_block(
            """
            Ну что, полезли в systemd.
            Здесь собраны все действия с сервисом, чтобы главное меню не выглядело как панель лифта.
            """
        )
        choice = prompt_choice(
            "Управление systemd",
            [
                "Установить или обновить systemd service",
                "Запустить systemd service",
                "Остановить systemd service",
                "Перезапустить systemd service",
                "Посмотреть статус systemd service",
                "Посмотреть логи systemd/runtime",
                "Удалить systemd service",
                "Назад",
            ],
        )
        if choice == 1:
            install_systemd_service(project_root)
        elif choice == 2:
            manage_systemd_service("start")
        elif choice == 3:
            manage_systemd_service("stop")
        elif choice == 4:
            manage_systemd_service("restart")
        elif choice == 5:
            show_systemd_service_status()
        elif choice == 6:
            show_systemd_service_logs(project_root)
        elif choice == 7:
            remove_systemd_service()
        else:
            return


def launcher_menu(project_root: Path) -> None:
    state_path = project_root / STATE_FILE_NAME
    state = load_json(state_path)
    while True:
        cls()
        print_block(
            """
            Дарова! Linux-пакет на месте.
            Нужно запустить бота, веб-панель, ковырнуть БЗ, проекты, модели или просто полюбоваться env?
            """
        )
        choice = prompt_choice(
            "Главное меню",
            [
                "Запустить бота",
                "Запустить терминальный режим",
                "Сессии",
                "Локальная БЗ",
                "Проекты",
                "Фоновые задачи",
                "Веб-панель",
                "Проверить обновление",
                "Бэкапы и откат",
                "Логи ошибок",
                "Справка по порту",
                "Менеджер моделей",
                "Управление systemd",
                "Посмотреть env",
                "Изменить env",
                "Выход",
            ],
        )
        if choice == 1:
            launch_bot(project_root)
        elif choice == 2:
            launch_terminal_mode(project_root)
        elif choice == 3:
            sessions_menu(project_root)
        elif choice == 4:
            knowledge_base_menu(project_root)
        elif choice == 5:
            projects_menu(project_root)
        elif choice == 6:
            tasks_menu(project_root)
        elif choice == 7:
            site_dashboard_menu(project_root)
        elif choice == 8:
            cls()
            check_for_project_update(project_root, show_if_latest=True)
            pause()
        elif choice == 9:
            backups_menu(project_root)
        elif choice == 10:
            show_error_logs(project_root)
        elif choice == 11:
            show_port_manual()
        elif choice == 12:
            model_menu(project_root, state)
            state = load_json(state_path)
        elif choice == 13:
            systemd_menu(project_root)
        elif choice == 14:
            show_env(project_root)
        elif choice == 15:
            edit_env(project_root)
            state = load_json(state_path)
        else:
            return


def main() -> int:
    ensure_utf8_output()
    if not sys.stdin.isatty():
        print(
            "HeyMate expects an interactive terminal. Run it from a normal shell or SSH TTY.\n"
            "For background Linux usage, install the systemd service from an interactive session first.",
            flush=True,
        )
        return 1
    root = project_root()
    maybe_offer_project_update(root)
    state_path = root / STATE_FILE_NAME
    state = load_json(state_path)

    if not state.get("setup_done"):
        state = setup_package(root, state)

    env_ready, env_values = validate_existing_env(root, state)
    if env_ready and not state.get("configured") and not state.get("env_review_required"):
        state = mark_state_configured_from_env(root, state, env_values)

    if state.get("env_review_required") or not state.get("configured"):
        configure_env(root, state)
        state = load_json(state_path)

    launcher_menu(root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nОстановлено вручную.", flush=True)
