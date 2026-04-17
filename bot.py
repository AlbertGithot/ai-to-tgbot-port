"""Telegram bot backed by llama.cpp and a local GGUF model."""

from __future__ import annotations

import asyncio
import json
import locale
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import readline  # noqa: F401
except ImportError:
    readline = None

from bot_control_db import (
    bootstrap_from_interactions,
    get_setting,
    init_db,
    is_user_blocked,
    record_event as record_db_event,
    set_setting,
    upsert_user,
)

ENV_FILE_PATH = Path(__file__).resolve().parent / ".env"


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


load_env_file(ENV_FILE_PATH)

init_db()

try:
    from aiogram import Bot, Dispatcher, F, Router
    from aiogram.client.session.aiohttp import AiohttpSession
    from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError, TelegramUnauthorizedError
    from aiogram.filters import Command, CommandStart
    from aiogram.types import (
        CallbackQuery,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Message,
    )
    from aiogram.utils.chat_action import ChatActionSender
except ImportError as exc:
    raise SystemExit(
        "Не найден aiogram. Установи его: pip install aiogram"
    ) from exc

try:
    import aiohttp
except ImportError as exc:
    raise SystemExit(
        "Не найден aiohttp. Установи его: pip install aiohttp"
    ) from exc


def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_path(name: str, default: str) -> Path:
    raw_value = os.getenv(name, default).strip()
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return (ENV_FILE_PATH.parent / path).resolve()


def update_env_file_value(path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    output: list[str] = []
    replaced = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped and not stripped.startswith("#") and "=" in raw_line:
            current_key = raw_line.split("=", 1)[0].strip()
            if current_key == key:
                output.append(f"{key}={value}")
                replaced = True
                continue
        output.append(raw_line)

    if not replaced:
        output.append(f"{key}={value}")

    path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def resolve_model_candidate(raw_value: str | Path | None) -> Path | None:
    if raw_value in (None, ""):
        return None
    try:
        candidate = Path(str(raw_value)).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
    except Exception:
        return None
    if candidate.is_file() and candidate.suffix.lower() == ".gguf":
        return candidate
    return None


def iter_common_model_roots() -> list[Path]:
    roots: list[Path] = []

    env_root = os.getenv("MODEL_PATH", "").strip()
    if env_root:
        try:
            env_path_candidate = Path(env_root).expanduser()
            roots.append(env_path_candidate if env_path_candidate.is_absolute() else (PROJECT_ROOT / env_path_candidate))
            roots.append((env_path_candidate.parent if env_path_candidate.is_absolute() else (PROJECT_ROOT / env_path_candidate).parent))
        except Exception:
            pass

    selected_model = get_setting("selected_model_path", "")
    if selected_model:
        try:
            selected_candidate = Path(selected_model).expanduser()
            roots.append(selected_candidate if selected_candidate.is_absolute() else (PROJECT_ROOT / selected_candidate))
            roots.append((selected_candidate.parent if selected_candidate.is_absolute() else (PROJECT_ROOT / selected_candidate).parent))
        except Exception:
            pass

    roots.extend(
        [
            PROJECT_ROOT / "models",
            PROJECT_ROOT,
            PROJECT_ROOT.parent / "models",
            Path.home() / "models",
            Path.home() / "Downloads",
            Path.home() / ".cache" / "huggingface",
        ]
    )
    if os.name == "nt":
        roots.append(Path("C:/Models"))
    else:
        roots.extend(
            [
                Path("/opt/models"),
                Path("/srv/models"),
                Path("/var/lib/models"),
                Path("/usr/local/share/models"),
            ]
        )

    unique_roots: list[Path] = []
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
        unique_roots.append(resolved)
    return unique_roots


def score_model_candidate(path: Path) -> tuple[int, int, int, str]:
    name = path.name.lower()
    score = 0
    if name == "qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive-q5_k_m.gguf":
        score += 1000
    if "qwen" in name:
        score += 200
    if "qwen3.5" in name or "qwen35" in name:
        score += 120
    if "instruct" in name:
        score += 50
    if "coder" in name:
        score += 40
    if "q5_k_m" in name:
        score += 80
    elif "q6_k" in name:
        score += 70
    elif "q4_k_m" in name:
        score += 60
    elif "bf16" in name:
        score += 20
    if "uncensored" in name:
        score += 10

    try:
        stat = path.stat()
        size = int(stat.st_size)
        mtime = int(stat.st_mtime)
    except OSError:
        size = 0
        mtime = 0

    return score, size, mtime, name


def model_identity_key(path: Path) -> str:
    try:
        return str(path.resolve()).lower()
    except Exception:
        return str(path).lower()


def model_profile_for_path(model_path: Path) -> dict[str, Any]:
    name = model_path.name.lower()

    if "qwen" in name:
        chat_format = "qwen"
    elif any(marker in name for marker in ("mistral", "mixtral", "zephyr", "hermes", "openchat", "chatml")):
        chat_format = "chatml"
    else:
        chat_format = ""

    if any(marker in name for marker in ("coder", "code")):
        max_tokens = 4096
        temperature = 0.35
    elif any(marker in name for marker in ("instruct", "chat", "assistant", "it", "qwen", "llama", "mistral", "mixtral", "deepseek", "gemma", "phi")):
        max_tokens = 3072
        temperature = 0.5
    else:
        max_tokens = 2048
        temperature = 0.6

    if any(marker in name for marker in ("70b", "72b", "32b", "35b", "34b", "30b", "27b")):
        n_ctx = 32768
    else:
        n_ctx = 16384

    return {
        "chat_format": chat_format,
        "n_ctx": n_ctx,
        "max_tokens": max_tokens,
        "brief_max_tokens": 240,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.1,
    }


def find_external_model_path() -> Path | None:
    best_match: Path | None = None
    best_score: tuple[int, int, int, str] | None = None
    seen_paths: set[str] = set()

    for root in iter_common_model_roots():
        if root.is_file() and root.suffix.lower() == ".gguf":
            marker = model_identity_key(root)
            if marker in seen_paths:
                continue
            seen_paths.add(marker)
            score = score_model_candidate(root)
            if best_score is None or score > best_score:
                best_match = root
                best_score = score
            continue
        if not root.is_dir():
            continue

        seen_files = 0
        for candidate in root.rglob("*.gguf"):
            seen_files += 1
            if seen_files > 200:
                break
            marker = model_identity_key(candidate)
            if marker in seen_paths:
                continue
            seen_paths.add(marker)
            score = score_model_candidate(candidate)
            if best_score is None or score > best_score:
                best_match = candidate
                best_score = score

    return best_match


def list_available_model_paths() -> list[Path]:
    seen_paths: set[str] = set()
    models: list[Path] = []
    for root in iter_common_model_roots():
        if root.is_file() and root.suffix.lower() == ".gguf":
            marker = model_identity_key(root)
            if marker not in seen_paths:
                seen_paths.add(marker)
                models.append(root)
            continue
        if not root.is_dir():
            continue
        seen_files = 0
        for candidate in root.rglob("*.gguf"):
            seen_files += 1
            if seen_files > 400:
                break
            marker = model_identity_key(candidate)
            if marker in seen_paths:
                continue
            seen_paths.add(marker)
            models.append(candidate)
    models.sort(key=score_model_candidate, reverse=True)
    return models


def humanize_file_size(size_bytes: int) -> str:
    size = float(max(0, int(size_bytes)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def render_models_text() -> str:
    models = list_available_model_paths()
    if not models:
        return "Локальных `.gguf` моделей не нашёл."
    current_model = resolve_model_candidate(MODEL_PATH)
    current_key = model_identity_key(current_model) if current_model is not None else ""
    lines = ["Доступные модели:"]
    for index, model_path in enumerate(models, start=1):
        marker = " [текущая]" if model_identity_key(model_path) == current_key else ""
        try:
            size_text = humanize_file_size(model_path.stat().st_size)
        except OSError:
            size_text = "размер неизвестен"
        lines.append(f"{index}. {model_path.name}{marker} | {size_text}")
        lines.append(f"   {model_path}")
    return "\n".join(lines)


def resolve_model_path_selection(query: str) -> Path | None:
    token = str(query or "").strip().lower()
    if not token:
        return None
    direct_path = is_existing_local_path(token)
    if direct_path is not None and direct_path.suffix.lower() == ".gguf":
        return direct_path
    models = list_available_model_paths()
    if token.isdigit():
        index = int(token)
        if 1 <= index <= len(models):
            return models[index - 1]
    for model_path in models:
        lowered_name = model_path.name.lower()
        lowered_full = str(model_path).lower()
        if token == lowered_name or token in lowered_name or token in lowered_full:
            return model_path
    return None


async def activate_model_path(model_path: Path) -> Path:
    selected = persist_model_path(model_path)
    if LLAMA_SERVER_PROCESS is not None:
        await restart_llama_server(f"model switched to {selected.name}")
    return selected


def persist_model_path(model_path: Path) -> Path:
    global MODEL_PATH
    global MODEL_PROFILE
    global N_CTX
    global MAX_TOKENS
    global BRIEF_MAX_TOKENS
    global TEMPERATURE
    global TOP_P
    global TOP_K
    global REPEAT_PENALTY

    resolved = model_path.resolve()
    MODEL_PATH = resolved
    MODEL_PROFILE = model_profile_for_path(resolved)
    N_CTX = int(MODEL_PROFILE["n_ctx"])
    MAX_TOKENS = int(MODEL_PROFILE["max_tokens"])
    BRIEF_MAX_TOKENS = int(MODEL_PROFILE["brief_max_tokens"])
    TEMPERATURE = float(MODEL_PROFILE["temperature"])
    TOP_P = float(MODEL_PROFILE["top_p"])
    TOP_K = int(MODEL_PROFILE["top_k"])
    REPEAT_PENALTY = float(MODEL_PROFILE["repeat_penalty"])
    os.environ["MODEL_PATH"] = str(resolved)
    set_setting("selected_model_path", str(resolved))
    try:
        update_env_file_value(ENV_FILE_PATH, "MODEL_PATH", str(resolved))
        update_env_file_value(ENV_FILE_PATH, "CHAT_FORMAT", str(MODEL_PROFILE["chat_format"]))
        update_env_file_value(ENV_FILE_PATH, "N_CTX", str(MODEL_PROFILE["n_ctx"]))
        update_env_file_value(ENV_FILE_PATH, "MAX_TOKENS", str(MODEL_PROFILE["max_tokens"]))
        update_env_file_value(ENV_FILE_PATH, "BRIEF_MAX_TOKENS", str(MODEL_PROFILE["brief_max_tokens"]))
        update_env_file_value(ENV_FILE_PATH, "TEMPERATURE", str(MODEL_PROFILE["temperature"]))
        update_env_file_value(ENV_FILE_PATH, "TOP_P", str(MODEL_PROFILE["top_p"]))
        update_env_file_value(ENV_FILE_PATH, "TOP_K", str(MODEL_PROFILE["top_k"]))
        update_env_file_value(ENV_FILE_PATH, "REPEAT_PENALTY", str(MODEL_PROFILE["repeat_penalty"]))
    except Exception:
        logger.exception("Не удалось обновить параметры модели в .env")
    return resolved


def ensure_valid_model_path() -> Path:
    candidate = resolve_model_candidate(MODEL_PATH)
    if candidate is not None:
        if candidate != MODEL_PATH:
            return persist_model_path(candidate)
        return candidate

    db_candidate = resolve_model_candidate(get_setting("selected_model_path", ""))
    if db_candidate is not None:
        logger.warning("MODEL_PATH в .env битый. Подхватываю модель из базы: %s", db_candidate)
        return persist_model_path(db_candidate)

    discovered = find_external_model_path()
    if discovered is not None:
        logger.warning("MODEL_PATH в .env битый. Сам нашел живую модель: %s", discovered)
        return persist_model_path(discovered)

    raise RuntimeError(f"Не найден файл модели: {MODEL_PATH}")

BOT_TOKEN = env_str("BOT_TOKEN")
MODEL_PATH = env_path(
    "MODEL_PATH",
    get_setting("selected_model_path", "./models/model.gguf") or "./models/model.gguf",
)
MODEL_PROFILE = model_profile_for_path(MODEL_PATH)
LLAMA_CPP_DIR = env_path("LLAMA_CPP_DIR", "./llama.cpp")
LLAMA_SERVER_FILENAME = "llama-server.exe" if os.name == "nt" else "llama-server"
LLAMA_SERVER_EXE = env_path(
    "LLAMA_SERVER_EXE",
    str(LLAMA_CPP_DIR / LLAMA_SERVER_FILENAME),
)
SOURCE_URL = env_str(
    "SOURCE_URL",
    "https://github.com/AlbertGithot/ai-to-tgbot-port",
)
LLAMA_SERVER_HOST = env_str("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = env_int("LLAMA_SERVER_PORT", 8080)
LLAMA_SERVER_BASE_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
LLAMA_SERVER_CHAT_URL = f"{LLAMA_SERVER_BASE_URL}/v1/chat/completions"
LLAMA_SERVER_HEALTH_URL = f"{LLAMA_SERVER_BASE_URL}/health"
LLAMA_SERVER_MODELS_URL = f"{LLAMA_SERVER_BASE_URL}/v1/models"
LLAMA_SERVER_START_TIMEOUT = env_int("LLAMA_SERVER_START_TIMEOUT", 180)
TELEGRAM_REQUEST_TIMEOUT = max(10, env_int("TELEGRAM_REQUEST_TIMEOUT", 60))
TELEGRAM_STARTUP_RETRY_DELAY_SECONDS = max(1, env_int("TELEGRAM_STARTUP_RETRY_DELAY_SECONDS", 5))
TELEGRAM_POLLING_RETRY_DELAY_SECONDS = max(1, env_int("TELEGRAM_POLLING_RETRY_DELAY_SECONDS", 5))
LLAMA_SERVER_REASONING = env_str("LLAMA_SERVER_REASONING", "off")
LLAMA_SERVER_REASONING_BUDGET = env_int("LLAMA_SERVER_REASONING_BUDGET", 0)
LLAMA_SERVER_REASONING_FORMAT = env_str("LLAMA_SERVER_REASONING_FORMAT", "deepseek")
LLAMA_SERVER_AUTO_RESTART = env_bool("LLAMA_SERVER_AUTO_RESTART", True)
LLAMA_SERVER_MAX_RESTART_ATTEMPTS = max(0, env_int("LLAMA_SERVER_MAX_RESTART_ATTEMPTS", 1))
LLAMA_SERVER_RESTART_DELAY_SECONDS = max(1, env_int("LLAMA_SERVER_RESTART_DELAY_SECONDS", 2))

SYSTEM_PROMPT = (
    "Ты локальный ассистент Telegram. "
    "Отвечай только на русском языке. "
    "Показывай только готовый финальный ответ пользователю. "
    "Не повторяй вопрос пользователя. "
    "Не пиши строки вроде 'User input:', 'User asks:', 'Question:' или служебные пояснения."
)

# Если модель сама не подхватывает chat template из GGUF, впиши сюда формат
# вроде 'chatml', 'llama-2', 'mistral-instruct' и т.д. Иначе оставь None.
CHAT_FORMAT: str | None = env_str("CHAT_FORMAT", str(MODEL_PROFILE["chat_format"])) or None

MAX_HISTORY_MESSAGES = env_int("MAX_HISTORY_MESSAGES", 10)
MAX_HISTORY_ENTRY_CHARS = env_int("MAX_HISTORY_ENTRY_CHARS", 2500)
MAX_ACTIVE_DIALOGS = env_int("MAX_ACTIVE_DIALOGS", 200)
MAX_USER_TEXT_CHARS = env_int("MAX_USER_TEXT_CHARS", 6000)
MAX_MULTI_REQUEST_TEXT_CHARS = env_int("MAX_MULTI_REQUEST_TEXT_CHARS", 2000)
MAX_LOG_TEXT_CHARS = env_int("MAX_LOG_TEXT_CHARS", 6000)
MAX_MODEL_REPLY_CHARS = env_int("MAX_MODEL_REPLY_CHARS", 24000)

N_CTX = env_int("N_CTX", int(MODEL_PROFILE["n_ctx"]))
N_THREADS = env_int("N_THREADS", max(1, (os.cpu_count() or 4) - 1))
N_BATCH = env_int("N_BATCH", 512)
N_GPU_LAYERS = env_int("N_GPU_LAYERS", 0)

MAX_TOKENS = env_int("MAX_TOKENS", int(MODEL_PROFILE["max_tokens"]))
BRIEF_MAX_TOKENS = env_int("BRIEF_MAX_TOKENS", int(MODEL_PROFILE["brief_max_tokens"]))
TEMPERATURE = env_float("TEMPERATURE", float(MODEL_PROFILE["temperature"]))
TOP_P = env_float("TOP_P", float(MODEL_PROFILE["top_p"]))
TOP_K = env_int("TOP_K", int(MODEL_PROFILE["top_k"]))
REPEAT_PENALTY = env_float("REPEAT_PENALTY", float(MODEL_PROFILE["repeat_penalty"]))
STOP_STRINGS = ["<|eot_id|>", "<|end|>", "</s>"]

TELEGRAM_SEGMENT_LIMIT = env_int("TELEGRAM_SEGMENT_LIMIT", 3800)
SHOW_MODEL_RAW = env_bool("SHOW_MODEL_RAW", False)
USE_RAW_MODEL_REPLY = env_bool("USE_RAW_MODEL_REPLY", True)
ENABLE_REPAIR_PASS = env_bool("ENABLE_REPAIR_PASS", True)
AI_ENABLED = env_bool("AI_ENABLED", True)
THINKING_PLACEHOLDER_TEXT = "Подожди, я думаю...."
MAX_TRACKED_BOT_MESSAGES = env_int("MAX_TRACKED_BOT_MESSAGES", 80)
PROJECT_ROOT = Path(__file__).resolve().parent
LICENSE_FILE_PATH = PROJECT_ROOT / "LICENSE"
KNOWLEDGE_BASE_ROOT = PROJECT_ROOT / "local_kb"
PROJECT_CONTEXT_ROOT = PROJECT_ROOT / "project_contexts"
TASK_QUEUE_ROOT = PROJECT_ROOT / "task_queue"
WAIT_INDICATOR_INTERVAL_SECONDS = max(1, env_int("WAIT_INDICATOR_INTERVAL_SECONDS", 1))
LIVE_STATUS_BAR_WIDTH = max(14, env_int("LIVE_STATUS_BAR_WIDTH", 22))
LONG_THINK_ROOT = PROJECT_ROOT / "deep_think_jobs"
TERMINAL_SESSIONS_ROOT = PROJECT_ROOT / "terminal_sessions"
TERMINAL_SESSION_TITLE_CHARS = max(20, env_int("TERMINAL_SESSION_TITLE_CHARS", 80))
TERMINAL_CLIPBOARD_MAX_CHARS = max(1000, env_int("TERMINAL_CLIPBOARD_MAX_CHARS", 32000))
TERMINAL_CLIPBOARD_PREVIEW_CHARS = max(80, env_int("TERMINAL_CLIPBOARD_PREVIEW_CHARS", 220))
SUPERVISOR_RESTART_BASE_DELAY_SECONDS = max(
    1,
    env_int("SUPERVISOR_RESTART_BASE_DELAY_SECONDS", 3),
)
SUPERVISOR_RESTART_MAX_DELAY_SECONDS = max(
    SUPERVISOR_RESTART_BASE_DELAY_SECONDS,
    env_int("SUPERVISOR_RESTART_MAX_DELAY_SECONDS", 60),
)
SUPERVISOR_RESTART_WINDOW_SECONDS = max(
    SUPERVISOR_RESTART_MAX_DELAY_SECONDS,
    env_int("SUPERVISOR_RESTART_WINDOW_SECONDS", 180),
)
SUPERVISOR_STORM_THRESHOLD = max(1, env_int("SUPERVISOR_STORM_THRESHOLD", 5))
SUPERVISOR_STABLE_UPTIME_SECONDS = max(
    5,
    env_int("SUPERVISOR_STABLE_UPTIME_SECONDS", 45),
)
LONG_THINK_MIN_DURATION_SECONDS = max(1, env_int("LONG_THINK_MIN_DURATION_SECONDS", 1))
LONG_THINK_MAX_DURATION_SECONDS = max(
    LONG_THINK_MIN_DURATION_SECONDS,
    env_int("LONG_THINK_MAX_DURATION_SECONDS", 86400),
)
LONG_THINK_MAX_ITERATIONS = max(1, env_int("LONG_THINK_MAX_ITERATIONS", 24))
LONG_THINK_STEP_MAX_TOKENS = max(
    512, env_int("LONG_THINK_STEP_MAX_TOKENS", min(MAX_TOKENS, 4096))
)
LONG_THINK_FINAL_MAX_TOKENS = max(
    512, env_int("LONG_THINK_FINAL_MAX_TOKENS", min(MAX_TOKENS, 4096))
)
LONG_THINK_TEMPLATE_MAX_TOKENS = max(
    256, env_int("LONG_THINK_TEMPLATE_MAX_TOKENS", min(MAX_TOKENS, 1536))
)
LONG_THINK_PLAN_MAX_TOKENS = max(
    256, env_int("LONG_THINK_PLAN_MAX_TOKENS", min(MAX_TOKENS, 1024))
)
LONG_THINK_PLAN_MAX_REPLY_CHARS = max(
    800, env_int("LONG_THINK_PLAN_MAX_REPLY_CHARS", 6000)
)
LONG_THINK_MAX_MODEL_REPLY_CHARS = env_int("LONG_THINK_MAX_MODEL_REPLY_CHARS", 0)
LONG_THINK_PROGRESS_UPDATE_SECONDS = max(
    10, env_int("LONG_THINK_PROGRESS_UPDATE_SECONDS", 30)
)
LONG_THINK_METRICS_SAMPLE_SECONDS = max(
    5, env_int("LONG_THINK_METRICS_SAMPLE_SECONDS", 15)
)
LONG_THINK_FINAL_BUFFER_RATIO = min(
    0.45,
    max(0.05, env_float("LONG_THINK_FINAL_BUFFER_RATIO", 0.15)),
)
LONG_THINK_FINAL_BUFFER_MIN_SECONDS = max(
    60, env_int("LONG_THINK_FINAL_BUFFER_MIN_SECONDS", 900)
)
LONG_THINK_FINAL_BUFFER_MAX_SECONDS = max(
    LONG_THINK_FINAL_BUFFER_MIN_SECONDS,
    env_int("LONG_THINK_FINAL_BUFFER_MAX_SECONDS", 7200),
)
LONG_THINK_ITERATION_SAFETY_SECONDS = max(
    30, env_int("LONG_THINK_ITERATION_SAFETY_SECONDS", 120)
)
LONG_THINK_CONTEXT_CHARS = max(2000, env_int("LONG_THINK_CONTEXT_CHARS", 16000))
LONG_THINK_HISTORY_ITERATIONS = max(
    1, env_int("LONG_THINK_HISTORY_ITERATIONS", 6)
)
LONG_THINK_STATUS_LIMIT = max(1, env_int("LONG_THINK_STATUS_LIMIT", 8))
LONG_THINK_PLAN_KEEP_SECONDS = max(
    300, env_int("LONG_THINK_PLAN_KEEP_SECONDS", 3600)
)
KB_MAX_SOURCE_BYTES = max(16_384, env_int("KB_MAX_SOURCE_BYTES", 2_000_000))
KB_CHUNK_CHARS = max(300, env_int("KB_CHUNK_CHARS", 1500))
KB_CHUNK_OVERLAP_CHARS = max(50, env_int("KB_CHUNK_OVERLAP_CHARS", 180))
KB_MAX_MATCHES = max(1, env_int("KB_MAX_MATCHES", 4))
KB_CONTEXT_MAX_CHARS = max(1000, env_int("KB_CONTEXT_MAX_CHARS", 5000))
PROJECT_SCAN_MAX_FILES = max(10, env_int("PROJECT_SCAN_MAX_FILES", 250))
PROJECT_SCAN_MAX_FILE_BYTES = max(16_384, env_int("PROJECT_SCAN_MAX_FILE_BYTES", 1_500_000))
PROJECT_CONTEXT_MAX_MATCHES = max(1, env_int("PROJECT_CONTEXT_MAX_MATCHES", 6))
PROJECT_CONTEXT_MAX_CHARS = max(1000, env_int("PROJECT_CONTEXT_MAX_CHARS", 7000))
TASK_QUEUE_HISTORY_LIMIT = max(5, env_int("TASK_QUEUE_HISTORY_LIMIT", 60))
TASK_QUEUE_IDLE_SLEEP_SECONDS = max(1, env_int("TASK_QUEUE_IDLE_SLEEP_SECONDS", 2))
TASK_QUEUE_PROGRESS_SAVE_STEP = max(1, env_int("TASK_QUEUE_PROGRESS_SAVE_STEP", 5))
LOCAL_TEXT_FILE_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".php", ".rb", ".swift",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".html", ".htm", ".xml", ".css", ".scss", ".sql", ".sh", ".bash",
    ".dockerfile", ".makefile", ".gradle", ".txt", ".csv",
}
PROJECT_IGNORED_DIR_NAMES = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "node_modules", "dist", "build", ".next", ".nuxt",
    ".idea", ".vscode", "coverage", "local_kb", "project_contexts",
    "task_queue", "deep_think_jobs", "terminal_sessions", "bot_logs",
}
LONG_THINK_SYSTEM_PROMPT = (
    "Ты работаешь в режиме длительного размышления над одной большой задачей. "
    "На каждом проходе улучшай рабочую версию результата. "
    "Не описывай свой процесс, не пиши self-check, reasoning, анализ и служебные заметки. "
    "Верни только обновлённую рабочую версию результата. "
    "Если задача про код, возвращай полноценный рабочий код и нужную структуру. "
    "Если задача про текст, возвращай полноценный материал, а не план на будущее."
)
LONG_THINK_PLAN_SYSTEM_PROMPT = (
    "Ты оцениваешь, сколько времени стоит выделить на режим deepthink для локальной модели. "
    "Нужно прагматично прикинуть длительность, чтобы задача была выполнена полно и без обрыва. "
    "Ответь строго одним JSON-объектом без markdown, без пояснений вне JSON и без служебного мусора."
)
RESET_DIALOG_CALLBACK = "dialog:reset_clear"
SOURCE_CODE_CALLBACK = "show_source_code"
DEEPPLAN_CALLBACK_PREFIX = "deepplan"
DEEPPLAN_START_CALLBACK_PREFIX = f"{DEEPPLAN_CALLBACK_PREFIX}:start:"
DEEPPLAN_CANCEL_CALLBACK_PREFIX = f"{DEEPPLAN_CALLBACK_PREFIX}:cancel:"
INEEDMORE_CALLBACK_PREFIX = "ineedmore"
INEEDMORE_ACTION_EDIT = f"{INEEDMORE_CALLBACK_PREFIX}:edit"
INEEDMORE_ACTION_ADD = f"{INEEDMORE_CALLBACK_PREFIX}:add"
INEEDMORE_ACTION_REMOVE = f"{INEEDMORE_CALLBACK_PREFIX}:remove"
INEEDMORE_ACTION_CONFIRM = f"{INEEDMORE_CALLBACK_PREFIX}:confirm"
INEEDMORE_ACTION_CANCEL = f"{INEEDMORE_CALLBACK_PREFIX}:cancel"
INEEDMORE_ACTION_EDIT_BACK = f"{INEEDMORE_CALLBACK_PREFIX}:edit_back"
INEEDMORE_ACTION_EDIT_TOPIC = f"{INEEDMORE_CALLBACK_PREFIX}:edit_topic"
INEEDMORE_ACTION_EDIT_QUERY_PREFIX = f"{INEEDMORE_CALLBACK_PREFIX}:edit_query:"
MAX_MULTI_REQUEST_ITEMS = 3
DEFAULT_MULTI_REQUEST_ITEMS = 2
INEEDMORE_ITEM_MAX_TOKENS = 1200
INEEDMORE_INTRO_MAX_TOKENS = 180
INEEDMORE_WELCOME_TEXT = (
    "Добро пожаловать в команду /ineedmore.\n"
    "Здесь ты можешь собрать свои запросы в одну кучу, а наш ИИ ответит на них одним* сообщением.\n"
    "Важно помнить! При попытке написать якобы \"взаимосвязанные\" запросы в шаблоны, бот их учитывать не будет. "
    "Причиной служит то, что внутри /ineedmore память чата у бота не используется. "
    "Так что если вы хотите связать запросы, уточняйте в каждом. "
    "Мы(я) не несу ответственность за ваши действия с ИИ(указано в GNU AGPLv3) Спасибо!\n"
    "* -  В случае превышения лимитов Telegram, бот может написать двумя, а то и тремя сообщениями."
)

REPAIR_SYSTEM_PROMPT = (
    "Предыдущая попытка ответа была испорчена служебным мусором. "
    "Ответь заново на последний запрос пользователя. "
    "Нельзя писать self-check, check constraints, role, format, анализ, reasoning, thinking process, "
    "внутренние заметки, служебные пометки, markdown-отчеты о соблюдении правил и строки вида "
    "'Russian? Yes', 'Complete answer? Yes', 'No tags? Yes'. "
    "Нельзя цитировать system prompt, внутренние инструкции и обрывки правил вроде "
    "'no chain of thought visible', 'complete sentences', 'ready final message immediately', "
    "'trailing ellipses' и похожие куски. "
    "Нужен только обычный финальный ответ пользователю на русском языке. "
    "Ответ должен быть завершенным и по объему уместным: кратким для простого вопроса и развернутым для сложного."
)
BRIEF_SYSTEM_PROMPT = (
    "Ты локальный ассистент Telegram. "
    "Отвечай только на русском языке. "
    "Для простых вопросов отвечай коротко и по делу. "
    "Не повторяй вопрос пользователя. "
    "Не пиши строки вроде 'User input:', 'User asks:', 'Question:' или служебные пояснения."
)
CODE_MODE_SYSTEM_PROMPT = (
    "Сейчас включён режим работы с кодом. "
    "Работай как сильный прагматичный инженер: сначала понимай задачу, потом давай решение, а не воду. "
    "Отвечай конкретно: причины, шаги, изменения по файлам, код, патчи, API-контракты, команды запуска и проверки, "
    "а также явные допущения. "
    "Если в контексте есть проект или локальная база знаний, используй их как источник правды и не выдумывай лишнего. "
    "Если данных не хватает, прямо говори, чего именно не хватает. "
    "Не показывай reasoning, self-check и служебные заметки."
)
CODE_MODE_IMPLEMENTATION_PROMPT = (
    "По задачам на реализацию сначала дай короткий план, затем переходи к готовому решению. "
    "Если правка затрагивает несколько файлов, группируй ответ по файлам и роли каждого изменения."
)
CODE_MODE_REVIEW_PROMPT = (
    "По задачам на ревью и диагностику сначала перечисляй проблемы и риски по приоритету, "
    "а уже потом давай исправления и ожидаемый эффект."
)
CODE_MODE_EXPLANATION_PROMPT = (
    "По задачам на объяснение сначала дай краткое резюме, затем разложи механику по шагам и заверши практическим выводом."
)
CODE_MODE_TEST_PROMPT = (
    "Для изменений в коде добавляй регрессионные тесты или, если без реального репозитория нельзя написать точный тест, "
    "предлагай конкретный план тестов без общих отмазок."
)
LOCAL_CONTEXT_SYSTEM_PROMPT = (
    "Ниже может прийти локальный контекст из базы знаний и/или проекта пользователя. "
    "Если он есть, считай его приоритетным источником фактов. "
    "Если контекст неполный, честно помечай допущения и не делай вид, что видел больше, чем реально было передано."
)

BRIEF_REPLY_STYLE_PROMPT = (
    "Это простой вопрос. "
    "Ответь коротко, естественно и сразу по делу. "
    "Только финальный ответ."
)
BRIEF_REPLY_MAX_CHARS = 320
PROMPT_SNAPSHOT_SIGNATURE = "v8-code-mode-contract"

LOG_DIR = Path("bot_logs")
RUNTIME_LOG_PATH = LOG_DIR / "runtime.log"
SUPERVISOR_LOG_PATH = LOG_DIR / "systemd_supervisor.log"
INTERACTIONS_LOG_PATH = LOG_DIR / "interactions.jsonl"
MODEL_RUNTIME_LOCK_PATH = LOG_DIR / "model_runtime.lock"
ERROR_LOG_PREVIEW_CHARS = max(2000, env_int("ERROR_LOG_PREVIEW_CHARS", 12000))
ERROR_LOG_STATUS_LIMIT = max(1, env_int("ERROR_LOG_STATUS_LIMIT", 3))
LICENSE_CALLBACK = "show_license"
logger = logging.getLogger("telegram_llama_bot")
router = Router()
console_live_lock = threading.RLock()
console_live_text = ""
console_live_render_len = 0
readline_prefill_lock = threading.Lock()
readline_prefill_text = ""

dialog_histories: dict[str, deque[dict[str, str]]] = {}
chat_locks: dict[int, asyncio.Lock] = {}
multi_request_sessions: dict[str, dict[str, Any]] = {}
bot_response_message_ids: dict[str, deque[int]] = {}
dialog_prompt_snapshots: dict[str, str] = {}
dialog_activity_order: OrderedDict[str, None] = OrderedDict()
dialog_runtime_settings: dict[str, dict[str, Any]] = {}
long_think_jobs: dict[str, dict[str, Any]] = {}
long_think_job_order: OrderedDict[str, None] = OrderedDict()
pending_long_think_plans: dict[str, dict[str, Any]] = {}
background_tasks: dict[str, dict[str, Any]] = {}
background_task_order: OrderedDict[str, None] = OrderedDict()
background_task_worker: asyncio.Task[Any] | None = None
background_task_event: asyncio.Event | None = None
model_lock: asyncio.Lock | None = None
model_pending_requests = 0
model_active_request_label: str | None = None
model_active_started_at: datetime | None = None
LLAMA_SERVER_PROCESS: subprocess.Popen[str] | None = None
LLAMA_SERVER_LOG_HANDLE: Any | None = None


def iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def parse_positive_int(raw_value: Any) -> int | None:
    text = str(raw_value or "").strip()
    if not text.isdigit():
        return None
    value = int(text)
    if value < 1:
        return None
    return value


def clamp_terminal_session_char_limit(raw_value: Any) -> int | None:
    value = parse_positive_int(raw_value)
    if value is None or value > 10000:
        return None
    return value


def coerce_nonnegative_int(raw_value: Any, default: int = 0) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(0, value)


def ensure_terminal_sessions_root() -> None:
    TERMINAL_SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)


def get_terminal_session_dialog_key(session_number: int) -> str:
    return f"terminal:session:{session_number}"


def get_terminal_session_file_path(session_number: int) -> Path:
    return TERMINAL_SESSIONS_ROOT / f"{session_number:04d}.json"


def list_terminal_session_numbers() -> list[int]:
    if not TERMINAL_SESSIONS_ROOT.is_dir():
        return []
    numbers: set[int] = set()
    for path in TERMINAL_SESSIONS_ROOT.glob("*.json"):
        session_number = parse_positive_int(path.stem)
        if session_number is not None:
            numbers.add(session_number)
    return sorted(numbers)


def normalize_terminal_session_history(raw_history: Any) -> list[dict[str, str]]:
    if not isinstance(raw_history, list):
        return []

    history: list[dict[str, str]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": trim_history_text(content)})
    return history


def build_terminal_session_title(text: str) -> str:
    collapsed = " ".join(text.strip().split())
    if not collapsed:
        return "Без названия"
    return truncate_text(collapsed, TERMINAL_SESSION_TITLE_CHARS)


def normalize_terminal_clipboard_text(raw_text: Any) -> str:
    text = str(raw_text or "").replace("\x00", "").strip()
    if not text:
        return ""
    return text[:TERMINAL_CLIPBOARD_MAX_CHARS]


def build_terminal_clipboard_preview(text: Any) -> str:
    normalized = normalize_terminal_clipboard_text(text)
    if not normalized:
        return "пусто"
    return summarize_long_think_iteration(
        normalized,
        max_chars=TERMINAL_CLIPBOARD_PREVIEW_CHARS,
    )


def ensure_feature_roots() -> None:
    for root in (KNOWLEDGE_BASE_ROOT, PROJECT_CONTEXT_ROOT, TASK_QUEUE_ROOT):
        root.mkdir(parents=True, exist_ok=True)


def read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def is_existing_local_path(raw_value: str) -> Path | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
    except Exception:
        return None
    if not candidate.exists():
        return None
    return candidate


def normalize_dialog_runtime_settings(raw_settings: Any = None) -> dict[str, Any]:
    payload = raw_settings if isinstance(raw_settings, dict) else {}
    response_mode = str(payload.get("response_mode") or "chat").strip().lower()
    if response_mode not in {"chat", "code"}:
        response_mode = "chat"
    active_project_id = str(payload.get("active_project_id") or "").strip()
    kb_enabled_raw = payload.get("kb_enabled")
    if kb_enabled_raw is None:
        kb_enabled = True
    else:
        kb_enabled = bool(kb_enabled_raw)
    return {
        "response_mode": response_mode,
        "active_project_id": active_project_id,
        "kb_enabled": kb_enabled,
    }


def get_dialog_runtime_settings(dialog_key: str) -> dict[str, Any]:
    settings = dialog_runtime_settings.get(dialog_key)
    if settings is None:
        settings = normalize_dialog_runtime_settings()
        dialog_runtime_settings[dialog_key] = settings
    return settings


def set_dialog_response_mode(dialog_key: str, mode: str) -> str:
    normalized = "code" if str(mode).strip().lower() == "code" else "chat"
    settings = get_dialog_runtime_settings(dialog_key)
    settings["response_mode"] = normalized
    return normalized


def set_dialog_kb_enabled(dialog_key: str, enabled: bool) -> bool:
    settings = get_dialog_runtime_settings(dialog_key)
    settings["kb_enabled"] = bool(enabled)
    return bool(settings["kb_enabled"])


def set_dialog_active_project(dialog_key: str, project_id: str | None) -> str:
    settings = get_dialog_runtime_settings(dialog_key)
    settings["active_project_id"] = str(project_id or "").strip()
    return str(settings["active_project_id"])


def build_dialog_runtime_settings_text(dialog_key: str) -> str:
    settings = get_dialog_runtime_settings(dialog_key)
    project_id = str(settings.get("active_project_id") or "").strip()
    project_label = "нет"
    if project_id:
        project = get_project_record(project_id)
        if project is None:
            project_label = project_id[:8]
        else:
            project_label = f"{project.get('title') or '-'} [{project_id[:8]}]"
    lines = [
        "Настройки текущего диалога:",
        f"- режим: {settings.get('response_mode')}",
        f"- локальная БЗ: {'вкл' if settings.get('kb_enabled') else 'выкл'}",
        f"- активный проект: {project_label}",
    ]
    if str(settings.get("response_mode") or "") == "code":
        lines.append("- профиль code mode: план, изменения по файлам, тесты и команды проверки")
    return "\n".join(lines)


def knowledge_doc_file_path(doc_id: str) -> Path:
    ensure_feature_roots()
    return KNOWLEDGE_BASE_ROOT / f"doc_{doc_id}.json"


def project_record_file_path(project_id: str) -> Path:
    ensure_feature_roots()
    return PROJECT_CONTEXT_ROOT / f"project_{project_id}.json"


def background_task_file_path(task_id: str) -> Path:
    ensure_feature_roots()
    return TASK_QUEUE_ROOT / f"task_{task_id}.json"


def normalize_local_context_text(text: str, max_chars: int) -> str:
    cleaned = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    return cleaned[:max_chars]


def normalize_code_mode_request_text(text: str) -> str:
    cleaned = normalize_local_context_text(text, max_chars=max(MAX_USER_TEXT_CHARS, 1200)).lower()
    cleaned = cleaned.replace("ё", "е")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def code_request_contains_any(normalized_text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in normalized_text for phrase in phrases)


def analyze_code_request(user_text: str) -> dict[str, Any]:
    normalized = normalize_code_mode_request_text(user_text)
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_./#+-]+", normalized)
    word_count = len(words)

    wants_write = code_request_contains_any(
        normalized,
        (
            "напиши", "создаи", "создай", "реализуи", "реализуй", "добавь", "собери",
            "сгенерируи", "сгенерируй", "сделаи", "сделай", "подготовь", "с нуля",
        ),
    )
    wants_fix = code_request_contains_any(
        normalized,
        (
            "исправь", "почини", "fix", "bug", "ошибк", "не работает", "сломал",
            "падает", "краш", "утечк", "broken", "не запускается",
        ),
    )
    wants_review = code_request_contains_any(
        normalized,
        (
            "ревью", "review", "проверь код", "code review", "что не так",
            "найди проблем", "найди баг", "аудит", "оцени код", "риски",
        ),
    )
    wants_explain = code_request_contains_any(
        normalized,
        (
            "объясни", "разбери", "как работает", "почему", "что делает",
            "за что отвечает", "поясни", "расскажи про код",
        ),
    )
    wants_refactor = code_request_contains_any(
        normalized,
        (
            "рефактор", "перепиши", "упрости", "оптимизируи", "оптимизируй",
            "улучши", "перестрои", "перестрой", "cleanup", "refactor",
        ),
    )
    wants_tests = code_request_contains_any(
        normalized,
        (
            "тест", "pytest", "unittest", "регрес", "покрытие", "coverage",
            "проверки", "e2e", "интеграц", "смоук",
        ),
    )
    wants_patch = code_request_contains_any(
        normalized,
        (
            "patch", "diff", "патч", "дифф", "unified diff",
        ),
    )
    wants_commands = code_request_contains_any(
        normalized,
        (
            "команд", "запуск", "run", "docker", "docker compose", "make ",
            "как запустить", "как проверить", "cli", "shell",
        ),
    )
    wants_architecture = code_request_contains_any(
        normalized,
        (
            "архитектур", "api", "контракт", "схем", "структур", "дизайн",
            "проект", "сервис", "микросервис", "backend", "frontend",
        ),
    )
    wants_files = code_request_contains_any(
        normalized,
        (
            "файл", "папк", "структур", "модул", ".py", ".js", ".ts", ".tsx",
            ".jsx", ".go", ".rs", ".java", ".kt", ".json", ".yml", ".yaml",
            "dockerfile", "makefile", "readme",
        ),
    )
    mentions_existing_code = code_request_contains_any(
        normalized,
        (
            "в проекте", "в репозитории", "в кодовои базе", "в кодовой базе",
            "в этом коде", "в существующем коде", "в приложении", "в сервисе",
            "в текущем коде",
        ),
    )

    complexity_score = 0
    for flag in (
        wants_write,
        wants_fix,
        wants_review,
        wants_refactor,
        wants_tests,
        wants_patch,
        wants_architecture,
        wants_files,
        wants_commands,
    ):
        complexity_score += int(flag)

    if mentions_existing_code:
        complexity_score += 1
    if word_count >= 20:
        complexity_score += 1
    if word_count >= 45:
        complexity_score += 1
    if code_request_contains_any(
        normalized,
        (
            "полностью", "целиком", "большои проект", "большой проект", "несколько модул",
            "5000 строк", "5к строк", "production", "прод", "полныи", "полный",
        ),
    ):
        complexity_score += 2

    if complexity_score >= 7:
        complexity = "large"
    elif complexity_score >= 3:
        complexity = "medium"
    else:
        complexity = "small"

    return {
        "normalized": normalized,
        "word_count": word_count,
        "complexity": complexity,
        "wants_write": wants_write,
        "wants_fix": wants_fix,
        "wants_review": wants_review,
        "wants_explain": wants_explain,
        "wants_refactor": wants_refactor,
        "wants_tests": wants_tests,
        "wants_patch": wants_patch,
        "wants_commands": wants_commands,
        "wants_architecture": wants_architecture,
        "wants_files": wants_files,
        "mentions_existing_code": mentions_existing_code,
    }


def summarize_project_extension_counts(project_record: dict[str, Any], *, limit: int = 6) -> str:
    counts: dict[str, int] = {}
    for item in project_record.get("files") or []:
        path_text = str(item.get("path") or "").strip()
        if not path_text:
            continue
        path = Path(path_text)
        suffix = path.suffix.lower()
        if suffix:
            label = suffix
        else:
            label = path.name.lower() or "без расширения"
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    pairs = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{name} x{amount}" for name, amount in pairs[:limit])


def build_project_overview_prompt(project_record: dict[str, Any], *, max_chars: int = 2600) -> str:
    files = project_record.get("files") or []
    lines = [
        "Обзор активного проекта:",
        f"- id: {str(project_record.get('project_id') or '')[:8]}",
        f"- название: {project_record.get('title') or '-'}",
        f"- путь: {project_record.get('project_path') or '-'}",
        f"- файлов в индексе: {project_record.get('file_count', 0)}",
        f"- символов в индексе: {project_record.get('total_chars', 0)}",
    ]
    extension_summary = summarize_project_extension_counts(project_record)
    if extension_summary:
        lines.append(f"- стек по файлам: {extension_summary}")
    if files:
        lines.append("Ключевые файлы:")
        ranked_files = sorted(
            files,
            key=lambda item: (-int(item.get("chars") or 0), str(item.get("path") or "")),
        )
        for item in ranked_files[:10]:
            preview = normalize_local_context_text(
                str(item.get("preview") or ""),
                max_chars=140,
            )
            line = (
                f"- {item.get('path') or '-'} | chars={int(item.get('chars') or 0)} "
                f"| chunks={int(item.get('chunk_count') or 0)}"
            )
            if preview:
                line += f" | {preview}"
            lines.append(line)
    return normalize_local_context_text("\n".join(lines), max_chars=max_chars)


def looks_like_binary_bytes(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    sample = data[: min(len(data), 4096)]
    bad = sum(1 for byte in sample if byte < 9 or (13 < byte < 32))
    return bad > max(8, len(sample) // 20)


def strip_html_markup(text: str) -> str:
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def is_supported_local_text_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in LOCAL_TEXT_FILE_EXTENSIONS:
        return True
    if name in {"dockerfile", "makefile"}:
        return True
    return False


def read_local_text_file(path: Path, *, max_bytes: int) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        logger.exception("Не удалось прочитать локальный файл %s", path)
        return ""
    if len(data) > max_bytes:
        data = data[:max_bytes]
    if looks_like_binary_bytes(data):
        return ""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="ignore")

    suffix = path.suffix.lower()
    if suffix in {".html", ".htm", ".xml"}:
        text = strip_html_markup(text)
    elif suffix == ".json":
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return normalize_local_context_text(text, max_chars=max_bytes * 2)


def iter_source_text_files(
    source_path: Path,
    *,
    max_files: int,
    ignored_dir_names: set[str] | None = None,
) -> list[Path]:
    ignored = {name.lower() for name in (ignored_dir_names or set())}
    if source_path.is_file():
        return [source_path] if is_supported_local_text_file(source_path) else []

    files: list[Path] = []
    try:
        for root, dir_names, file_names in os.walk(source_path):
            dir_names[:] = [
                name for name in dir_names
                if name.lower() not in ignored and not name.startswith(".git")
            ]
            current_root = Path(root)
            for file_name in sorted(file_names):
                candidate = current_root / file_name
                if not is_supported_local_text_file(candidate):
                    continue
                files.append(candidate)
                if len(files) >= max_files:
                    return files
    except Exception:
        logger.exception("Не удалось просканировать папку %s", source_path)
    return files


def split_text_into_context_chunks(
    text: str,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[str]:
    normalized = normalize_local_context_text(text, max_chars=max(len(text), chunk_chars))
    if not normalized:
        return []
    if len(normalized) <= chunk_chars:
        return [normalized]

    step = max(50, chunk_chars - overlap_chars)
    chunks: list[str] = []
    cursor = 0
    while cursor < len(normalized):
        chunk = normalized[cursor : cursor + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        cursor += step
        if len(chunks) >= 500:
            break
    return chunks


def build_context_chunks_for_files(
    files: list[Path],
    *,
    source_root: Path | None,
    max_bytes_per_file: int,
    chunk_chars: int,
    overlap_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    file_records: list[dict[str, Any]] = []

    for file_index, file_path in enumerate(files, start=1):
        text = read_local_text_file(file_path, max_bytes=max_bytes_per_file)
        if not text.strip():
            continue
        try:
            relative_path = str(file_path.relative_to(source_root)) if source_root is not None else file_path.name
        except Exception:
            relative_path = file_path.name
        split_chunks = split_text_into_context_chunks(
            text,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
        )
        if not split_chunks:
            continue
        file_records.append(
            {
                "path": relative_path,
                "absolute_path": str(file_path),
                "chars": len(text),
                "chunk_count": len(split_chunks),
                "preview": summarize_long_think_iteration(text, max_chars=220),
            }
        )
        for chunk_index, chunk_text in enumerate(split_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": f"{file_index}-{chunk_index}",
                    "path": relative_path,
                    "text": chunk_text,
                }
            )
    return chunks, file_records


def build_knowledge_doc_payload(source_path: Path) -> dict[str, Any]:
    resolved = source_path.resolve()
    files = iter_source_text_files(resolved, max_files=PROJECT_SCAN_MAX_FILES)
    if not files:
        raise RuntimeError("Не нашёл ни одного подходящего текстового файла для индексации.")
    chunks, file_records = build_context_chunks_for_files(
        files,
        source_root=resolved if resolved.is_dir() else resolved.parent,
        max_bytes_per_file=KB_MAX_SOURCE_BYTES,
        chunk_chars=KB_CHUNK_CHARS,
        overlap_chars=KB_CHUNK_OVERLAP_CHARS,
    )
    if not chunks:
        raise RuntimeError("Текст для локальной БЗ пустой или нечитабельный.")
    doc_id = uuid.uuid4().hex
    return {
        "doc_id": doc_id,
        "title": resolved.name,
        "source_path": str(resolved),
        "source_type": "dir" if resolved.is_dir() else "file",
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "chunk_count": len(chunks),
        "file_count": len(file_records),
        "files": file_records,
        "chunks": chunks,
    }


def build_project_record_payload(project_path: Path) -> dict[str, Any]:
    resolved = project_path.resolve()
    if not resolved.is_dir():
        raise RuntimeError("Проект должен быть папкой, а не отдельным файлом.")
    files = iter_source_text_files(
        resolved,
        max_files=PROJECT_SCAN_MAX_FILES,
        ignored_dir_names=PROJECT_IGNORED_DIR_NAMES,
    )
    if not files:
        raise RuntimeError("В проекте не нашёл ни одного подходящего текстового файла.")
    chunks, file_records = build_context_chunks_for_files(
        files,
        source_root=resolved,
        max_bytes_per_file=PROJECT_SCAN_MAX_FILE_BYTES,
        chunk_chars=KB_CHUNK_CHARS,
        overlap_chars=KB_CHUNK_OVERLAP_CHARS,
    )
    if not chunks:
        raise RuntimeError("Проектовый контекст пустой: читать там особо нечего.")
    total_chars = sum(int(item.get("chars") or 0) for item in file_records)
    project_id = uuid.uuid4().hex
    return {
        "project_id": project_id,
        "title": resolved.name,
        "project_path": str(resolved),
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "file_count": len(file_records),
        "total_chars": total_chars,
        "files": file_records,
        "chunks": chunks,
    }


def list_knowledge_docs() -> list[dict[str, Any]]:
    ensure_feature_roots()
    docs: list[dict[str, Any]] = []
    for path in sorted(KNOWLEDGE_BASE_ROOT.glob("doc_*.json")):
        try:
            payload = read_json_file(path)
        except Exception:
            logger.exception("Не удалось прочитать KB-док %s", path)
            continue
        if not isinstance(payload, dict):
            continue
        docs.append(payload)
    docs.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return docs


def list_projects() -> list[dict[str, Any]]:
    ensure_feature_roots()
    projects: list[dict[str, Any]] = []
    for path in sorted(PROJECT_CONTEXT_ROOT.glob("project_*.json")):
        try:
            payload = read_json_file(path)
        except Exception:
            logger.exception("Не удалось прочитать проектовый индекс %s", path)
            continue
        if not isinstance(payload, dict):
            continue
        projects.append(payload)
    projects.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return projects


def get_knowledge_doc(doc_id: str) -> dict[str, Any] | None:
    path = knowledge_doc_file_path(doc_id)
    if not path.is_file():
        return None
    try:
        payload = read_json_file(path)
    except Exception:
        logger.exception("Не удалось прочитать KB-док %s", path)
        return None
    return payload if isinstance(payload, dict) else None


def get_project_record(project_id: str) -> dict[str, Any] | None:
    path = project_record_file_path(project_id)
    if not path.is_file():
        return None
    try:
        payload = read_json_file(path)
    except Exception:
        logger.exception("Не удалось прочитать проект %s", path)
        return None
    return payload if isinstance(payload, dict) else None


def render_knowledge_docs_text() -> str:
    docs = list_knowledge_docs()
    if not docs:
        return "Локальная БЗ пока пустая."
    lines = ["Локальная БЗ:"]
    for index, doc in enumerate(docs, start=1):
        lines.append(
            f"{index}. [{str(doc.get('doc_id') or '')[:8]}] {doc.get('title') or 'без названия'} | "
            f"файлов: {doc.get('file_count', 0)} | чанков: {doc.get('chunk_count', 0)}"
        )
        lines.append(f"   Источник: {doc.get('source_path') or '-'}")
    return "\n".join(lines)


def render_projects_text() -> str:
    projects = list_projects()
    if not projects:
        return "Проектов пока нет."
    lines = ["Проекты:"]
    for index, project in enumerate(projects, start=1):
        lines.append(
            f"{index}. [{str(project.get('project_id') or '')[:8]}] {project.get('title') or 'без названия'} | "
            f"файлов: {project.get('file_count', 0)} | символов: {project.get('total_chars', 0)}"
        )
        lines.append(f"   Путь: {project.get('project_path') or '-'}")
    return "\n".join(lines)


def resolve_knowledge_doc(query: str) -> dict[str, Any] | None:
    token = str(query or "").strip().lower()
    if not token:
        return None
    docs = list_knowledge_docs()
    if token.isdigit():
        index = int(token)
        if 1 <= index <= len(docs):
            return docs[index - 1]
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "")
        title = str(doc.get("title") or "").lower()
        source_path = str(doc.get("source_path") or "").lower()
        if doc_id.startswith(token) or token in title or token in source_path:
            return doc
    return None


def resolve_project_record(query: str) -> dict[str, Any] | None:
    token = str(query or "").strip().lower()
    if not token:
        return None
    projects = list_projects()
    if token.isdigit():
        index = int(token)
        if 1 <= index <= len(projects):
            return projects[index - 1]
    for project in projects:
        project_id = str(project.get("project_id") or "")
        title = str(project.get("title") or "").lower()
        project_path = str(project.get("project_path") or "").lower()
        if project_id.startswith(token) or token in title or token in project_path:
            return project
    return None


def delete_knowledge_doc(doc_id: str) -> bool:
    path = knowledge_doc_file_path(doc_id)
    if not path.is_file():
        return False
    path.unlink(missing_ok=True)
    return True


def delete_project_record(project_id: str) -> bool:
    path = project_record_file_path(project_id)
    if not path.is_file():
        return False
    path.unlink(missing_ok=True)
    for settings in dialog_runtime_settings.values():
        if str(settings.get("active_project_id") or "") == project_id:
            settings["active_project_id"] = ""
    return True


def render_project_record_detail(project: dict[str, Any]) -> str:
    files = project.get("files") or []
    lines = [
        f"Проект [{str(project.get('project_id') or '')[:8]}]: {project.get('title') or '-'}",
        f"Путь: {project.get('project_path') or '-'}",
        f"Файлов в индексе: {project.get('file_count', 0)}",
        f"Символов: {project.get('total_chars', 0)}",
    ]
    extension_summary = summarize_project_extension_counts(project)
    if extension_summary:
        lines.append(f"Файлы по типам: {extension_summary}")
    if files:
        lines.append("Файлы:")
        for item in files[:12]:
            lines.append(
                f"- {item.get('path') or '-'} | chars={item.get('chars', 0)} | chunks={item.get('chunk_count', 0)}"
            )
    return "\n".join(lines)


def tokenize_local_search_text(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_./:-]+", str(text or "").lower())
    normalized = [token for token in tokens if len(token) >= 2]
    return normalized[:60]


def score_context_chunk(query_tokens: list[str], haystack: str, path_hint: str = "") -> float:
    if not query_tokens:
        return 0.0
    haystack_lower = haystack.lower()
    path_lower = path_hint.lower()
    score = 0.0
    for token in query_tokens:
        if token in path_lower:
            score += 3.0
        count = haystack_lower.count(token)
        if count:
            score += min(4.0, 1.0 + count * 0.7)
    if score <= 0:
        return 0.0
    score += min(2.0, len(query_tokens) * 0.05)
    return score


def search_chunks_in_payloads(
    payloads: list[dict[str, Any]],
    query: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    query_tokens = tokenize_local_search_text(query)
    if not query_tokens:
        return []
    matches: list[dict[str, Any]] = []
    for payload in payloads:
        source_id = str(payload.get("doc_id") or payload.get("project_id") or "")
        source_title = str(payload.get("title") or "")
        for chunk in payload.get("chunks", []):
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text") or "").strip()
            path_hint = str(chunk.get("path") or "").strip()
            score = score_context_chunk(query_tokens, text, path_hint)
            if score <= 0:
                continue
            matches.append(
                {
                    "score": score,
                    "source_id": source_id,
                    "source_title": source_title,
                    "path": path_hint,
                    "text": text,
                }
            )
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:limit]


def render_context_matches(
    label: str,
    matches: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    if not matches:
        return ""
    lines = [label]
    used_chars = 0
    for match in matches:
        snippet = summarize_long_think_iteration(str(match.get("text") or ""), max_chars=420)
        block = (
            f"- {match.get('source_title') or 'источник'}"
            f" | {match.get('path') or '-'}\n"
            f"  {snippet}"
        )
        if used_chars + len(block) > max_chars and used_chars > 0:
            break
        lines.append(block)
        used_chars += len(block)
    return "\n".join(lines).strip()


def touch_background_task(task_id: str) -> None:
    background_task_order.pop(task_id, None)
    background_task_order[task_id] = None
    while len(background_task_order) > TASK_QUEUE_HISTORY_LIMIT:
        stale_id, _ = background_task_order.popitem(last=False)
        if stale_id not in background_tasks:
            continue
        stale_task = background_tasks[stale_id]
        status = str(stale_task.get("status") or "")
        if status in {"completed", "failed", "cancelled", "interrupted"}:
            background_tasks.pop(stale_id, None)


def serialize_background_task(task: dict[str, Any]) -> dict[str, Any]:
    payload = dict(task)
    payload.pop("runtime_task", None)
    return payload


def persist_background_task(task: dict[str, Any]) -> None:
    task["updated_at"] = iso_now()
    task_id = str(task["task_id"])
    touch_background_task(task_id)
    atomic_write_json(background_task_file_path(task_id), serialize_background_task(task))


def load_background_task_from_disk(task_id: str) -> dict[str, Any] | None:
    path = background_task_file_path(task_id)
    if not path.is_file():
        return None
    try:
        payload = read_json_file(path)
    except Exception:
        logger.exception("Не удалось прочитать task %s", path)
        return None
    if not isinstance(payload, dict):
        return None
    payload["runtime_task"] = None
    return payload


def load_background_tasks_from_disk() -> None:
    ensure_feature_roots()
    for path in TASK_QUEUE_ROOT.glob("task_*.json"):
        try:
            payload = read_json_file(path)
        except Exception:
            logger.exception("Не удалось прочитать task-файл %s", path)
            continue
        if not isinstance(payload, dict):
            continue
        task_id = str(payload.get("task_id") or path.stem.replace("task_", ""))
        payload["runtime_task"] = None
        if str(payload.get("status") or "") == "running":
            payload["status"] = "interrupted"
            payload["error"] = payload.get("error") or "Фоновая задача была прервана перезапуском процесса."
            payload["completed_at"] = iso_now()
        background_tasks[task_id] = payload
        touch_background_task(task_id)


def list_background_tasks(owner_key: str | None = None) -> list[dict[str, Any]]:
    if not background_tasks and TASK_QUEUE_ROOT.is_dir():
        load_background_tasks_from_disk()
    tasks = list(background_tasks.values())
    if owner_key is not None:
        tasks = [task for task in tasks if str(task.get("owner_key") or "") == owner_key]
    tasks.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return tasks[:TASK_QUEUE_HISTORY_LIMIT]


def render_background_tasks_text(owner_key: str | None = None) -> str:
    tasks = list_background_tasks(owner_key)
    if not tasks:
        return "Фоновых задач пока нет."
    lines = ["Фоновые задачи:"]
    for index, task in enumerate(tasks, start=1):
        lines.append(
            f"{index}. [{str(task.get('task_id') or '')[:8]}] {task.get('kind')} | "
            f"{task.get('status')} | {task.get('description') or '-'}"
        )
        progress_total = int(task.get("progress_total") or 0)
        progress_current = int(task.get("progress_current") or 0)
        progress_label = str(task.get("progress_label") or "").strip()
        if progress_total > 0:
            lines.append(
                f"   Прогресс: {progress_current}/{progress_total}"
                + (f" | {progress_label}" if progress_label else "")
            )
        elif progress_label:
            lines.append(f"   {progress_label}")
        if task.get("error"):
            lines.append(f"   Ошибка: {task['error']}")
    return "\n".join(lines)


def resolve_background_task(query: str, owner_key: str | None = None) -> dict[str, Any] | None:
    token = str(query or "").strip().lower()
    if not token:
        return None
    tasks = list_background_tasks(owner_key)
    if token.isdigit():
        index = int(token)
        if 1 <= index <= len(tasks):
            return tasks[index - 1]
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        description = str(task.get("description") or "").lower()
        if task_id.startswith(token) or token in description:
            return task
    return None


def create_background_task(
    *,
    kind: str,
    owner_key: str,
    description: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    ensure_feature_roots()
    task = {
        "task_id": uuid.uuid4().hex,
        "kind": kind,
        "owner_key": owner_key,
        "status": "queued",
        "description": description.strip(),
        "payload": payload,
        "result": {},
        "error": "",
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "started_at": None,
        "completed_at": None,
        "progress_current": 0,
        "progress_total": 0,
        "progress_label": "",
        "cancel_requested": False,
        "runtime_task": None,
    }
    background_tasks[str(task["task_id"])] = task
    persist_background_task(task)
    if background_task_event is not None:
        background_task_event.set()
    return task


def delete_background_task(task_id: str) -> None:
    background_tasks.pop(task_id, None)
    background_task_order.pop(task_id, None)
    background_task_file_path(task_id).unlink(missing_ok=True)


def set_background_task_progress(
    task: dict[str, Any],
    *,
    current: int | None = None,
    total: int | None = None,
    label: str | None = None,
) -> None:
    if current is not None:
        task["progress_current"] = max(0, int(current))
    if total is not None:
        task["progress_total"] = max(0, int(total))
    if label is not None:
        task["progress_label"] = str(label).strip()
    persist_background_task(task)


async def run_background_task_payload(task: dict[str, Any]) -> dict[str, Any]:
    kind = str(task.get("kind") or "")
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}

    if kind == "kb_ingest":
        source_path = is_existing_local_path(str(payload.get("source_path") or ""))
        if source_path is None:
            raise RuntimeError("Источник для локальной БЗ не найден.")
        set_background_task_progress(task, label="Собираю локальную базу знаний...")
        doc = await asyncio.to_thread(build_knowledge_doc_payload, source_path)
        atomic_write_json(knowledge_doc_file_path(str(doc["doc_id"])), doc)
        return {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "chunk_count": doc["chunk_count"],
            "file_count": doc["file_count"],
        }

    if kind == "project_scan":
        project_path = is_existing_local_path(str(payload.get("project_path") or ""))
        if project_path is None:
            raise RuntimeError("Папка проекта не найдена.")
        set_background_task_progress(task, label="Сканирую проект и строю индекс...")
        project_record = await asyncio.to_thread(build_project_record_payload, project_path)
        replace_id = str(payload.get("replace_project_id") or "").strip()
        if replace_id:
            existing = get_project_record(replace_id)
            if existing is not None:
                project_record["project_id"] = replace_id
                project_record["created_at"] = existing.get("created_at") or project_record["created_at"]
                project_record["updated_at"] = iso_now()
                project_record["title"] = existing.get("title") or project_record["title"]
        atomic_write_json(project_record_file_path(str(project_record["project_id"])), project_record)
        return {
            "project_id": project_record["project_id"],
            "title": project_record["title"],
            "file_count": project_record["file_count"],
            "total_chars": project_record["total_chars"],
        }

    raise RuntimeError(f"Неизвестный тип фоновой задачи: {kind}")


async def process_background_task(task: dict[str, Any]) -> None:
    if task.get("cancel_requested"):
        task["status"] = "cancelled"
        task["completed_at"] = iso_now()
        persist_background_task(task)
        return
    task["status"] = "running"
    task["started_at"] = task.get("started_at") or iso_now()
    persist_background_task(task)
    try:
        result = await run_background_task_payload(task)
    except Exception as exc:
        task["status"] = "failed"
        task["error"] = str(exc)
        task["completed_at"] = iso_now()
        persist_background_task(task)
        logger.exception("Фоновая задача упала: task_id=%s kind=%s", task.get("task_id"), task.get("kind"))
        return
    task["result"] = result
    task["status"] = "completed"
    task["completed_at"] = iso_now()
    task["error"] = ""
    set_background_task_progress(task, label="Готово.")
    persist_background_task(task)


async def run_background_task_worker_loop() -> None:
    try:
        while True:
            queued_task: dict[str, Any] | None = None
            queued_tasks = [
                task
                for task in background_tasks.values()
                if str(task.get("status") or "") == "queued"
            ]
            if queued_tasks:
                queued_tasks.sort(key=lambda item: str(item.get("created_at") or ""))
                queued_task = queued_tasks[0]
            if queued_task is None:
                if background_task_event is None:
                    await asyncio.sleep(TASK_QUEUE_IDLE_SLEEP_SECONDS)
                else:
                    try:
                        await asyncio.wait_for(
                            background_task_event.wait(),
                            timeout=TASK_QUEUE_IDLE_SLEEP_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        pass
                    if background_task_event is not None:
                        background_task_event.clear()
                continue
            await process_background_task(queued_task)
    except asyncio.CancelledError:
        return


def ensure_background_task_worker_running() -> None:
    global background_task_worker
    global background_task_event

    ensure_feature_roots()
    if not background_tasks and TASK_QUEUE_ROOT.is_dir():
        load_background_tasks_from_disk()
    if background_task_event is None:
        background_task_event = asyncio.Event()
    if background_task_worker is None or background_task_worker.done():
        background_task_worker = asyncio.create_task(run_background_task_worker_loop())


def cancel_background_task(task: dict[str, Any]) -> None:
    task["cancel_requested"] = True
    if str(task.get("status") or "") == "queued":
        task["status"] = "cancelled"
        task["completed_at"] = iso_now()
    else:
        task["error"] = task.get("error") or "Задачу пометили на отмену."
    persist_background_task(task)


def create_terminal_session_payload(session_number: int) -> dict[str, Any]:
    now = iso_now()
    return {
        "session_number": session_number,
        "session_id": uuid.uuid4().hex,
        "title": "",
        "created_at": now,
        "updated_at": now,
        "last_opened_at": now,
        "prompt_signature": PROMPT_SNAPSHOT_SIGNATURE,
        "saved_char_limit": None,
        "char_limit_save_choice_made": False,
        "request_count": 0,
        "clipboard_text": "",
        "last_user_text": "",
        "last_bot_text": "",
        "settings": normalize_dialog_runtime_settings(),
        "history": [],
    }


def save_terminal_session(session: dict[str, Any]) -> None:
    ensure_terminal_sessions_root()
    session_number = int(session["session_number"])
    path = get_terminal_session_file_path(session_number)
    payload = {
        "session_number": session_number,
        "session_id": str(session.get("session_id") or uuid.uuid4().hex),
        "title": str(session.get("title") or "").strip(),
        "created_at": str(session.get("created_at") or iso_now()),
        "updated_at": str(session.get("updated_at") or iso_now()),
        "last_opened_at": str(session.get("last_opened_at") or iso_now()),
        "prompt_signature": PROMPT_SNAPSHOT_SIGNATURE,
        "saved_char_limit": clamp_terminal_session_char_limit(session.get("saved_char_limit")),
        "char_limit_save_choice_made": bool(session.get("char_limit_save_choice_made")),
        "request_count": coerce_nonnegative_int(session.get("request_count")),
        "clipboard_text": normalize_terminal_clipboard_text(session.get("clipboard_text")),
        "last_user_text": normalize_terminal_clipboard_text(session.get("last_user_text")),
        "last_bot_text": normalize_terminal_clipboard_text(session.get("last_bot_text")),
        "settings": normalize_dialog_runtime_settings(session.get("settings")),
        "history": normalize_terminal_session_history(session.get("history", [])),
    }
    temp_path = path.with_suffix(".json.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def load_terminal_session(session_number: int) -> tuple[dict[str, Any] | None, bool]:
    path = get_terminal_session_file_path(session_number)
    if not path.is_file():
        return None, False

    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Не удалось прочитать terminal-сессию #%s", session_number)
        return None, False

    if not isinstance(raw_payload, dict):
        return None, False

    session = create_terminal_session_payload(session_number)
    session_id = str(raw_payload.get("session_id") or "").strip()
    if session_id:
        session["session_id"] = session_id

    title = str(raw_payload.get("title") or "").strip()
    if title:
        session["title"] = truncate_text(title, TERMINAL_SESSION_TITLE_CHARS)

    for key in ("created_at", "updated_at", "last_opened_at"):
        raw_value = str(raw_payload.get(key) or "").strip()
        if raw_value:
            session[key] = raw_value

    session["saved_char_limit"] = clamp_terminal_session_char_limit(raw_payload.get("saved_char_limit"))
    session["char_limit_save_choice_made"] = bool(raw_payload.get("char_limit_save_choice_made"))
    session["request_count"] = coerce_nonnegative_int(raw_payload.get("request_count"))
    session["clipboard_text"] = normalize_terminal_clipboard_text(raw_payload.get("clipboard_text"))
    session["last_user_text"] = normalize_terminal_clipboard_text(raw_payload.get("last_user_text"))
    session["last_bot_text"] = normalize_terminal_clipboard_text(raw_payload.get("last_bot_text"))
    session["settings"] = normalize_dialog_runtime_settings(raw_payload.get("settings"))

    history = normalize_terminal_session_history(raw_payload.get("history", []))
    raw_signature = str(raw_payload.get("prompt_signature") or "").strip()
    history_cleared = bool(history) and bool(raw_signature) and raw_signature != PROMPT_SNAPSHOT_SIGNATURE
    session["history"] = [] if history_cleared else history
    session["prompt_signature"] = PROMPT_SNAPSHOT_SIGNATURE
    return session, history_cleared


def create_terminal_session() -> dict[str, Any]:
    existing_numbers = list_terminal_session_numbers()
    session_number = (existing_numbers[-1] + 1) if existing_numbers else 1
    session = create_terminal_session_payload(session_number)
    save_terminal_session(session)
    return session


def open_terminal_session(
    session_number: int | None,
) -> tuple[dict[str, Any], bool, bool]:
    if session_number is None:
        session = create_terminal_session()
        session["previous_opened_at"] = ""
        return session, True, False

    session, history_cleared = load_terminal_session(session_number)
    if session is None:
        raise RuntimeError(f"Сессия #{session_number} не найдена.")

    session["previous_opened_at"] = str(session.get("last_opened_at") or "")
    now = iso_now()
    session["last_opened_at"] = now
    session["updated_at"] = now
    save_terminal_session(session)
    return session, False, history_cleared


def restore_terminal_session_runtime(session: dict[str, Any], dialog_key: str) -> None:
    history_limit = None if MAX_HISTORY_MESSAGES < 1 else MAX_HISTORY_MESSAGES
    history = deque(maxlen=history_limit)
    history.extend(normalize_terminal_session_history(session.get("history", [])))
    dialog_histories[dialog_key] = history
    dialog_prompt_snapshots[dialog_key] = PROMPT_SNAPSHOT_SIGNATURE
    dialog_runtime_settings[dialog_key] = normalize_dialog_runtime_settings(session.get("settings"))
    touch_dialog_state(dialog_key)


def persist_terminal_session_runtime(
    session: dict[str, Any],
    dialog_key: str,
    saved_char_limit: int | None,
    char_limit_save_choice_made: bool,
) -> None:
    session["saved_char_limit"] = clamp_terminal_session_char_limit(saved_char_limit)
    session["char_limit_save_choice_made"] = bool(char_limit_save_choice_made)
    session["updated_at"] = iso_now()
    session["prompt_signature"] = PROMPT_SNAPSHOT_SIGNATURE
    session["settings"] = normalize_dialog_runtime_settings(
        dialog_runtime_settings.get(dialog_key)
    )

    history = dialog_histories.get(dialog_key)
    session["history"] = normalize_terminal_session_history(list(history) if history else [])

    if not str(session.get("title") or "").strip():
        for item in session["history"]:
            if item["role"] == "user":
                session["title"] = build_terminal_session_title(item["content"])
                break

    try:
        save_terminal_session(session)
    except Exception:
        logger.exception(
            "Не удалось сохранить terminal-сессию #%s",
            session.get("session_number"),
        )


def remember_terminal_session_turn(
    session: dict[str, Any],
    dialog_key: str,
    user_text: str,
    saved_char_limit: int | None,
    char_limit_save_choice_made: bool,
    bot_text: str | None = None,
) -> None:
    if not str(session.get("title") or "").strip():
        session["title"] = build_terminal_session_title(user_text)
    session["request_count"] = coerce_nonnegative_int(session.get("request_count")) + 1
    session["last_user_text"] = normalize_terminal_clipboard_text(user_text)
    session["clipboard_text"] = session["last_user_text"]
    if bot_text is not None:
        session["last_bot_text"] = normalize_terminal_clipboard_text(bot_text)
    persist_terminal_session_runtime(
        session,
        dialog_key,
        saved_char_limit,
        char_limit_save_choice_made,
    )


def build_terminal_session_welcome_text(
    session: dict[str, Any],
    *,
    created: bool,
    history_cleared: bool,
    resume_text: str = "",
) -> str:
    history = normalize_terminal_session_history(session.get("history", []))
    settings = normalize_dialog_runtime_settings(session.get("settings"))
    lines = [
        "Терминальный режим активен.",
        (
            f"Создал новую сессию #{session['session_number']}."
            if created
            else f"Поднял сохранённую сессию #{session['session_number']}."
        ),
    ]
    if session.get("title"):
        lines.append(f"Название: {session['title']}")
    if history:
        lines.append(f"Сообщений в памяти: {len(history)}")
    lines.append(f"Режим: {settings.get('response_mode')}")
    lines.append(f"Локальная БЗ: {'вкл' if settings.get('kb_enabled') else 'выкл'}")
    active_project_id = str(settings.get("active_project_id") or "").strip()
    if active_project_id:
        lines.append(f"Активный проект: {active_project_id[:8]}")
    if history_cleared:
        lines.append(
            "Старую историю пришлось очистить: системный промпт уже менялся, "
            "так что тащить старый контекст было бы криво."
        )
    saved_char_limit = clamp_terminal_session_char_limit(session.get("saved_char_limit"))
    if saved_char_limit is not None:
        lines.append(f"Сохранённый лимит: {saved_char_limit} символов.")
    lines.extend(
        [
            "Пиши запрос сюда. Для выхода введи /exit, exit или quit.",
            "Команды: /help, /mode, /kb, /project, /model, /models, /tasks, /session, /limit, /reset, /repeat, /clipboard, /paste, /deepplan, /deepthink, /deepstatus, /deepcancel, /errors.",
        ]
    )
    if resume_text.strip():
        lines.extend(["", resume_text.strip()])
    return "\n".join(lines) + "\n"


def parse_iso_datetime(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError:
        return None


def ensure_stdout_utf8() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


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


def set_readline_prefill_text(text: str) -> None:
    global readline_prefill_text

    with readline_prefill_lock:
        readline_prefill_text = normalize_terminal_clipboard_text(text)


def consume_readline_prefill_text() -> str:
    global readline_prefill_text

    with readline_prefill_lock:
        value = readline_prefill_text
        readline_prefill_text = ""
    return value


def read_console_input(prompt: str = "") -> str:
    if (
        readline is not None
        and hasattr(sys.stdin, "isatty")
        and sys.stdin.isatty()
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    ):
        prefill_text = consume_readline_prefill_text()
        if prefill_text:
            def apply_prefill() -> None:
                readline.insert_text(prefill_text)
                readline.redisplay()

            readline.set_startup_hook(apply_prefill)
        try:
            raw_line = input(prompt)
        finally:
            readline.set_startup_hook(None)
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


def console_supports_live_updates() -> bool:
    return bool(hasattr(sys.stdout, "isatty") and sys.stdout.isatty())


def clear_console_live_line_unlocked() -> None:
    if not console_supports_live_updates():
        return
    if console_live_render_len <= 0:
        return
    sys.stdout.write("\r" + (" " * console_live_render_len) + "\r")


def restore_console_live_line_unlocked() -> None:
    if not console_supports_live_updates():
        return
    if not console_live_text:
        return
    sys.stdout.write("\r" + console_live_text.ljust(console_live_render_len))


def is_console_live_line_active() -> bool:
    with console_live_lock:
        return bool(console_live_text)


def write_console_output(text: str, *, end: str = "\n", flush: bool = True) -> None:
    with console_live_lock:
        had_live = bool(console_live_text)
        if had_live:
            clear_console_live_line_unlocked()
        sys.stdout.write(text + end)
        if had_live:
            restore_console_live_line_unlocked()
        if flush:
            sys.stdout.flush()


def update_console_live_line(text: str) -> None:
    global console_live_text
    global console_live_render_len

    normalized = text.replace("\r", " ").replace("\n", " ").strip()
    if not console_supports_live_updates():
        return

    with console_live_lock:
        console_live_text = normalized
        console_live_render_len = max(console_live_render_len, len(normalized))
        sys.stdout.write("\r" + normalized.ljust(console_live_render_len))
        sys.stdout.flush()


def clear_console_live_line_permanently() -> None:
    global console_live_text
    global console_live_render_len

    if not console_supports_live_updates():
        console_live_text = ""
        console_live_render_len = 0
        return

    with console_live_lock:
        clear_console_live_line_unlocked()
        console_live_text = ""
        console_live_render_len = 0
        sys.stdout.flush()


class LiveAwareStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            with console_live_lock:
                had_live = bool(console_live_text) and self.stream is sys.stdout
                if had_live:
                    clear_console_live_line_unlocked()
                super().emit(record)
                if had_live:
                    restore_console_live_line_unlocked()
                self.flush()
        except Exception:
            self.handleError(record)


def append_supervisor_log(level: str, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {level.upper()} | {message.strip()}"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with SUPERVISOR_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def get_supervisor_restart_delay_seconds(recent_failures: int) -> int:
    exponent = max(0, recent_failures - 1)
    delay = SUPERVISOR_RESTART_BASE_DELAY_SECONDS * (2**exponent)
    return min(SUPERVISOR_RESTART_MAX_DELAY_SECONDS, delay)


def setup_logging(*, console_enabled: bool = True) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        RUNTIME_LOG_PATH,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    if console_enabled:
        stream_handler = LiveAwareStreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


@asynccontextmanager
async def acquire_cross_process_model_lock():
    if fcntl is None:
        yield
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = MODEL_RUNTIME_LOCK_PATH.open("a+", encoding="utf-8")
    try:
        await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def sanitize_for_log(value: Any) -> Any:
    if isinstance(value, str):
        return truncate_text(value, MAX_LOG_TEXT_CHARS)
    if isinstance(value, dict):
        return {str(key): sanitize_for_log(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_log(item) for item in value]
    return value


def append_jsonl(record: dict[str, Any]) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with INTERACTIONS_LOG_PATH.open("a", encoding="utf-8") as handle:
            json.dump(sanitize_for_log(record), handle, ensure_ascii=False)
            handle.write("\n")
        record_db_event(record)
    except Exception:
        logger.exception("Не удалось записать JSONL лог.")


def trim_history_text(text: str) -> str:
    return truncate_text(text.strip(), MAX_HISTORY_ENTRY_CHARS)


def append_reply_chunk(
    current: str,
    chunk: str,
    max_chars: int | None = MAX_MODEL_REPLY_CHARS,
) -> tuple[str, bool]:
    updated = current + chunk
    if max_chars is None or max_chars <= 0:
        return updated, False
    if len(updated) <= max_chars:
        return updated, False
    return updated[:max_chars], True


def touch_dialog_state(dialog_key: str) -> None:
    dialog_activity_order.pop(dialog_key, None)
    dialog_activity_order[dialog_key] = None
    prune_dialog_state()


def prune_dialog_state() -> None:
    while len(dialog_activity_order) > MAX_ACTIVE_DIALOGS:
        stale_dialog_key, _ = dialog_activity_order.popitem(last=False)
        dialog_histories.pop(stale_dialog_key, None)
        bot_response_message_ids.pop(stale_dialog_key, None)
        dialog_prompt_snapshots.pop(stale_dialog_key, None)
        multi_request_sessions.pop(stale_dialog_key, None)
        logger.info("Удаляю старое состояние диалога из памяти: dialog_key=%s", stale_dialog_key)


def build_license_notice_text() -> str:
    return (
        "Перейдите на репозиторий разработчика и кликните по LICENSE - "
        "https://github.com/AlbertGithot/ai-to-tgbot-port"
    )


def is_ai_enabled() -> bool:
    return AI_ENABLED


def build_start_keyboard() -> InlineKeyboardMarkup:
    inline_keyboard = [
        [
            InlineKeyboardButton(
                text="Показать лицензию GNU AGPLv3",
                callback_data=LICENSE_CALLBACK,
            )
        ],
    ]
    if SOURCE_URL:
        inline_keyboard.append(
            [
                InlineKeyboardButton(
                    text="Source Code",
                    callback_data=SOURCE_CODE_CALLBACK,
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=inline_keyboard)


def build_response_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Сбросить диалог и очистить чат",
                    callback_data=RESET_DIALOG_CALLBACK,
                )
            ]
        ]
    )


def build_ineedmore_keyboard(query_count: int) -> InlineKeyboardMarkup:
    edit_button = InlineKeyboardButton(
        text="Редактировать", callback_data=INEEDMORE_ACTION_EDIT
    )
    add_button = InlineKeyboardButton(
        text="Добавить", callback_data=INEEDMORE_ACTION_ADD
    )
    remove_button = InlineKeyboardButton(
        text="Убрать", callback_data=INEEDMORE_ACTION_REMOVE
    )
    confirm_button = InlineKeyboardButton(
        text="Подтвердить отправку", callback_data=INEEDMORE_ACTION_CONFIRM
    )
    cancel_button = InlineKeyboardButton(
        text="Отмена", callback_data=INEEDMORE_ACTION_CANCEL
    )

    first_row = [edit_button, add_button]
    if query_count > 1:
        first_row.append(remove_button)

    return InlineKeyboardMarkup(
        inline_keyboard=[
            first_row,
            [confirm_button],
            [cancel_button],
        ]
    )


def build_ineedmore_edit_keyboard(query_count: int) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="Тема", callback_data=INEEDMORE_ACTION_EDIT_TOPIC)]
    ]
    for index in range(query_count):
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"Запрос {index + 1}",
                    callback_data=f"{INEEDMORE_ACTION_EDIT_QUERY_PREFIX}{index}",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="Назад", callback_data=INEEDMORE_ACTION_EDIT_BACK)]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def create_multi_request_session() -> dict[str, Any]:
    return {
        "topic": "",
        "queries": [""] * DEFAULT_MULTI_REQUEST_ITEMS,
        "edit_target": None,
    }


def get_multi_request_session(dialog_key: str) -> dict[str, Any] | None:
    session = multi_request_sessions.get(dialog_key)
    if session is not None:
        touch_dialog_state(dialog_key)
    return session


def render_multi_request_form(topic: str, queries: list[str]) -> str:
    lines = [
        "Шаблон /ineedmore",
        "",
        f"Тема: {topic}".rstrip(),
    ]
    for index, query in enumerate(queries, start=1):
        lines.append(f"{index}. {query}".rstrip())
    lines.extend(
        [
            "",
            "Отправь этот шаблон одним сообщением или жми кнопки ниже, чтобы править поля по отдельности.",
        ]
    )
    return "\n".join(lines)


def parse_multi_request_form(text: str, expected_count: int) -> tuple[str, list[str]] | None:
    normalized_lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not normalized_lines:
        return None

    topic = ""
    queries: dict[int, str] = {}

    for line in normalized_lines:
        if line.lower().startswith("тема:"):
            topic = line.split(":", 1)[1].strip()
            continue

        match = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
        if match:
            item_index = int(match.group(1))
            queries[item_index] = match.group(2).strip()

    ordered_queries = [
        queries.get(index, "").strip() for index in range(1, expected_count + 1)
    ]
    if not any(ordered_queries) and not topic:
        return None
    return topic, ordered_queries


def shorten_status_label(text: str, max_len: int = 64) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""
    if len(cleaned) <= max_len:
        return cleaned
    shortened = cleaned[: max_len - 1].rsplit(" ", 1)[0].strip()
    return (shortened or cleaned[: max_len - 1]).rstrip(".,;:-") + "…"


def render_multi_request_status(
    topic: str,
    queries: list[str],
    statuses: list[str],
    *,
    elapsed_seconds: int = 0,
    extra_line: str | None = None,
) -> str:
    def estimate_eta(position: int) -> str:
        if position <= 1:
            return "примерно 1-2 мин."
        start = (position - 1) * 2
        end = position * 2
        return f"примерно {start}-{end} мин."

    def render_status_label(index: int, status: str) -> str:
        if status == "done":
            return "готово! ✅"
        if status == "thinking":
            return f"думаем 🤔 {estimate_eta(1)}"

        pending_position = 1
        for previous_status in statuses[:index]:
            if previous_status != "done":
                pending_position += 1
        return f"ждём очередь ⏳ {estimate_eta(pending_position)}"

    lines = [
        THINKING_PLACEHOLDER_TEXT,
        f"Ожидание: {format_clock_duration(elapsed_seconds)}",
        "",
        "/ineedmore: статус шаблонов",
        "",
    ]
    if topic.strip():
        lines.append(f"Тема: {topic}")
        lines.append("")

    for index, (query, status) in enumerate(zip(queries, statuses), start=1):
        label = shorten_status_label(query) or f"Запрос {index}"
        lines.append(f"{index}. {label} - {render_status_label(index - 1, status)}")

    if extra_line:
        lines.extend(["", extra_line])

    return "\n".join(lines)


def render_multi_request_edit_menu(topic: str, queries: list[str]) -> str:
    lines = [
        "Что редактируем в /ineedmore?",
        "",
        f"Тема: {topic or 'пусто'}",
    ]
    for index, query in enumerate(queries, start=1):
        lines.append(f"{index}. {query or 'пусто'}")
    return "\n".join(lines)


def render_multi_request_edit_prompt(target: str, queries: list[str]) -> str:
    if target == "topic":
        return "Пришли новое значение для темы одним сообщением."
    if target.startswith("query:"):
        index = int(target.split(":", 1)[1])
        current_value = queries[index] if 0 <= index < len(queries) else ""
        return f"Пришли новый текст для запроса {index + 1}.\nСейчас: {current_value or 'пусто'}"
    return "Пришли новое значение одним сообщением."


async def safe_edit_message(
    message: Message,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> Message:
    try:
        result = await message.edit_text(text, reply_markup=reply_markup)
        if isinstance(result, Message):
            return result
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            logger.exception("Не удалось отредактировать сообщение Telegram.")
    except Exception:
        logger.exception("Не удалось отредактировать сообщение Telegram.")
    return message


async def answer_long(message: Message, text: str, dialog_key: str | None = None) -> None:
    dialog_key = dialog_key or get_dialog_key(message)
    reply_markup = build_response_keyboard()
    transport_text = truncate_text(text, MAX_MODEL_REPLY_CHARS)
    chunks = [
        transport_text[i : i + TELEGRAM_SEGMENT_LIMIT]
        for i in range(0, len(transport_text), TELEGRAM_SEGMENT_LIMIT)
    ] or [transport_text]

    first_chunk = chunks[0]
    try:
        result = await message.edit_text(first_chunk, reply_markup=reply_markup)
        if isinstance(result, Message):
            message = result
    except TelegramBadRequest:
        message = await message.answer(first_chunk, reply_markup=reply_markup)
    except Exception:
        logger.exception("Не удалось отправить первый фрагмент длинного ответа через edit_text.")
        message = await message.answer(first_chunk, reply_markup=reply_markup)
    track_bot_message(dialog_key, message)

    for chunk in chunks[1:]:
        try:
            message = await message.answer(chunk, reply_markup=reply_markup)
            track_bot_message(dialog_key, message)
        except Exception:
            logger.exception("Не удалось отправить длинный фрагмент ответа в Telegram.")
            break


def get_text_limit_error(text: str, limit: int, label: str) -> str | None:
    if len(text) <= limit:
        return None
    return (
        f"{label} слишком длинный. Сейчас {len(text)} символов, "
        f"лимит {limit}. Укороти и отправь ещё раз."
    )


def validate_runtime_config() -> None:
    ensure_valid_model_path()

    if not LLAMA_SERVER_EXE.is_file():
        raise RuntimeError(f"Не найден llama-server: {LLAMA_SERVER_EXE}")


def validate_telegram_config() -> None:
    if not BOT_TOKEN or "PASTE_TELEGRAM_BOT_TOKEN_HERE" in BOT_TOKEN:
        raise RuntimeError("Вставь токен бота в BOT_TOKEN.")


def validate_config() -> None:
    validate_runtime_config()
    validate_telegram_config()


def build_llama_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    library_dirs: list[str] = []
    for candidate in (
        LLAMA_SERVER_EXE.parent,
        LLAMA_SERVER_EXE.parent / "lib",
        LLAMA_SERVER_EXE.parent.parent / "lib",
        LLAMA_CPP_DIR,
        LLAMA_CPP_DIR / "lib",
        LLAMA_CPP_DIR.parent / "lib",
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


def ensure_llama_server_executable() -> None:
    if os.name == "nt" or not LLAMA_SERVER_EXE.is_file():
        return
    try:
        current_mode = LLAMA_SERVER_EXE.stat().st_mode
        LLAMA_SERVER_EXE.chmod(current_mode | 0o111)
    except Exception:
        logger.exception("Не удалось выставить исполняемый бит для llama-server")


def validate_llama_runtime() -> None:
    ensure_llama_server_executable()
    try:
        completed = subprocess.run(
            [str(LLAMA_SERVER_EXE), "--version"],
            cwd=str(LLAMA_CPP_DIR),
            env=build_llama_runtime_env(),
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


async def ensure_telegram_ready(bot: Bot) -> None:
    while True:
        try:
            me = await bot.get_me(request_timeout=TELEGRAM_REQUEST_TIMEOUT)
            logger.info(
                "Поднял Telegram-бота: id=%s username=@%s",
                me.id,
                me.username or "",
            )
            await bot.delete_webhook(
                drop_pending_updates=False,
                request_timeout=TELEGRAM_REQUEST_TIMEOUT,
            )
            logger.info("Webhook очищен, перехожу на polling.")
            return
        except TelegramNetworkError as exc:
            logger.warning(
                "Telegram API недоступен на старте: %s. Повтор через %s сек.",
                exc,
                TELEGRAM_STARTUP_RETRY_DELAY_SECONDS,
            )
            await asyncio.sleep(TELEGRAM_STARTUP_RETRY_DELAY_SECONDS)


async def run_polling_forever(bot: Bot, dispatcher: Dispatcher) -> None:
    while True:
        try:
            await dispatcher.start_polling(
                bot,
                allowed_updates=dispatcher.resolve_used_update_types(),
                polling_timeout=30,
                request_timeout=TELEGRAM_REQUEST_TIMEOUT,
            )
            return
        except TelegramNetworkError as exc:
            logger.warning(
                "Polling потерял соединение с Telegram: %s. Повтор через %s сек.",
                exc,
                TELEGRAM_POLLING_RETRY_DELAY_SECONDS,
            )
            await asyncio.sleep(TELEGRAM_POLLING_RETRY_DELAY_SECONDS)


def get_dialog_key(message: Message) -> str:
    user_id = message.from_user.id if message.from_user else 0
    return f"{message.chat.id}:{user_id}"


def get_callback_dialog_key(callback: CallbackQuery) -> str:
    chat_id = callback.message.chat.id if callback.message is not None else 0
    user_id = callback.from_user.id if callback.from_user else 0
    return f"{chat_id}:{user_id}"


def get_tracked_bot_messages(dialog_key: str) -> deque[int]:
    touch_dialog_state(dialog_key)
    tracked = bot_response_message_ids.get(dialog_key)
    if tracked is None:
        tracked = deque(maxlen=MAX_TRACKED_BOT_MESSAGES)
        bot_response_message_ids[dialog_key] = tracked
    return tracked


def track_bot_message(dialog_key: str, message: Message | None) -> None:
    if message is None:
        return
    tracked = get_tracked_bot_messages(dialog_key)
    if message.message_id in tracked:
        return
    tracked.append(message.message_id)


def forget_tracked_bot_messages(dialog_key: str) -> list[int]:
    tracked = bot_response_message_ids.pop(dialog_key, None)
    if tracked is None:
        return []
    return list(dict.fromkeys(tracked))


def get_dialog_history(dialog_key: str) -> deque[dict[str, str]]:
    touch_dialog_state(dialog_key)
    history = dialog_histories.get(dialog_key)
    if history is None:
        history_limit = None if MAX_HISTORY_MESSAGES < 1 else MAX_HISTORY_MESSAGES
        history = deque(maxlen=history_limit)
        dialog_histories[dialog_key] = history
    return history


def ensure_prompt_snapshot(dialog_key: str) -> None:
    previous_snapshot = dialog_prompt_snapshots.get(dialog_key)
    if previous_snapshot == PROMPT_SNAPSHOT_SIGNATURE:
        return

    if dialog_histories.get(dialog_key):
        logger.info(
            "Системный промпт изменился, сбрасываю старую память диалога: dialog_key=%s",
            dialog_key,
        )
        dialog_histories.pop(dialog_key, None)

    dialog_prompt_snapshots[dialog_key] = PROMPT_SNAPSHOT_SIGNATURE


def get_chat_lock(chat_id: int) -> asyncio.Lock:
    lock = chat_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        chat_locks[chat_id] = lock
    return lock


@asynccontextmanager
async def acquire_model_slot(label: str):
    global model_pending_requests
    global model_active_request_label
    global model_active_started_at

    if model_lock is None:
        raise RuntimeError("Лок модели не инициализирован.")

    model_pending_requests += 1
    queue_position = model_pending_requests
    acquired = False

    try:
        await model_lock.acquire()
        acquired = True
        model_pending_requests = max(0, model_pending_requests - 1)
        model_active_request_label = label
        model_active_started_at = datetime.now()
        try:
            async with acquire_cross_process_model_lock():
                yield queue_position
        finally:
            model_active_request_label = None
            model_active_started_at = None
            model_lock.release()
    finally:
        if not acquired:
            model_pending_requests = max(0, model_pending_requests - 1)


def get_model_runtime_snapshot() -> dict[str, Any]:
    active_for_seconds: int | None = None
    if model_active_started_at is not None:
        active_for_seconds = max(
            0, int((datetime.now() - model_active_started_at).total_seconds())
        )
    return {
        "pending_requests": model_pending_requests,
        "active_label": model_active_request_label,
        "active_for_seconds": active_for_seconds,
    }


def user_payload(message: Message) -> dict[str, Any]:
    user = message.from_user
    if user is None:
        return {}
    return {
        "id": user.id,
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "full_name": user.full_name,
        "language_code": user.language_code,
        "is_bot": user.is_bot,
    }


def chat_payload(message: Message) -> dict[str, Any]:
    return {
        "id": message.chat.id,
        "type": message.chat.type,
        "title": getattr(message.chat, "title", None),
    }


async def reject_if_blocked_message(message: Message) -> bool:
    user_id = message.from_user.id if message.from_user else None
    if not is_user_blocked(user_id):
        return False
    await message.answer("Доступ к боту заблокирован администратором.")
    return True


async def reject_if_blocked_callback(callback: CallbackQuery) -> bool:
    user_id = callback.from_user.id if callback.from_user else None
    if not is_user_blocked(user_id):
        return False
    await callback.answer("Доступ к боту заблокирован администратором.", show_alert=True)
    return True


def should_answer_briefly(user_text: str) -> bool:
    normalized = re.sub(r"\s+", " ", user_text.strip().lower())
    if not normalized:
        return False

    normalized_no_punct = normalized.strip("!?.,:; ")
    if not normalized_no_punct:
        return False

    detail_markers = (
        "подробно",
        "развернуто",
        "в деталях",
        "поподробнее",
        "объясни",
        "распиши",
        "детально",
    )
    if any(marker in normalized_no_punct for marker in detail_markers):
        return False

    brief_exact_phrases = {
        "привет",
        "здарова",
        "здравствуйте",
        "добрый день",
        "добрый вечер",
        "доброе утро",
        "как дела",
        "как ты",
        "ты как",
        "ты кто",
        "кто ты",
        "что умеешь",
        "что делаешь",
        "чем занимаешься",
        "как тебя зовут",
        "ты тут",
        "ты на месте",
        "спасибо",
        "благодарю",
        "понял",
        "ясно",
        "ок",
        "окей",
    }
    if normalized_no_punct in brief_exact_phrases:
        return True

    normalized_aliases = (
        normalized_no_punct.replace("чё", "че").replace("чо", "че").replace("ё", "е")
    )
    if any(phrase in normalized_aliases for phrase in ("че как", "че кого", "как сам", "как сам то")):
        return True

    brief_patterns = (
        # r"\bв\s+ч[её]м\s+твой\b.*\bсмысл\b",
        # r"\bкакой\s+твой\b.*\bсмысл\b",
        # r"\bв\s+ч[её]м\b.*\bсмысл\b",
        # r"\bзачем\s+ты\b",
        # r"\bнахуя\s+ты\b",
        # r"\bты\s+кто\b",
        # r"\bкто\s+ты\b",
        # r"\bчто\s+умеешь\b",
        # r"\bкак\s+дела\b",
        # r"\bче\s+как\b",
        # r"\bче\s+кого\b",
    )
    if any(re.search(pattern, normalized_no_punct) for pattern in brief_patterns):
        return True

    word_count = len(normalized_no_punct.split())
    looks_technical = (
        any(ch.isdigit() for ch in normalized_no_punct)
        or any(symbol in normalized for symbol in ("/", "\\", "`", "=", "_", ":", ";", "(", ")", "[", "]", "{", "}"))
    )

    return (
        len(normalized_no_punct) <= 24
        and word_count <= 3
        and not looks_technical
    )


def should_answer_briefly_for_dialog(dialog_key: str, user_text: str) -> bool:
    settings = get_dialog_runtime_settings(dialog_key)
    if str(settings.get("response_mode") or "") == "code":
        return False
    return should_answer_briefly(user_text)


def get_request_max_tokens(user_text: str) -> int:
    return BRIEF_MAX_TOKENS if should_answer_briefly(user_text) else MAX_TOKENS


def get_code_mode_max_tokens(dialog_key: str, user_text: str) -> int:
    settings = get_dialog_runtime_settings(dialog_key)
    analysis = analyze_code_request(user_text)
    small_budget = min(MAX_TOKENS, max(BRIEF_MAX_TOKENS, 1024))
    medium_budget = min(MAX_TOKENS, max(small_budget, int(MAX_TOKENS * 0.72)))
    large_budget = min(MAX_TOKENS, max(medium_budget, int(MAX_TOKENS * 0.95)))

    if analysis["complexity"] == "large":
        target = large_budget
    elif analysis["complexity"] == "medium":
        target = medium_budget
    else:
        target = small_budget

    if str(settings.get("active_project_id") or "").strip():
        target = max(target, medium_budget)
    if (
        analysis["wants_tests"]
        or analysis["wants_patch"]
        or analysis["wants_files"]
        or analysis["wants_fix"]
        or analysis["wants_refactor"]
    ):
        target = max(target, medium_budget)

    return min(MAX_TOKENS, max(BRIEF_MAX_TOKENS, int(target)))


def get_request_max_tokens_for_dialog(dialog_key: str, user_text: str) -> int:
    settings = get_dialog_runtime_settings(dialog_key)
    if str(settings.get("response_mode") or "") == "code":
        return get_code_mode_max_tokens(dialog_key, user_text)
    return get_request_max_tokens(user_text)


def estimate_max_tokens_for_char_limit(user_text: str, char_limit: int) -> int:
    request_tokens = get_request_max_tokens(user_text)
    estimated_from_chars = max(64, min(request_tokens, (char_limit // 3) + 64))
    return min(request_tokens, estimated_from_chars)


def estimate_max_tokens_for_dialog_char_limit(dialog_key: str, user_text: str, char_limit: int) -> int:
    request_tokens = get_request_max_tokens_for_dialog(dialog_key, user_text)
    estimated_from_chars = max(64, min(request_tokens, (char_limit // 3) + 64))
    return min(request_tokens, estimated_from_chars)


def get_request_history(dialog_key: str, user_text: str) -> list[dict[str, str]]:
    if should_answer_briefly(user_text):
        return []
    return list(get_dialog_history(dialog_key))


def get_request_history_for_dialog(dialog_key: str, user_text: str) -> list[dict[str, str]]:
    if should_answer_briefly_for_dialog(dialog_key, user_text):
        return []
    return list(get_dialog_history(dialog_key))


def get_brief_fallback_reply(user_text: str) -> str | None:
    return None


def sanitize_assistant_reply_text(full_reply: str, raw_reply: str) -> str:
    candidates: list[str] = []
    if full_reply.strip():
        candidates.extend(
            [
                extract_relaxed_visible_reply(full_reply),
                full_reply,
            ]
        )
    if raw_reply.strip():
        candidates.extend(
            [
                extract_relaxed_visible_reply(raw_reply),
                extract_visible_reply(raw_reply, final=True),
                normalize_raw_model_reply(raw_reply),
            ]
        )

    fallback = ""
    for candidate in candidates:
        cleaned = candidate.strip()
        if not cleaned:
            continue
        relaxed_cleaned = extract_relaxed_visible_reply(cleaned)
        if relaxed_cleaned.strip():
            cleaned = relaxed_cleaned.strip()
        if not fallback:
            fallback = cleaned
        if (
            looks_like_reasoning(cleaned)
            or looks_like_prompt_leak(cleaned)
            or is_strict_meta_answer_candidate(cleaned)
        ):
            continue
        return cleaned

    return fallback


def sanitize_assistant_reply_for_history(full_reply: str, raw_reply: str) -> str:
    return trim_history_text(sanitize_assistant_reply_text(full_reply, raw_reply))


def build_system_message_content(
    user_text: str,
    *extra_prompts: str | None,
    base_prompt: str | None = None,
    brief_override: bool | None = None,
) -> str:
    is_brief = should_answer_briefly(user_text) if brief_override is None else bool(brief_override)
    prompt_parts = [
        base_prompt or (BRIEF_SYSTEM_PROMPT if is_brief else SYSTEM_PROMPT)
    ]
    if is_brief:
        prompt_parts.append(BRIEF_REPLY_STYLE_PROMPT)
    prompt_parts.extend(
        prompt.strip() for prompt in extra_prompts if isinstance(prompt, str) and prompt.strip()
    )
    return "\n\n".join(prompt_parts)


def get_system_prompt_for_request(user_text: str) -> str:
    if should_answer_briefly(user_text):
        return BRIEF_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def build_code_mode_instruction(dialog_key: str, user_text: str) -> str:
    settings = get_dialog_runtime_settings(dialog_key)
    analysis = analyze_code_request(user_text)
    has_active_project = bool(str(settings.get("active_project_id") or "").strip())
    lines: list[str] = []

    if analysis["wants_review"]:
        lines.append(CODE_MODE_REVIEW_PROMPT)
    elif analysis["wants_explain"] and not any(
        analysis[key] for key in ("wants_write", "wants_fix", "wants_refactor", "wants_tests")
    ):
        lines.append(CODE_MODE_EXPLANATION_PROMPT)
    else:
        lines.append(CODE_MODE_IMPLEMENTATION_PROMPT)

    if analysis["wants_patch"]:
        lines.append("Если пользователь просит patch или diff, показывай изменения в формате unified diff.")

    if analysis["wants_files"] or analysis["wants_write"] or analysis["wants_fix"] or analysis["wants_refactor"]:
        if has_active_project:
            lines.append(
                "Если правка затрагивает несколько файлов, сначала перечисли файлы и роль каждого изменения, "
                "потом дай код или дифф по каждому файлу."
            )
        else:
            lines.append(
                "Если создаёшь решение с нуля или затрагиваешь несколько файлов, сначала покажи дерево файлов, "
                "потом дай содержимое или дифф по файлам."
            )

    if analysis["wants_tests"] or analysis["wants_write"] or analysis["wants_fix"] or analysis["wants_refactor"]:
        lines.append(CODE_MODE_TEST_PROMPT)

    if analysis["wants_commands"] or analysis["wants_write"] or analysis["wants_fix"] or analysis["wants_refactor"]:
        lines.append("В конце давай конкретные команды запуска, проверки, линтинга и тестов, если они уместны.")

    if has_active_project:
        lines.append(
            "Есть активный проект. Переиспользуй существующие файлы, модули и соглашения вместо выдумывания новой структуры без причины."
        )
    else:
        lines.append(
            "Активного проекта нет. Если опираешься на предположение о структуре файлов, прямо помечай это как шаблон или пример."
        )

    if analysis["mentions_existing_code"] and not has_active_project:
        lines.append(
            "Если запрос выглядит как правка существующего кода, но контекста проекта нет, сначала коротко назови недостающие данные, "
            "а затем дай аккуратный шаблон решения."
        )

    if analysis["complexity"] == "large":
        lines.append(
            "Задача большая: дроби решение на этапы, сначала минимально рабочий вариант, затем расширения, риски и план докрутки."
        )
    elif analysis["complexity"] == "medium":
        lines.append("Если задача состоит из нескольких шагов, придерживайся короткого плана перед реализацией.")

    return " ".join(line.strip() for line in lines if line.strip())


def build_local_context_prompt(dialog_key: str, user_text: str) -> str:
    settings = get_dialog_runtime_settings(dialog_key)
    sections: list[str] = []
    is_code_mode = str(settings.get("response_mode") or "") == "code"

    active_project_id = str(settings.get("active_project_id") or "").strip()
    if active_project_id:
        project_record = get_project_record(active_project_id)
        if project_record is None:
            set_dialog_active_project(dialog_key, "")
        else:
            if is_code_mode:
                project_overview = build_project_overview_prompt(project_record)
                if project_overview:
                    sections.append(project_overview)
            project_matches = search_chunks_in_payloads(
                [project_record],
                user_text,
                limit=PROJECT_CONTEXT_MAX_MATCHES + (2 if is_code_mode else 0),
            )
            project_section = render_context_matches(
                (
                    "Активный проект:\n"
                    f"- id: {str(project_record.get('project_id') or '')[:8]}\n"
                    f"- название: {project_record.get('title') or '-'}\n"
                    f"- путь: {project_record.get('project_path') or '-'}\n"
                    "Релевантные куски проекта:"
                ),
                project_matches,
                max_chars=PROJECT_CONTEXT_MAX_CHARS + (1800 if is_code_mode else 0),
            )
            if project_section:
                sections.append(project_section)

    if settings.get("kb_enabled", True):
        kb_matches = search_chunks_in_payloads(
            list_knowledge_docs(),
            user_text,
            limit=KB_MAX_MATCHES + (2 if is_code_mode else 0),
        )
        kb_section = render_context_matches(
            "Локальная база знаний:",
            kb_matches,
            max_chars=KB_CONTEXT_MAX_CHARS + (1200 if is_code_mode else 0),
        )
        if kb_section:
            sections.append(kb_section)

    return "\n\n".join(section for section in sections if section.strip()).strip()


def build_dialog_prompt_extras(dialog_key: str, user_text: str) -> list[str]:
    settings = get_dialog_runtime_settings(dialog_key)
    extra_prompts: list[str] = []
    if str(settings.get("response_mode") or "chat") == "code":
        extra_prompts.append(CODE_MODE_SYSTEM_PROMPT)
        extra_prompts.append(build_code_mode_instruction(dialog_key, user_text))
    context_prompt = build_local_context_prompt(dialog_key, user_text)
    if context_prompt:
        extra_prompts.append(LOCAL_CONTEXT_SYSTEM_PROMPT)
        extra_prompts.append(context_prompt)
    return extra_prompts


def build_messages(dialog_key: str, user_text: str) -> list[dict[str, str]]:
    ensure_prompt_snapshot(dialog_key)
    is_brief = should_answer_briefly_for_dialog(dialog_key, user_text)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_message_content(
                user_text,
                *build_dialog_prompt_extras(dialog_key, user_text),
                brief_override=is_brief,
            ),
        }
    ]
    messages.extend(get_request_history_for_dialog(dialog_key, user_text))
    messages.append({"role": "user", "content": user_text})
    return messages


def build_repair_messages(dialog_key: str, user_text: str) -> list[dict[str, str]]:
    ensure_prompt_snapshot(dialog_key)
    is_brief = should_answer_briefly_for_dialog(dialog_key, user_text)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_message_content(
                user_text,
                REPAIR_SYSTEM_PROMPT,
                *build_dialog_prompt_extras(dialog_key, user_text),
                brief_override=is_brief,
            ),
        },
    ]
    messages.extend(get_request_history_for_dialog(dialog_key, user_text))
    messages.append({"role": "user", "content": user_text})
    return messages


def build_multi_request_item_messages(topic: str, query: str) -> list[dict[str, str]]:
    prompt = f"Тема: {topic}\nЗапрос: {query}" if topic.strip() else query
    return [
        {
            "role": "system",
            "content": build_system_message_content(
                query,
                "Это один пункт внутри команды /ineedmore. "
                "Ответь только на этот пункт, не ссылайся на другие шаблоны и не делай вид, что между ними есть память.",
                base_prompt=SYSTEM_PROMPT,
            ),
        },
        {"role": "user", "content": prompt},
    ]


def build_multi_request_intro_messages(
    topic: str, queries: list[str], answers: list[str]
) -> list[dict[str, str]]:
    lines: list[str] = []
    if topic.strip():
        lines.append(f"Тема: {topic}")
    for index, (query, answer) in enumerate(zip(queries, answers), start=1):
        lines.append(f"{index}. {query}")
        lines.append(answer)
    joined = "\n".join(lines)
    return [
        {
            "role": "system",
            "content": build_system_message_content(
                joined,
                "Сделай очень короткое общее вступление на 1-2 предложения для пакета ответов. "
                "Без списка, без тегов, без рассуждений.",
                base_prompt=SYSTEM_PROMPT,
            ),
        },
        {"role": "user", "content": joined},
    ]


def build_brief_retry_messages(user_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": build_system_message_content(
                user_text,
                "Предыдущий короткий ответ оборвался. Ответь заново коротко, цельно и без списков. "
                "Нужно 1-3 коротких предложения, полностью завершенная мысль."
            ),
        },
        {"role": "user", "content": user_text},
    ]


def remember_turn(dialog_key: str, user_text: str, bot_text: str) -> None:
    history = get_dialog_history(dialog_key)
    history.append({"role": "user", "content": trim_history_text(user_text)})
    history.append({"role": "assistant", "content": trim_history_text(bot_text)})


def reset_dialog(dialog_key: str) -> None:
    dialog_histories.pop(dialog_key, None)
    dialog_prompt_snapshots.pop(dialog_key, None)
    dialog_activity_order.pop(dialog_key, None)


async def is_llama_server_ready() -> bool:
    timeout = aiohttp.ClientTimeout(total=3)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for url in (LLAMA_SERVER_HEALTH_URL, LLAMA_SERVER_MODELS_URL):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return True
                except aiohttp.ClientError:
                    continue
    except Exception:
        return False
    return False


def start_llama_server() -> None:
    global LLAMA_SERVER_PROCESS
    global LLAMA_SERVER_LOG_HANDLE

    if LLAMA_SERVER_PROCESS is not None and LLAMA_SERVER_PROCESS.poll() is None:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "llama_server.log"
    LLAMA_SERVER_LOG_HANDLE = log_path.open("a", encoding="utf-8")
    validate_llama_runtime()
    command = build_llama_server_command()
    logger.info("Запускаю llama-server: %s", " ".join(command))
    LLAMA_SERVER_PROCESS = subprocess.Popen(
        command,
        cwd=str(LLAMA_CPP_DIR),
        env=build_llama_runtime_env(),
        stdout=LLAMA_SERVER_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def get_llama_server_process_state() -> dict[str, Any]:
    process = LLAMA_SERVER_PROCESS
    if process is None:
        return {"state": "stopped", "pid": None, "exit_code": None}
    exit_code = process.poll()
    if exit_code is None:
        return {"state": "running", "pid": process.pid, "exit_code": None}
    return {"state": "exited", "pid": process.pid, "exit_code": exit_code}


def build_llama_server_command() -> list[str]:
    model_path = ensure_valid_model_path()
    return [
        str(LLAMA_SERVER_EXE),
        "--model",
        str(model_path),
        "--host",
        LLAMA_SERVER_HOST,
        "--port",
        str(LLAMA_SERVER_PORT),
        "--ctx-size",
        str(N_CTX),
        "--threads",
        str(N_THREADS),
        "--threads-batch",
        str(N_THREADS),
        "--batch-size",
        str(N_BATCH),
        "--jinja",
        "--reasoning",
        LLAMA_SERVER_REASONING,
        "--reasoning-budget",
        str(LLAMA_SERVER_REASONING_BUDGET),
        "--reasoning-format",
        LLAMA_SERVER_REASONING_FORMAT,
        "--n-gpu-layers",
        str(N_GPU_LAYERS),
    ]


async def ensure_llama_server_running() -> None:
    if await is_llama_server_ready():
        logger.info("Использую уже запущенный llama-server: %s", LLAMA_SERVER_BASE_URL)
        return

    model_path = ensure_valid_model_path()
    logger.info("Запуск локального llama-server для модели: %s", model_path)
    start_llama_server()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + LLAMA_SERVER_START_TIMEOUT
    while loop.time() < deadline:
        if await is_llama_server_ready():
            logger.info("llama-server готов: %s", LLAMA_SERVER_BASE_URL)
            return
        if LLAMA_SERVER_PROCESS is not None and LLAMA_SERVER_PROCESS.poll() is not None:
            raise RuntimeError(
                f"llama-server завершился с кодом {LLAMA_SERVER_PROCESS.poll()}. "
                f"Проверь лог: {LOG_DIR / 'llama_server.log'}"
            )
        await asyncio.sleep(1)

    raise RuntimeError(
        f"llama-server не поднялся за {LLAMA_SERVER_START_TIMEOUT} секунд. "
        f"Проверь лог: {LOG_DIR / 'llama_server.log'}"
    )


def stop_llama_server() -> None:
    global LLAMA_SERVER_PROCESS
    global LLAMA_SERVER_LOG_HANDLE

    if LLAMA_SERVER_PROCESS is not None and LLAMA_SERVER_PROCESS.poll() is None:
        LLAMA_SERVER_PROCESS.terminate()
        try:
            LLAMA_SERVER_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            LLAMA_SERVER_PROCESS.kill()
            LLAMA_SERVER_PROCESS.wait(timeout=5)

    LLAMA_SERVER_PROCESS = None

    if LLAMA_SERVER_LOG_HANDLE is not None:
        LLAMA_SERVER_LOG_HANDLE.close()
        LLAMA_SERVER_LOG_HANDLE = None


async def restart_llama_server(reason: str) -> None:
    logger.warning("Перезапускаю llama-server: %s", reason)
    stop_llama_server()
    await asyncio.sleep(LLAMA_SERVER_RESTART_DELAY_SECONDS)
    await ensure_llama_server_running()


def is_retryable_llama_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "не поднялся",
        "завершился с кодом",
        "cannot connect",
        "server disconnected",
        "connection reset",
        "connection refused",
        "broken pipe",
    )
    return any(marker in text for marker in markers)


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "нет"
    minutes, secs = divmod(max(0, seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}ч {minutes}м {secs}с"
    if minutes:
        return f"{minutes}м {secs}с"
    return f"{secs}с"


def format_clock_duration(seconds: int | None) -> str:
    total_seconds = max(0, int(seconds or 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration_for_humans(seconds: int | None) -> str:
    total_seconds = max(0, int(seconds or 0))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def render_live_banner(
    label: str,
    detail: str,
    *,
    progress: float | None = None,
    pulse_step: int | None = None,
    done: bool = False,
) -> str:
    title = "Все готово :3" if done else label
    body = f"{title}: {detail}" if detail else title
    width = LIVE_STATUS_BAR_WIDTH
    if console_supports_live_updates():
        try:
            columns = shutil.get_terminal_size((140, 24)).columns
        except OSError:
            columns = 140
        max_side_width = max(8, (columns - len(body) - 4) // 2)
        width = max(8, min(width, max_side_width))

    if done:
        filled = width
    elif progress is not None:
        clamped = max(0.0, min(1.0, float(progress)))
        filled = min(width, int(round(clamped * width)))
    else:
        step = max(0, int(pulse_step or 0))
        filled = step % (width + 1)

    padding = max(0, width - filled)
    left = ("-" * padding) + ("=" * filled)
    right = ("=" * filled) + ("-" * padding)
    return f"<{left}|{body}|{right}>"


def render_waiting_text(
    title: str,
    elapsed_seconds: int,
    extra_lines: list[str] | None = None,
) -> str:
    lines = [
        render_live_banner(
            title,
            format_clock_duration(elapsed_seconds),
            pulse_step=elapsed_seconds,
        )
    ]
    if extra_lines:
        lines.extend(["", *extra_lines])
    return "\n".join(lines)


def format_metric_percent(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.1f}%"


def read_proc_cpu_snapshot() -> tuple[int, int] | None:
    path = Path("/proc/stat")
    if not path.is_file():
        return None
    try:
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        return None
    parts = first_line.split()
    if len(parts) < 5 or parts[0] != "cpu":
        return None
    try:
        values = [int(item) for item in parts[1:]]
    except ValueError:
        return None
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return total, idle


def compute_cpu_percent(
    previous: tuple[int, int] | None,
    current: tuple[int, int] | None,
) -> float | None:
    if previous is None or current is None:
        return None
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    busy_ratio = 1.0 - (idle_delta / total_delta)
    return round(max(0.0, min(100.0, busy_ratio * 100.0)), 1)


def read_ram_percent() -> float | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    values: dict[str, int] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in raw_line:
                continue
            key, rest = raw_line.split(":", 1)
            number_text = rest.strip().split()[0]
            if number_text.isdigit():
                values[key.strip()] = int(number_text)
    except Exception:
        return None

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or total <= 0:
        return None
    if available is None:
        free = values.get("MemFree", 0)
        buffers = values.get("Buffers", 0)
        cached = values.get("Cached", 0)
        available = free + buffers + cached
    used_percent = (1.0 - (available / total)) * 100.0
    return round(max(0.0, min(100.0, used_percent)), 1)


def read_gpu_percent() -> float | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    values: list[float] = []
    for raw_line in (completed.stdout or "").splitlines():
        candidate = raw_line.strip().split()[0] if raw_line.strip() else ""
        try:
            values.append(float(candidate))
        except (TypeError, ValueError, IndexError):
            continue
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def build_long_think_metrics_payload() -> dict[str, Any]:
    return {
        "samples": 0,
        "cpu_sum": 0.0,
        "cpu_samples": 0,
        "ram_sum": 0.0,
        "ram_samples": 0,
        "gpu_sum": 0.0,
        "gpu_samples": 0,
    }


def ensure_long_think_metrics(job: dict[str, Any]) -> dict[str, Any]:
    metrics = job.get("metrics")
    if not isinstance(metrics, dict):
        metrics = build_long_think_metrics_payload()
        job["metrics"] = metrics
    for key, value in build_long_think_metrics_payload().items():
        metrics.setdefault(key, value)
    return metrics


def refresh_long_think_metric_averages(job: dict[str, Any]) -> None:
    metrics = ensure_long_think_metrics(job)
    job["average_cpu_percent"] = (
        round(float(metrics["cpu_sum"]) / int(metrics["cpu_samples"]), 1)
        if int(metrics["cpu_samples"]) > 0
        else None
    )
    job["average_ram_percent"] = (
        round(float(metrics["ram_sum"]) / int(metrics["ram_samples"]), 1)
        if int(metrics["ram_samples"]) > 0
        else None
    )
    job["average_gpu_percent"] = (
        round(float(metrics["gpu_sum"]) / int(metrics["gpu_samples"]), 1)
        if int(metrics["gpu_samples"]) > 0
        else None
    )


def add_long_think_metrics_sample(
    job: dict[str, Any],
    *,
    cpu_percent: float | None,
    ram_percent: float | None,
    gpu_percent: float | None,
) -> None:
    metrics = ensure_long_think_metrics(job)
    metrics["samples"] = int(metrics.get("samples", 0)) + 1
    if cpu_percent is not None:
        metrics["cpu_sum"] = float(metrics.get("cpu_sum", 0.0)) + float(cpu_percent)
        metrics["cpu_samples"] = int(metrics.get("cpu_samples", 0)) + 1
    if ram_percent is not None:
        metrics["ram_sum"] = float(metrics.get("ram_sum", 0.0)) + float(ram_percent)
        metrics["ram_samples"] = int(metrics.get("ram_samples", 0)) + 1
    if gpu_percent is not None:
        metrics["gpu_sum"] = float(metrics.get("gpu_sum", 0.0)) + float(gpu_percent)
        metrics["gpu_samples"] = int(metrics.get("gpu_samples", 0)) + 1
    refresh_long_think_metric_averages(job)


def compute_long_think_progress_percent(job: dict[str, Any]) -> int:
    status = str(job.get("status") or "")
    phase = str(job.get("phase") or "")
    if status == "completed":
        return 100

    duration_seconds = max(1, int(job.get("duration_seconds") or 1))
    elapsed_seconds = max(0, int(long_think_job_elapsed_seconds(job) or 0))
    progress = int(round((elapsed_seconds / duration_seconds) * 100))
    progress = max(0, min(99, progress))

    if str(job.get("template_outline") or "").strip():
        progress = max(progress, 10)

    planned_iterations = max(1, int(job.get("planned_iterations") or 1))
    completed_iterations = len(job.get("iterations", []))
    if completed_iterations > 0:
        iteration_progress = 12 + int(round((completed_iterations / planned_iterations) * 72))
        progress = max(progress, min(84, iteration_progress))

    if phase == "planning":
        progress = max(progress, 5)
    elif phase.startswith("iteration_"):
        progress = max(progress, 15)
    elif phase == "sleeping":
        progress = max(progress, 20)
    elif phase == "finalizing":
        final_window = max(1, int(job.get("final_buffer_seconds") or 1))
        remaining = max(0, int(long_think_job_remaining_seconds(job) or 0))
        final_elapsed = max(0, final_window - min(final_window, remaining))
        final_progress = 85 + int(round((final_elapsed / final_window) * 14))
        progress = max(progress, min(99, final_progress))

    if status in {"failed", "cancelled", "interrupted"}:
        return max(0, min(99, progress))
    return max(0, min(99, progress))


def build_long_think_progress_banner(job: dict[str, Any]) -> str:
    remaining = format_clock_duration(long_think_job_remaining_seconds(job))
    progress_percent = compute_long_think_progress_percent(job)
    return render_live_banner(
        "У модели осталось времени на ответ",
        f"{remaining}. Ответ готов на: {progress_percent}%",
        progress=progress_percent / 100 if progress_percent > 0 else 0.0,
        done=progress_percent >= 100,
    )


def refresh_long_think_progress_snapshot(job: dict[str, Any]) -> None:
    job["progress_percent"] = compute_long_think_progress_percent(job)
    job["progress_banner"] = build_long_think_progress_banner(job)


def truncate_middle_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n...\n[середина вырезана]\n...\n"
    if max_chars <= len(marker) + 2:
        return text[:max_chars]
    side = (max_chars - len(marker)) // 2
    return text[:side].rstrip() + marker + text[-side:].lstrip()


def parse_duration_spec(raw_value: str) -> int:
    value = raw_value.strip().lower()
    if not value:
        raise ValueError(
            "Нужна длительность вроде 4s, 45m, 2h, 1d, 00:10:00 или 00:00:10:00."
        )

    clock_parts = value.split(":")
    if 2 <= len(clock_parts) <= 4 and all(re.fullmatch(r"\d+", part) for part in clock_parts):
        numbers = [int(part) for part in clock_parts]
        if len(numbers) == 2:
            days = 0
            hours = 0
            minutes, seconds = numbers
        elif len(numbers) == 3:
            days = 0
            hours, minutes, seconds = numbers
        else:
            days, hours, minutes, seconds = numbers

        if seconds >= 60 or minutes >= 60:
            raise ValueError("В clock-формате секунды и минуты должны быть меньше 60.")
        if len(numbers) == 4 and hours >= 24:
            raise ValueError("В формате DD:HH:MM:SS часы должны быть меньше 24.")

        total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    else:
        unit_matches = list(re.finditer(r"(\d+)\s*([dhms])", value))
        if not unit_matches:
            raise ValueError(
                "Не понял длительность. Примеры: 4s, 45m, 6h, 1d, 00:30:00, 00:00:30:00."
            )
        consumed = "".join(match.group(0) for match in unit_matches).replace(" ", "")
        if consumed != value.replace(" ", ""):
            raise ValueError(
                "В длительности есть мусор. Примеры: 4s, 45m, 6h, 1d, 00:30:00, 00:00:30:00."
            )
        total_seconds = 0
        for match in unit_matches:
            amount = int(match.group(1))
            unit = match.group(2)
            if unit == "d":
                total_seconds += amount * 86400
            elif unit == "h":
                total_seconds += amount * 3600
            elif unit == "m":
                total_seconds += amount * 60
            else:
                total_seconds += amount

    if total_seconds < LONG_THINK_MIN_DURATION_SECONDS:
        raise ValueError(
            "Минимальная длительность: "
            f"{format_clock_duration(LONG_THINK_MIN_DURATION_SECONDS)} "
            f"({format_duration_for_humans(LONG_THINK_MIN_DURATION_SECONDS)})."
        )
    if total_seconds > LONG_THINK_MAX_DURATION_SECONDS:
        raise ValueError(
            "Максимальная длительность: "
            f"{format_clock_duration(LONG_THINK_MAX_DURATION_SECONDS)} "
            f"({format_duration_for_humans(LONG_THINK_MAX_DURATION_SECONDS)})."
        )
    return total_seconds


def choose_long_think_iteration_count(duration_seconds: int) -> int:
    if duration_seconds <= 0:
        return 0
    if duration_seconds < 15 * 60:
        return 1
    return min(
        LONG_THINK_MAX_ITERATIONS,
        max(1, round(duration_seconds / 1800)),
    )


def make_safe_job_slug(text: str, max_len: int = 48) -> str:
    lowered = text.strip().lower()
    cleaned = re.sub(r"[^a-zа-яё0-9]+", "-", lowered, flags=re.IGNORECASE)
    cleaned = cleaned.strip("-")
    if not cleaned:
        return "job"
    return cleaned[:max_len].strip("-") or "job"


def get_long_think_final_buffer_seconds(duration_seconds: int) -> int:
    if duration_seconds <= 1:
        return 1
    if duration_seconds <= 10:
        return max(1, duration_seconds // 4)
    if duration_seconds <= 60:
        return max(2, min(duration_seconds // 3, duration_seconds - 1))
    if duration_seconds <= 5 * 60:
        return max(5, min(duration_seconds // 3, duration_seconds - 5))

    reserved = int(duration_seconds * LONG_THINK_FINAL_BUFFER_RATIO)
    adaptive_min = min(LONG_THINK_FINAL_BUFFER_MIN_SECONDS, max(15, duration_seconds // 5))
    adaptive_max = min(LONG_THINK_FINAL_BUFFER_MAX_SECONDS, max(30, duration_seconds // 2))
    reserved = max(adaptive_min, reserved)
    reserved = min(adaptive_max, reserved)
    return min(reserved, max(30, duration_seconds // 2))


def get_long_think_work_phase_seconds(duration_seconds: int) -> int:
    return max(0, duration_seconds - get_long_think_final_buffer_seconds(duration_seconds))


def get_long_think_iteration_safety_seconds(duration_seconds: int) -> int:
    if duration_seconds <= 10:
        return 1
    if duration_seconds <= 60:
        return 2
    if duration_seconds <= 5 * 60:
        return min(10, max(2, duration_seconds // 12))
    return min(
        LONG_THINK_ITERATION_SAFETY_SECONDS,
        max(10, duration_seconds // 12),
    )


def choose_long_think_token_budget(window_seconds: int, default_tokens: int) -> int:
    safe_window = max(0, int(window_seconds))
    if safe_window <= 15 * 60:
        return min(default_tokens, 768)
    if safe_window <= 30 * 60:
        return min(default_tokens, 1024)
    if safe_window <= 60 * 60:
        return min(default_tokens, 1536)
    if safe_window <= 3 * 60 * 60:
        return min(default_tokens, 2048)
    return default_tokens


def round_long_think_duration_seconds(raw_seconds: Any) -> int:
    try:
        seconds = int(float(raw_seconds))
    except (TypeError, ValueError):
        seconds = LONG_THINK_MIN_DURATION_SECONDS
    seconds = max(LONG_THINK_MIN_DURATION_SECONDS, min(LONG_THINK_MAX_DURATION_SECONDS, seconds))
    if seconds <= 10 * 60:
        step = 60
    elif seconds <= 2 * 60 * 60:
        step = 5 * 60
    elif seconds <= 12 * 60 * 60:
        step = 15 * 60
    else:
        step = 30 * 60
    rounded = int(round(seconds / step)) * step
    return max(LONG_THINK_MIN_DURATION_SECONDS, min(LONG_THINK_MAX_DURATION_SECONDS, rounded or step))


def normalize_long_think_task_type(raw_value: Any) -> str:
    lowered = str(raw_value or "").strip().lower()
    if lowered in {"code", "coding", "dev"}:
        return "code"
    if lowered in {"text", "article", "summary", "research"}:
        return "text"
    if lowered in {"mixed", "hybrid", "code+text", "text+code"}:
        return "mixed"
    return "mixed"


def normalize_long_think_complexity(raw_value: Any) -> str:
    lowered = str(raw_value or "").strip().lower()
    if lowered in {"low", "simple", "easy"}:
        return "low"
    if lowered in {"medium", "normal", "moderate"}:
        return "medium"
    if lowered in {"high", "hard", "complex"}:
        return "high"
    if lowered in {"extreme", "huge", "very_high"}:
        return "extreme"
    return "medium"


def render_long_think_task_type_label(task_type: str) -> str:
    return {
        "code": "код / реализация",
        "text": "текст / исследование",
        "mixed": "смешанная задача",
    }.get(task_type, "смешанная задача")


def render_long_think_complexity_label(complexity: str) -> str:
    return {
        "low": "низкая",
        "medium": "средняя",
        "high": "высокая",
        "extreme": "очень высокая",
    }.get(complexity, "средняя")


def extract_json_object_from_text(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"(?is)^```(?:json)?\s*|\s*```$", "", str(text or "").strip())
    if not cleaned:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def normalize_long_think_duration_candidate(raw_value: Any, fallback_seconds: int) -> int:
    text = str(raw_value or "").strip()
    if not text:
        return round_long_think_duration_seconds(fallback_seconds)
    if re.fullmatch(r"\d+", text):
        return round_long_think_duration_seconds(int(text))
    try:
        return round_long_think_duration_seconds(parse_duration_spec(text))
    except ValueError:
        return round_long_think_duration_seconds(fallback_seconds)


def derive_long_think_deliverables_from_request(request_text: str, task_type: str) -> list[str]:
    if task_type == "code":
        return [
            "Разбить задачу на рабочие части и не потерять связь между ними.",
            "Собрать полный код без обрыва на середине логики.",
            "Довести результат до связного финального варианта, а не до огрызка.",
        ]
    if task_type == "text":
        return [
            "Собрать материал по теме без воды и служебки.",
            "Разложить ответ по понятной структуре.",
            "Закрыть задачу цельным финальным выводом.",
        ]
    return [
        "Собрать сильный каркас результата под задачу.",
        "Постепенно нарастить полноту и детали без развала структуры.",
        "Закрыть итог связным финальным ответом.",
    ]


def build_heuristic_long_think_plan(request_text: str) -> dict[str, Any]:
    lowered = request_text.lower()
    word_count = len(re.findall(r"\w+", request_text, flags=re.UNICODE))
    code_markers = (
        "код",
        "python",
        "typescript",
        "javascript",
        "node",
        "go",
        "golang",
        "rust",
        "fastapi",
        "react",
        "бот",
        "сервер",
        "api",
        "протокол",
        "tls",
        "архитектур",
        "рефактор",
    )
    text_markers = (
        "конспект",
        "статья",
        "отчёт",
        "исследован",
        "разбор",
        "обзор",
        "доклад",
        "документац",
    )
    huge_markers = (
        "полностью",
        "с нуля",
        "большой",
        "огромн",
        "несколько часов",
        "на сутки",
        "суточ",
        "5к",
        "5000",
        "тысяч",
        "полный проект",
        "максимально",
    )

    code_hits = sum(1 for marker in code_markers if marker in lowered)
    text_hits = sum(1 for marker in text_markers if marker in lowered)
    huge_hits = sum(1 for marker in huge_markers if marker in lowered)

    if code_hits and text_hits:
        task_type = "mixed"
    elif code_hits:
        task_type = "code"
    elif text_hits:
        task_type = "text"
    else:
        task_type = "mixed"

    score = 1.0
    score += min(3.5, word_count / 120.0)
    score += min(3.0, code_hits * 0.9)
    score += min(2.5, text_hits * 0.5)
    score += min(3.5, huge_hits * 1.1)
    if re.search(r"(?i)\b(?:[5-9]|\d{2,})\s*(?:k|к)\b", lowered):
        score += 2.5
    if re.search(r"(?i)\b5000\s*(?:строк|lines?)\b", lowered):
        score += 3.0

    recommended_seconds = 20 * 60 + int(score * 45 * 60)
    if task_type == "code":
        recommended_seconds = max(recommended_seconds, 90 * 60)
    elif task_type == "mixed":
        recommended_seconds = max(recommended_seconds, 60 * 60)
    else:
        recommended_seconds = max(recommended_seconds, 30 * 60)

    minimum_seconds = max(
        LONG_THINK_MIN_DURATION_SECONDS,
        int(recommended_seconds * (0.55 if task_type == "code" else 0.6)),
    )
    maximum_useful_seconds = max(
        recommended_seconds + 30 * 60,
        int(recommended_seconds * (1.9 if task_type == "code" else 1.7)),
    )

    recommended_seconds = round_long_think_duration_seconds(recommended_seconds)
    minimum_seconds = min(
        recommended_seconds,
        round_long_think_duration_seconds(minimum_seconds),
    )
    maximum_useful_seconds = max(
        recommended_seconds,
        round_long_think_duration_seconds(maximum_useful_seconds),
    )

    if score < 2.5:
        complexity = "low"
    elif score < 4.5:
        complexity = "medium"
    elif score < 6.5:
        complexity = "high"
    else:
        complexity = "extreme"

    estimated_iterations = max(
        1,
        choose_long_think_iteration_count(
            get_long_think_work_phase_seconds(recommended_seconds)
        ),
    )
    summary = truncate_text(" ".join(request_text.split()), 220) or "Большая задача без краткого описания"
    why = (
        "Срок поднял из-за объёма и требований к полноте. "
        "Для таких задач лучше иметь запас на несколько проходов и финальную сборку."
        if score >= 4.5
        else "Задача выглядит посильной, но всё равно требует хотя бы несколько проходов и финальную полировку."
    )
    return {
        "summary": summary,
        "task_type": task_type,
        "complexity": complexity,
        "recommended_duration_seconds": recommended_seconds,
        "minimum_duration_seconds": minimum_seconds,
        "maximum_useful_duration_seconds": maximum_useful_seconds,
        "estimated_iterations": estimated_iterations,
        "why": why,
        "deliverables": derive_long_think_deliverables_from_request(request_text, task_type),
        "source": "heuristic",
        "note": "Оценку собрал локально по эвристике, потому что структурированный ответ модели не взлетел.",
    }


def normalize_long_think_plan_payload(
    payload: dict[str, Any] | None,
    request_text: str,
) -> dict[str, Any]:
    heuristic = build_heuristic_long_think_plan(request_text)
    if not isinstance(payload, dict):
        return heuristic

    task_type = normalize_long_think_task_type(payload.get("task_type") or heuristic["task_type"])
    recommended_seconds = normalize_long_think_duration_candidate(
        payload.get("recommended_duration_seconds"),
        heuristic["recommended_duration_seconds"],
    )
    minimum_seconds = normalize_long_think_duration_candidate(
        payload.get("minimum_duration_seconds"),
        heuristic["minimum_duration_seconds"],
    )
    maximum_useful_seconds = normalize_long_think_duration_candidate(
        payload.get("maximum_useful_duration_seconds"),
        heuristic["maximum_useful_duration_seconds"],
    )
    minimum_seconds = min(minimum_seconds, recommended_seconds)
    maximum_useful_seconds = max(maximum_useful_seconds, recommended_seconds)

    estimated_iterations = parse_positive_int(payload.get("estimated_iterations"))
    if estimated_iterations is None:
        estimated_iterations = heuristic["estimated_iterations"]
    estimated_iterations = max(1, min(estimated_iterations, LONG_THINK_MAX_ITERATIONS))

    raw_deliverables = payload.get("deliverables")
    deliverables: list[str] = []
    if isinstance(raw_deliverables, list):
        for item in raw_deliverables:
            text = " ".join(str(item or "").split())
            if text:
                deliverables.append(truncate_text(text, 160))
    if not deliverables:
        deliverables = heuristic["deliverables"]

    summary = truncate_text(" ".join(str(payload.get("summary") or "").split()), 220)
    if not summary:
        summary = heuristic["summary"]

    why = truncate_text(" ".join(str(payload.get("why") or "").split()), 320)
    if not why:
        why = heuristic["why"]

    return {
        "summary": summary,
        "task_type": task_type,
        "complexity": normalize_long_think_complexity(
            payload.get("complexity") or heuristic["complexity"]
        ),
        "recommended_duration_seconds": recommended_seconds,
        "minimum_duration_seconds": minimum_seconds,
        "maximum_useful_duration_seconds": maximum_useful_seconds,
        "estimated_iterations": estimated_iterations,
        "why": why,
        "deliverables": deliverables[:5],
        "source": "model",
        "note": "",
    }


def build_long_think_plan_messages(request_text: str) -> list[dict[str, str]]:
    schema = (
        '{'
        '"summary":"короткое описание задачи",'
        '"task_type":"code|text|mixed",'
        '"complexity":"low|medium|high|extreme",'
        '"recommended_duration_seconds":7200,'
        '"minimum_duration_seconds":3600,'
        '"maximum_useful_duration_seconds":14400,'
        '"estimated_iterations":8,'
        '"why":"почему нужен такой срок",'
        '"deliverables":["что именно нужно закрыть","ещё пункт","ещё пункт"]'
        '}'
    )
    user_lines = [
        "Оцени задачу для режима deepthink локальной модели.",
        f"Запрос пользователя:\n{request_text}",
        f"Минимально допустимая длительность: {LONG_THINK_MIN_DURATION_SECONDS} секунд.",
        f"Максимально допустимая длительность: {LONG_THINK_MAX_DURATION_SECONDS} секунд.",
        "Нужно оценить, сколько времени реально стоит дать на несколько проходов и финальную полировку.",
        "Если задача явно большая, не жмись по времени.",
        "Верни только JSON по этой схеме:",
        schema,
    ]
    return [
        {"role": "system", "content": LONG_THINK_PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_lines)},
    ]


async def estimate_long_think_plan(request_text: str) -> dict[str, Any]:
    heuristic = build_heuristic_long_think_plan(request_text)
    try:
        reply_text, raw_reply, _ = await collect_model_reply(
            build_long_think_plan_messages(request_text),
            LONG_THINK_PLAN_MAX_TOKENS,
            label="deepplan",
            max_reply_chars=LONG_THINK_PLAN_MAX_REPLY_CHARS,
        )
    except Exception as exc:
        logger.warning("Не удалось получить deepplan-оценку от модели: %s", exc)
        return heuristic

    payload = extract_json_object_from_text(reply_text) or extract_json_object_from_text(raw_reply)
    if payload is None:
        logger.warning("Модель вернула кривой deepplan-JSON, падаю в эвристику.")
        return heuristic
    return normalize_long_think_plan_payload(payload, request_text)


def prune_pending_long_think_plans() -> None:
    now_dt = datetime.now().astimezone()
    stale_ids: list[str] = []
    for plan_id, plan in pending_long_think_plans.items():
        created_dt = parse_iso_datetime(str(plan.get("created_at") or ""))
        if created_dt is None:
            stale_ids.append(plan_id)
            continue
        if (now_dt - created_dt).total_seconds() > LONG_THINK_PLAN_KEEP_SECONDS:
            stale_ids.append(plan_id)
    for plan_id in stale_ids:
        pending_long_think_plans.pop(plan_id, None)


async def create_pending_long_think_plan(
    *,
    mode: str,
    owner_key: str,
    request_text: str,
    chat_id: int | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    plan = await estimate_long_think_plan(request_text)
    plan["plan_id"] = uuid.uuid4().hex
    plan["mode"] = mode
    plan["owner_key"] = owner_key
    plan["request_text"] = request_text.strip()
    plan["chat_id"] = chat_id
    plan["user_id"] = user_id
    plan["created_at"] = iso_now()
    prune_pending_long_think_plans()
    pending_long_think_plans[str(plan["plan_id"])] = plan
    return plan


def get_pending_long_think_plan(
    plan_id: str,
    *,
    owner_key: str | None = None,
) -> dict[str, Any] | None:
    prune_pending_long_think_plans()
    plan = pending_long_think_plans.get(plan_id)
    if plan is None:
        return None
    if owner_key is not None and str(plan.get("owner_key") or "") != owner_key:
        return None
    return plan


def pop_pending_long_think_plan(
    plan_id: str,
    *,
    owner_key: str | None = None,
) -> dict[str, Any] | None:
    plan = get_pending_long_think_plan(plan_id, owner_key=owner_key)
    if plan is None:
        return None
    pending_long_think_plans.pop(plan_id, None)
    return plan


def discard_pending_long_think_plan(plan_id: str, *, owner_key: str | None = None) -> None:
    if get_pending_long_think_plan(plan_id, owner_key=owner_key) is not None:
        pending_long_think_plans.pop(plan_id, None)


def build_deepplan_usage_text(prefix: str) -> str:
    return (
        f"Команда: {prefix} <запрос>\n"
        "Что делает:\n"
        "- Сначала просит модель оценить объём задачи и прикинуть адекватный срок для deepthink.\n"
        "- Показывает минимум, рекомендуемую длительность и верхнюю границу пользы.\n"
        "- После этого даёт тебе согласиться на запуск или отказаться.\n"
        "- Нужна именно для больших задач, когда ты не хочешь угадывать срок на глаз.\n\n"
        "Примеры:\n"
        f"- {prefix} Напиши большой план архитектуры для сервиса\n"
        f"- {prefix} Собери подробный конспект по TLS 1.3\n"
        f"- {prefix} Подготовь большой Python-проект с несколькими модулями"
    )


def build_mode_usage_text(prefix: str) -> str:
    return (
        f"Команда: {prefix} show|chat|code\n"
        "- show: показать текущий режим и привязки диалога.\n"
        "- chat: обычный разговорный режим.\n"
        "- code: инженерный режим с планом, файлами/патчами, тестами, командами проверки и упором на проектный контекст."
    )


def build_kb_usage_text(prefix: str) -> str:
    return (
        f"Команда: {prefix} list|add <путь>|remove <id>|clear|search <запрос>|on|off|show\n"
        "- list/show: показать документы локальной БЗ.\n"
        "- add <путь>: поставить файл или папку в очередь на индексацию.\n"
        "- remove <id>: удалить документ из БЗ.\n"
        "- clear: снести всю локальную БЗ.\n"
        "- search <запрос>: поиск по БЗ без отправки в модель.\n"
        "- on/off: включить или выключить использование БЗ для текущего диалога."
    )


def build_project_usage_text(prefix: str) -> str:
    return (
        f"Команда: {prefix} list|add <путь>|use <id>|off|show [id]|remove <id>|rescan <id>|search <запрос>\n"
        "- list: показать проиндексированные проекты.\n"
        "- add <путь>: поставить папку проекта в очередь на индексацию.\n"
        "- use <id>: привязать проект к текущему диалогу.\n"
        "- off: отвязать проект от текущего диалога.\n"
        "- show [id]: показать детали проекта.\n"
        "- remove <id>: удалить проектовый индекс.\n"
        "- rescan <id>: заново просканировать проект.\n"
        "- search <запрос>: поиск по активному проекту или по всем проектам."
    )


def build_model_usage_text(prefix: str) -> str:
    return (
        f"Команда: {prefix} list|current|use <id>|delete <id>|rescan\n"
        "- list: показать найденные `.gguf` модели.\n"
        "- current: показать текущую модель.\n"
        "- use <id>: переключить модель и при необходимости перезапустить llama-server.\n"
        "- delete <id>: удалить модель с диска, если это не текущая активная.\n"
        "- rescan: заново просканировать папки с моделями."
    )


def build_tasks_usage_text(prefix: str) -> str:
    return (
        f"Команда: {prefix} list|cancel <id>|delete <id>|clear\n"
        "- list: показать очередь фоновых задач.\n"
        "- cancel <id>: отменить задачу, если ещё можно.\n"
        "- delete <id>: убрать задачу из истории.\n"
        "- clear: удалить завершённые/ошибочные задачи из истории."
    )


def build_session_usage_text(prefix: str) -> str:
    return (
        f"Команда: {prefix} show|rename <имя>|export [json|md]\n"
        "- show: показать текущую terminal-сессию.\n"
        "- rename <имя>: вручную переименовать сессию.\n"
        "- export [json|md]: выгрузить историю сессии в файл."
    )


async def execute_mode_command(argument: str, dialog_key: str) -> str:
    normalized = str(argument or "").strip().lower()
    if not normalized or normalized == "show":
        return build_dialog_runtime_settings_text(dialog_key)
    if normalized in {"chat", "code"}:
        selected = set_dialog_response_mode(dialog_key, normalized)
        if selected == "code":
            return (
                "Режим переключил на `code`.\n"
                "Теперь стараюсь отвечать как инженер: план, изменения по файлам, тесты и команды проверки.\n"
                + build_dialog_runtime_settings_text(dialog_key)
            )
        return (
            "Режим переключил на `chat`.\n"
            + build_dialog_runtime_settings_text(dialog_key)
        )
    return build_mode_usage_text("/mode")


async def execute_kb_command(argument: str, dialog_key: str, owner_key: str) -> str:
    normalized = str(argument or "").strip()
    if not normalized:
        return build_kb_usage_text("/kb")
    parts = normalized.split(maxsplit=1)
    action = parts[0].lower()
    value = parts[1].strip() if len(parts) > 1 else ""

    if action in {"show", "list"}:
        return render_knowledge_docs_text()
    if action == "on":
        set_dialog_kb_enabled(dialog_key, True)
        return "Локальную БЗ включил для текущего диалога."
    if action == "off":
        set_dialog_kb_enabled(dialog_key, False)
        return "Локальную БЗ выключил для текущего диалога."
    if action == "search":
        if not value:
            return build_kb_usage_text("/kb")
        matches = search_chunks_in_payloads(list_knowledge_docs(), value, limit=KB_MAX_MATCHES)
        rendered = render_context_matches("Поиск по локальной БЗ:", matches, max_chars=KB_CONTEXT_MAX_CHARS)
        return rendered or "По локальной БЗ ничего внятного не нашёл."
    if action == "add":
        source_path = is_existing_local_path(value)
        if source_path is None:
            return "Путь не найден. Проверь, что файл или папка реально существуют."
        task = create_background_task(
            kind="kb_ingest",
            owner_key=owner_key,
            description=f"Индексация БЗ: {source_path.name}",
            payload={"source_path": str(source_path)},
        )
        return (
            f"Поставил индексацию БЗ в очередь.\n"
            f"Task: {str(task['task_id'])[:8]}\n"
            f"Источник: {source_path}"
        )
    if action == "remove":
        doc = resolve_knowledge_doc(value)
        if doc is None:
            return "Такой документ БЗ не найден."
        delete_knowledge_doc(str(doc["doc_id"]))
        return f"Документ БЗ [{str(doc['doc_id'])[:8]}] удалил."
    if action == "clear":
        removed = 0
        for doc in list_knowledge_docs():
            if delete_knowledge_doc(str(doc.get("doc_id") or "")):
                removed += 1
        return f"Локальную БЗ зачистил. Удалено документов: {removed}."
    return build_kb_usage_text("/kb")


async def execute_project_command(argument: str, dialog_key: str, owner_key: str) -> str:
    normalized = str(argument or "").strip()
    if not normalized:
        return build_project_usage_text("/project")
    parts = normalized.split(maxsplit=1)
    action = parts[0].lower()
    value = parts[1].strip() if len(parts) > 1 else ""

    if action == "list":
        return render_projects_text()
    if action == "off":
        set_dialog_active_project(dialog_key, "")
        return "Активный проект для текущего диалога отвязал."
    if action == "use":
        project = resolve_project_record(value)
        if project is None:
            return "Такой проект не найден."
        set_dialog_active_project(dialog_key, str(project["project_id"]))
        return (
            f"Привязал проект [{str(project['project_id'])[:8]}] {project.get('title') or '-'} "
            "к текущему диалогу."
        )
    if action == "show":
        project = resolve_project_record(value) if value else (
            get_project_record(str(get_dialog_runtime_settings(dialog_key).get("active_project_id") or ""))
        )
        if project is None:
            return "Проект не найден. Либо укажи id, либо сначала выбери активный проект."
        return render_project_record_detail(project)
    if action == "search":
        if not value:
            return build_project_usage_text("/project")
        active_project_id = str(get_dialog_runtime_settings(dialog_key).get("active_project_id") or "")
        payloads = [get_project_record(active_project_id)] if active_project_id else list_projects()
        payloads = [payload for payload in payloads if isinstance(payload, dict)]
        if not payloads:
            return "Проектов для поиска пока нет."
        matches = search_chunks_in_payloads(payloads, value, limit=PROJECT_CONTEXT_MAX_MATCHES)
        rendered = render_context_matches("Поиск по проекту:", matches, max_chars=PROJECT_CONTEXT_MAX_CHARS)
        return rendered or "По проектному индексу ничего внятного не нашёл."
    if action == "add":
        project_path = is_existing_local_path(value)
        if project_path is None or not project_path.is_dir():
            return "Нужна существующая папка проекта."
        task = create_background_task(
            kind="project_scan",
            owner_key=owner_key,
            description=f"Скан проекта: {project_path.name}",
            payload={"project_path": str(project_path)},
        )
        return (
            f"Поставил скан проекта в очередь.\n"
            f"Task: {str(task['task_id'])[:8]}\n"
            f"Путь: {project_path}"
        )
    if action == "rescan":
        project = resolve_project_record(value)
        if project is None:
            return "Такой проект не найден."
        task = create_background_task(
            kind="project_scan",
            owner_key=owner_key,
            description=f"Перескан проекта: {project.get('title') or '-'}",
            payload={
                "project_path": str(project.get("project_path") or ""),
                "replace_project_id": str(project.get("project_id") or ""),
            },
        )
        return f"Перескан проекта поставил в очередь. Task: {str(task['task_id'])[:8]}"
    if action == "remove":
        project = resolve_project_record(value)
        if project is None:
            return "Такой проект не найден."
        delete_project_record(str(project["project_id"]))
        return f"Проект [{str(project['project_id'])[:8]}] удалил."
    return build_project_usage_text("/project")


async def execute_model_command(argument: str) -> str:
    normalized = str(argument or "").strip()
    if not normalized:
        return build_model_usage_text("/model")
    parts = normalized.split(maxsplit=1)
    action = parts[0].lower()
    value = parts[1].strip() if len(parts) > 1 else ""

    if action in {"show", "list", "rescan"}:
        return render_models_text()
    if action == "current":
        current = resolve_model_candidate(MODEL_PATH)
        if current is None:
            return f"Текущая модель сломана или отсутствует: {MODEL_PATH}"
        return f"Текущая модель:\n- {current.name}\n- {current}"
    if action == "use":
        selected = resolve_model_path_selection(value)
        if selected is None:
            return "Такую модель не нашёл."
        activated = await activate_model_path(selected)
        return (
            f"Модель переключил на:\n- {activated.name}\n- {activated}\n"
            "Если llama-server был поднят, он уже перезапущен."
        )
    if action == "delete":
        selected = resolve_model_path_selection(value)
        if selected is None:
            return "Такую модель не нашёл."
        current = resolve_model_candidate(MODEL_PATH)
        if current is not None and model_identity_key(current) == model_identity_key(selected):
            return "Текущую активную модель удалять не дам. Сначала переключись на другую."
        try:
            selected.unlink()
        except Exception as exc:
            return f"Не удалось удалить модель: {exc}"
        return f"Модель `{selected.name}` удалил с диска."
    return build_model_usage_text("/model")


async def execute_tasks_command(argument: str, owner_key: str) -> str:
    normalized = str(argument or "").strip()
    if not normalized or normalized.lower() in {"show", "list"}:
        return render_background_tasks_text(owner_key)
    parts = normalized.split(maxsplit=1)
    action = parts[0].lower()
    value = parts[1].strip() if len(parts) > 1 else ""

    if action == "cancel":
        task = resolve_background_task(value, owner_key)
        if task is None:
            return "Такую задачу не нашёл."
        cancel_background_task(task)
        return f"Задачу [{str(task['task_id'])[:8]}] пометил на отмену."
    if action == "delete":
        task = resolve_background_task(value, owner_key)
        if task is None:
            return "Такую задачу не нашёл."
        if str(task.get("status") or "") in {"queued", "running"}:
            return "Активную задачу из истории не удаляю. Сначала отменяй."
        delete_background_task(str(task["task_id"]))
        return f"Задачу [{str(task['task_id'])[:8]}] удалил из истории."
    if action == "clear":
        removed = 0
        for task in list_background_tasks(owner_key):
            if str(task.get("status") or "") in {"completed", "failed", "cancelled", "interrupted"}:
                delete_background_task(str(task["task_id"]))
                removed += 1
        return f"Почистил историю фоновых задач. Удалено: {removed}."
    return build_tasks_usage_text("/tasks")


def render_terminal_session_details(session: dict[str, Any], dialog_key: str) -> str:
    history = normalize_terminal_session_history(session.get("history", []))
    settings = get_dialog_runtime_settings(dialog_key)
    project_id = str(settings.get("active_project_id") or "").strip()
    if project_id:
        project = get_project_record(project_id)
        project_label = (
            f"{project.get('title') or '-'} [{project_id[:8]}]"
            if project is not None
            else project_id[:8]
        )
    else:
        project_label = "нет"
    return (
        f"Terminal-сессия #{session.get('session_number')}:\n"
        f"- id: {session.get('session_id')}\n"
        f"- название: {session.get('title') or 'Без названия'}\n"
        f"- запросов: {session.get('request_count', 0)}\n"
        f"- сообщений в памяти: {len(history)}\n"
        f"- режим: {settings.get('response_mode')}\n"
        f"- локальная БЗ: {'вкл' if settings.get('kb_enabled') else 'выкл'}\n"
        f"- активный проект: {project_label}\n"
        f"- обновлена: {session.get('updated_at') or '-'}"
    )


def export_terminal_session(session: dict[str, Any], *, fmt: str) -> Path:
    exports_root = TERMINAL_SESSIONS_ROOT / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title_slug = make_safe_job_slug(str(session.get("title") or f"session-{session.get('session_number')}"))
    export_path = exports_root / f"session_{int(session['session_number']):04d}_{stamp}_{title_slug}.{fmt}"
    history = normalize_terminal_session_history(session.get("history", []))
    if fmt == "json":
        atomic_write_json(export_path, session)
        return export_path
    lines = [
        f"# Session #{session.get('session_number')}",
        "",
        f"Title: {session.get('title') or 'Без названия'}",
        f"Updated: {session.get('updated_at') or '-'}",
        "",
    ]
    for item in history:
        role = "USER" if item["role"] == "user" else "BOT"
        lines.append(f"## {role}")
        lines.append("")
        lines.append(item["content"])
        lines.append("")
    export_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return export_path


async def execute_session_command(
    argument: str,
    terminal_session: dict[str, Any],
    dialog_key: str,
) -> str:
    normalized = str(argument or "").strip()
    if not normalized or normalized.lower() == "show":
        return render_terminal_session_details(terminal_session, dialog_key)
    parts = normalized.split(maxsplit=1)
    action = parts[0].lower()
    value = parts[1].strip() if len(parts) > 1 else ""

    if action == "rename":
        if not value:
            return build_session_usage_text("/session")
        terminal_session["title"] = truncate_text(value, TERMINAL_SESSION_TITLE_CHARS)
        terminal_session["updated_at"] = iso_now()
        return f"Сессию переименовал в: {terminal_session['title']}"
    if action == "export":
        fmt = value.lower() if value else "md"
        if fmt not in {"json", "md"}:
            return build_session_usage_text("/session")
        export_path = export_terminal_session(terminal_session, fmt=fmt)
        return f"Сессию выгрузил в: {export_path}"
    return build_session_usage_text("/session")
def build_long_think_plan_text(
    plan: dict[str, Any],
    *,
    terminal_mode: bool,
) -> str:
    lines = [
        "Планировщик deepthink прикинул задачу.",
        f"План: {str(plan.get('plan_id') or '')[:8]}",
        f"Кратко: {plan.get('summary') or 'без сводки'}",
        f"Тип задачи: {render_long_think_task_type_label(str(plan.get('task_type') or 'mixed'))}",
        f"Сложность: {render_long_think_complexity_label(str(plan.get('complexity') or 'medium'))}",
        f"Минимум: {format_clock_duration(plan.get('minimum_duration_seconds'))}",
        f"Рекомендую: {format_clock_duration(plan.get('recommended_duration_seconds'))}",
        f"Верхняя граница пользы: {format_clock_duration(plan.get('maximum_useful_duration_seconds'))}",
        f"Стартовый ориентир проходов: {plan.get('estimated_iterations')}",
    ]
    deliverables = plan.get("deliverables") or []
    if deliverables:
        lines.extend(["", "Что надо закрыть:"])
        lines.extend(f"- {item}" for item in deliverables)
    why = str(plan.get("why") or "").strip()
    if why:
        lines.extend(["", f"Почему столько: {why}"])
    note = str(plan.get("note") or "").strip()
    if note:
        lines.extend(["", f"Заметка: {note}"])
    lines.extend(
        [
            "",
            (
                "Если устраивает, ответь на следующий вопрос и я запущу deepthink."
                if terminal_mode
                else "Если устраивает, жми кнопку запуска ниже."
            ),
        ]
    )
    return "\n".join(lines)


def build_deepplan_keyboard(plan_id: str, recommended_duration_seconds: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"Запустить на {format_clock_duration(recommended_duration_seconds)}",
                    callback_data=f"{DEEPPLAN_START_CALLBACK_PREFIX}{plan_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="Отмена",
                    callback_data=f"{DEEPPLAN_CANCEL_CALLBACK_PREFIX}{plan_id}",
                )
            ],
        ]
    )


def build_long_think_started_text(
    job: dict[str, Any],
    *,
    intro_line: str = "Запустил long-think job.",
    extra_note: str = "Модель получила ваш запрос! Обработано: 0%",
) -> str:
    lines = [
        intro_line,
        f"ID: {job['job_id'][:8]}",
        f"Длительность: {format_clock_duration(job['duration_seconds'])}",
        f"Рабочая фаза: {format_clock_duration(job['work_phase_seconds'])}",
        f"Финальное окно: {format_clock_duration(job['final_buffer_seconds'])}",
        f"Итераций запланировано: {job['planned_iterations']}",
        f"Папка: {job['artifact_dir']}",
    ]
    if extra_note.strip():
        lines.append(extra_note)
    return "\n".join(lines)


def append_long_think_started_event(
    *,
    job: dict[str, Any],
    request_text: str,
    source: str,
    chat: dict[str, Any] | None = None,
    user: dict[str, Any] | None = None,
    plan_id: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": iso_now(),
        "event": "long_think_started",
        "job_id": job["job_id"],
        "duration_seconds": job["duration_seconds"],
        "artifact_dir": job["artifact_dir"],
        "text": request_text,
        "mode": job["mode"],
        "owner_key": job["owner_key"],
        "source": source,
    }
    if chat is not None:
        payload["chat"] = chat
    if user is not None:
        payload["user"] = user
    if plan_id:
        payload["plan_id"] = plan_id
    append_jsonl(payload)


async def build_runtime_status_text() -> str:
    model_path_error: str | None = None
    try:
        model_path = ensure_valid_model_path()
    except Exception as exc:
        model_path = MODEL_PATH
        model_path_error = str(exc)
    ready = await is_llama_server_ready()
    process_state = get_llama_server_process_state()
    queue_state = get_model_runtime_snapshot()
    active_long_think_jobs = sum(
        1
        for job in long_think_jobs.values()
        if is_long_think_active_status(str(job.get("status") or ""))
    )
    queued_tasks = sum(
        1 for task in background_tasks.values() if str(task.get("status") or "") == "queued"
    )
    running_tasks = sum(
        1 for task in background_tasks.values() if str(task.get("status") or "") == "running"
    )

    lines = [
        "Статус системы:",
        f"AI: {'включен' if is_ai_enabled() else 'выключен'}",
        f"llama-server API: {'доступен' if ready else 'недоступен'}",
        f"llama-server процесс: {process_state['state']}",
        f"PID: {process_state['pid'] or '-'}",
        f"Код выхода: {process_state['exit_code'] if process_state['exit_code'] is not None else '-'}",
        f"Модель: {model_path.name}",
        f"MODEL_PATH: {model_path}",
        f"Ожидают в очереди: {queue_state['pending_requests']}",
        f"Активный запрос: {queue_state['active_label'] or '-'}",
        f"Длительность активного запроса: {format_clock_duration(queue_state['active_for_seconds'])}",
        f"Активных long-think job'ов: {active_long_think_jobs}",
        f"Фоновых задач в очереди: {queued_tasks}",
        f"Фоновых задач в работе: {running_tasks}",
        f"Историй в памяти: {len(dialog_histories)}",
    ]
    if model_path_error:
        lines.append(f"Ошибка модели: {model_path_error}")
    return "\n".join(lines)


def extract_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""

    choice = choices[0]

    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content

    text = choice.get("text")
    if isinstance(text, str):
        return text

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    return ""


def strip_visible_prefixes(text: str) -> str:
    stripped = text.lstrip()
    prefixes = (
        "ответ:",
        "финальный ответ:",
        "answer:",
        "final answer:",
    )
    lower_stripped = stripped.lower()
    for prefix in prefixes:
        if lower_stripped.startswith(prefix):
            return stripped[len(prefix) :].lstrip()
    return stripped


def normalize_visible_reply(text: str) -> str:
    text = text.replace("…", ".")
    text = re.sub(r"\.{3,}", ".", text)
    return text


def normalize_raw_model_reply(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"(?is)<think>\s*</think>", "", cleaned)
    cleaned = re.sub(r"(?is)<thinking>\s*</thinking>", "", cleaned)
    cleaned = cleaned.strip()
    cleaned = strip_visible_prefixes(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return normalize_visible_reply(cleaned)


def compress_brief_reply(text: str) -> str:
    cleaned = normalize_visible_reply(strip_visible_prefixes(text.strip()))
    if not cleaned:
        return ""

    cleaned = re.sub(r"(?m)^\s*[*#>\-]+\s*", "", cleaned)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if not cleaned:
        return ""

    sentence_matches = re.findall(r"[^.!?]+[.!?]?", cleaned)
    sentences = [match.strip() for match in sentence_matches if match.strip()]
    if sentences:
        cleaned = " ".join(sentences[:2]).strip()

    if len(cleaned) <= BRIEF_REPLY_MAX_CHARS:
        return cleaned

    shortened = cleaned[:BRIEF_REPLY_MAX_CHARS].rsplit(" ", 1)[0].strip()
    if not shortened:
        shortened = cleaned[:BRIEF_REPLY_MAX_CHARS].strip()
    if shortened and shortened[-1] not in ".!?":
        shortened += "."
    return shortened


def looks_truncated_reply(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith((".", "!", "?", '"', "'", ")", "]", "}", "»")):
        return False
    if stripped.endswith((",", ":", ";", "-", "—")):
        return True
    return stripped[-1].isalnum()


def extract_finish_reason(chunk: dict[str, Any]) -> str | None:
    choices = chunk.get("choices") or []
    if not choices:
        return None
    finish_reason = choices[0].get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def looks_like_reasoning(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "<think>",
        "</think>",
        "<thinking>",
        "</thinking>",
        "thinking",
        "thinking process",
        "the user is asking",
        "let me think",
        "let me craft",
        "as per instructions",
        "keep it brief",
        "reasoning",
        "chain of thought",
        "thought process",
        "analysis",
        "step 1",
        "step 2",
        "раздум",
        "рассужд",
        "ход мыслей",
        "анализ",
    )
    if any(marker in lowered for marker in markers):
        return True

    numbered_reasoning = re.search(
        r"(?im)^\s*1\.\s*(?:\*\*|__)?\s*(analyze|analysis|goal|option|step|шаг|анализ)",
        text,
    )
    return numbered_reasoning is not None


def looks_like_prompt_leak(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return False

    leak_markers = (
        "system prompt",
        "internal instructions",
        "user input:",
        "user asks:",
        "question:",
        "prompt:",
        "analyze the request",
        "analyze the input",
        "determine the response",
        "select best option",
        "example responses",
        "avoid meta-text",
        "needs to be short",
        "follow the instructions",
        "follow these instructions",
        "visible output",
        "visible final",
        "only final answer",
        "only final response",
        "no chain of thought",
        "chain of thought visible",
        "no reasoning visible",
        "complete sentences",
        "trailing ellipses",
        "ready final message",
        "ready final answer",
        "plain text only",
        "without tags",
        "no markdown formatting",
        "без chain of thought",
        "внутренние инструкции",
        "служебные инструкции",
        "цитируй system prompt",
    )
    if any(marker in lowered for marker in leak_markers):
        return True

    if re.match(r"^[\)\]\}\.,;:\-]+\s*[a-z]", stripped):
        return True

    if not re.search(r"[а-яё]", lowered):
        english_hits = re.findall(
            r"\b(?:no|without|ready|complete|final|visible|message|answer|sentences|ellipses|thought|reasoning|prompt|instruction|constraints)\b",
            lowered,
        )
        if len(english_hits) >= 3:
            return True

    return False


def is_meta_answer_candidate(text: str) -> bool:
    lowered = text.strip().lower()
    if looks_like_prompt_leak(text):
        return True
    meta_markers = (
        "final answer",
        "hidden thought",
        "продолжение ответа",
        "финальный ответ",
        "готовый ответ",
        "visible output",
        "видимый вывод",
    )
    return any(marker in lowered for marker in meta_markers)


def is_final_reply_candidate(text: str) -> bool:
    stripped = text.strip()
    meta_stripped = strip_meta_prefixes(stripped)
    lowered = meta_stripped.lower()

    if not stripped:
        return False
    if (
        looks_like_reasoning(stripped)
        or looks_like_prompt_leak(stripped)
        or is_meta_answer_candidate(stripped)
    ):
        return False
    if re.match(r"(?im)^\s*(?:[*-]|\d+\.)\s+", stripped):
        return False

    bad_prefixes = (
        "role:",
        "format:",
        "user:",
        "user input:",
        "user asks:",
        "question:",
        "assistant:",
        "language:",
        "style:",
        "request:",
        "analyze",
        "analysis",
    )
    if any(lowered.startswith(prefix) for prefix in bad_prefixes):
        return False

    bad_fragments = (
        "local telegram assistant",
        "as per instructions",
        "analyze the request",
        "analyze the input",
        "determine the response",
        "select best option",
        "example responses",
        "avoid meta-text",
        "keep it brief",
        "user input:",
        "user asks:",
        "the user is asking",
        "user says",
    )
    if any(fragment in lowered for fragment in bad_fragments):
        return False

    return True


def is_strict_meta_answer_candidate(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if looks_like_prompt_leak(stripped):
        return True
    if is_meta_answer_candidate(stripped):
        return True

    extra_markers = (
        "check constraints",
        "constraints:",
        "constraint check",
        "looks like plain text",
        "no bold/italics",
        "no tags?",
        "complete answer?",
        "detailed steps?",
        "russian? yes",
        "complete answer? yes",
        "no tags? yes",
        "detailed steps? yes",
    )
    if any(marker in lowered for marker in extra_markers):
        return True

    if re.search(r"(?im)^\*?\s*check constraints\s*:?\*?\s*$", stripped):
        return True

    yes_no_lines = re.findall(
        r"(?im)^\s*(?:[*-]\s*)?[a-zа-яё0-9 _/'\"()-]+\?\s*(?:yes|no|да|нет)\.?\s*$",
        stripped,
    )
    return len(yes_no_lines) >= 2


def is_strict_final_reply_candidate(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()

    if not is_final_reply_candidate(stripped):
        return False
    if re.match(r"(?im)^\s*\*[^*]+:\*\s*$", stripped):
        return False

    extra_bad_prefixes = (
        "check constraints",
        "constraints:",
    )
    if any(lowered.startswith(prefix) for prefix in extra_bad_prefixes):
        return False

    extra_bad_fragments = (
        "check constraints",
        "looks like plain text",
        "no bold/italics",
        "no tags?",
        "complete answer?",
        "detailed steps?",
    )
    if any(fragment in lowered for fragment in extra_bad_fragments):
        return False

    return not is_strict_meta_answer_candidate(stripped)


def needs_repair_pass(full_reply: str, raw_reply: str, brief_mode: bool = False) -> bool:
    if not ENABLE_REPAIR_PASS:
        return False
    if not raw_reply.strip():
        return False
    if not full_reply.strip():
        return True
    if brief_mode:
        return (
            looks_like_prompt_leak(full_reply)
            or looks_like_prompt_leak(raw_reply)
            or is_strict_meta_answer_candidate(full_reply)
        )
    return not is_strict_final_reply_candidate(full_reply)


def find_answer_tag_start(lower_text: str) -> int:
    candidates = [match.start() for match in re.finditer(re.escape("<answer>"), lower_text)]
    if not candidates:
        return -1

    ignore_context_markers = (
        "format",
        "template",
        "schema",
        "prompt",
        "instruction",
        "формат",
        "схема",
        "шаблон",
        "инструкц",
        "пример",
    )

    for idx in reversed(candidates):
        context = lower_text[max(0, idx - 140) : idx]
        if any(marker in context for marker in ignore_context_markers):
            continue
        return idx

    return -1


def strip_meta_prefixes(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*(?:[*>\-`_#]+\s*)+", "", cleaned)
    cleaned = re.sub(r"^\s*(?:\d+\.\s*)+", "", cleaned)
    cleaned = re.sub(r"^\s*(?:[*>\-`_#]+\s*)+", "", cleaned)
    cleaned = re.sub(r"^\s*(?:\*\*|__|`)+", "", cleaned)
    return cleaned.strip()


def is_relaxed_visible_line(text: str) -> bool:
    stripped = strip_visible_prefixes(text.strip())
    meta_stripped = strip_meta_prefixes(stripped)
    lowered = meta_stripped.lower()
    if not stripped:
        return False
    if looks_like_reasoning(stripped) or looks_like_prompt_leak(stripped):
        return False
    if is_strict_meta_answer_candidate(stripped):
        return False
    if re.match(
        r"(?im)^\s*(role|format|user|assistant|language|style|request|type|constraints)\s*:",
        meta_stripped,
    ):
        return False
    if re.match(
        r"(?im)^\s*(?:[*-]\s*)?[a-zа-яё0-9 _/'\"()-]+\?\s*(?:yes|no|да|нет)\.?\s*$",
        meta_stripped,
    ):
        return False
    if re.match(r"(?im)^\s*(user input|user asks|question|prompt)\s*:", meta_stripped):
        return False
    if lowered.startswith(
        (
            "user says:",
            "type:",
            "needs to be",
            "avoid meta-text",
            "example responses:",
            "option 1:",
            "option 2:",
            "option 3:",
            "analyze the request",
            "analyze the input",
            "determine the response",
            "select best option",
            "internal note",
            "internal notes",
            "note to self",
            "scratchpad",
            "inner note",
            "inner monologue",
            "thinking",
            "reasoning",
            "analysis",
        )
    ):
        return False
    if re.match(r'^[\"«].+[\"»].+\s-\s', meta_stripped):
        return False
    if re.match(r'(?i)^["\'«].*\([a-z][^)]*$', meta_stripped):
        return False
    if re.match(r'(?i)^["\'«].*\([a-z][^)]+\)', meta_stripped):
        return False
    if lowered in {
        "thinking process:",
        "reasoning:",
        "analysis:",
        "ответ:",
        "final answer:",
        "answer:",
    }:
        return False
    return re.search(r"[a-zа-яё0-9]", stripped, flags=re.IGNORECASE) is not None


def extract_relaxed_visible_reply(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?is)<think>.*?</think>", "\n", text)
    text = re.sub(r"(?is)<thinking>.*?</thinking>", "\n", text)
    text = text.replace("<think>", "\n").replace("</think>", "\n")
    text = text.replace("<thinking>", "\n").replace("</thinking>", "\n")

    kept_lines: list[str] = []
    previous_kept_was_blank = True

    for raw_line in text.splitlines():
        line = strip_visible_prefixes(raw_line.strip())
        if not line:
            if kept_lines and not previous_kept_was_blank:
                kept_lines.append("")
                previous_kept_was_blank = True
            continue

        if not is_relaxed_visible_line(line):
            continue

        kept_lines.append(line)
        previous_kept_was_blank = False

    cleaned = "\n".join(kept_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return normalize_visible_reply(cleaned)


def extract_visible_reply(raw_text: str, final: bool = False) -> str:
    lower_text = raw_text.lower()

    answer_open_tag = "<answer>"
    answer_close_tag = "</answer>"
    thinking_close_tags = ("</think>", "</thinking>")

    answer_open_idx = find_answer_tag_start(lower_text)
    if answer_open_idx != -1:
        answer_start = answer_open_idx + len(answer_open_tag)
        answer_close_idx = lower_text.find(answer_close_tag, answer_start)
        if answer_close_idx == -1:
            answer = raw_text[answer_start:]
        else:
            answer = raw_text[answer_start:answer_close_idx]
        cleaned = strip_visible_prefixes(answer.strip() if final else answer)
        if cleaned and not is_strict_meta_answer_candidate(cleaned):
            return normalize_visible_reply(cleaned)

    closing_think_hits = [
        (lower_text.rfind(tag), tag)
        for tag in thinking_close_tags
        if lower_text.rfind(tag) != -1
    ]
    if closing_think_hits:
        thinking_close_idx, closing_tag = max(
            closing_think_hits, key=lambda item: item[0]
        )
        answer = raw_text[thinking_close_idx + len(closing_tag) :]
        cleaned = strip_visible_prefixes(answer.strip() if final else answer)
        if cleaned and not is_strict_meta_answer_candidate(cleaned):
            return normalize_visible_reply(cleaned)

    answer_heading_matches = list(
        re.finditer(
            r"(?is)(?:^|\n)\s*(?:[#>*`-]+\s*)?(?:final answer|answer|финальный ответ|ответ)\s*[:\-]\s*",
            raw_text,
        )
    )
    if answer_heading_matches:
        match = answer_heading_matches[-1]
        answer = raw_text[match.end() :]
        cleaned = answer.strip() if final else answer.lstrip()
        if cleaned and not is_strict_meta_answer_candidate(cleaned):
            return normalize_visible_reply(cleaned)

    if looks_like_reasoning(raw_text) and not final:
        return ""

    if not final:
        return ""

    paragraphs = [
        strip_visible_prefixes(part.strip())
        for part in re.split(r"\n\s*\n+", raw_text)
        if part.strip()
    ]
    for candidate in reversed(paragraphs):
        if is_strict_final_reply_candidate(candidate):
            return normalize_visible_reply(candidate)

    lines = [
        strip_visible_prefixes(line.strip())
        for line in raw_text.splitlines()
        if line.strip()
    ]
    for candidate in reversed(lines):
        if is_strict_final_reply_candidate(candidate):
            return normalize_visible_reply(candidate)

    relaxed = extract_relaxed_visible_reply(raw_text)
    if relaxed:
        return relaxed

    return ""


async def stream_model_reply(messages: list[dict[str, str]], max_tokens: int) -> Any:
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repeat_penalty": REPEAT_PENALTY,
        "stop": STOP_STRINGS,
        "stream": True,
    }

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    finish_reason: str | None = None

    show_raw_in_console = (
        SHOW_MODEL_RAW
        and not is_console_live_line_active()
        and bool(getattr(sys.stdout, "isatty", lambda: False)())
    )
    if show_raw_in_console:
        print("\n[MODEL RAW] ", end="", flush=True)

    try:
        if not is_ai_enabled():
            raise RuntimeError("ИИ выключен через панель управления.")
        await ensure_llama_server_running()
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(LLAMA_SERVER_CHAT_URL, json=payload) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(
                        f"llama-server вернул HTTP {response.status}: {body[:500]}"
                    )

                while not response.content.at_eof():
                    raw_line = await response.content.readline()
                    if not raw_line:
                        break

                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    chunk_finish_reason = extract_finish_reason(chunk)
                    if chunk_finish_reason:
                        finish_reason = chunk_finish_reason

                    delta = extract_delta_text(chunk)
                    if not delta:
                        continue

                    if show_raw_in_console:
                        print(delta, end="", flush=True)
                    yield {"type": "token", "text": delta}
    finally:
        if show_raw_in_console:
            print("", flush=True)

    yield {"type": "done", "finish_reason": finish_reason}


async def stream_model_reply_resilient(
    messages: list[dict[str, str]],
    max_tokens: int,
) -> Any:
    if not is_ai_enabled():
        raise RuntimeError("ИИ выключен через панель управления.")

    max_attempts = 1 + (
        LLAMA_SERVER_MAX_RESTART_ATTEMPTS if LLAMA_SERVER_AUTO_RESTART else 0
    )
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            async for event in stream_model_reply(messages, max_tokens):
                yield event
            return
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise RuntimeError(f"Ошибка соединения с llama-server: {exc}") from exc
            logger.warning(
                "Сбой потока llama-server, пробую перезапуск: attempt=%s/%s error=%s",
                attempt,
                max_attempts,
                exc,
            )
            await restart_llama_server(f"stream error: {exc}")
        except RuntimeError as exc:
            last_error = exc
            if attempt >= max_attempts or not is_retryable_llama_error(exc):
                raise
            logger.warning(
                "Retry после ошибки llama-server: attempt=%s/%s error=%s",
                attempt,
                max_attempts,
                exc,
            )
            await restart_llama_server(str(exc))

    if last_error is not None:
        raise RuntimeError(str(last_error))


async def collect_model_reply_unlocked(
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    label: str = "collect",
    brief_mode: bool = False,
    max_reply_chars: int | None = MAX_MODEL_REPLY_CHARS,
) -> tuple[str, str, str | None]:
    raw_reply = ""
    finish_reason: str | None = None

    async for event in stream_model_reply_resilient(messages, max_tokens):
        if event["type"] == "done":
            finish_reason = event.get("finish_reason")
            continue
        raw_reply, truncated = append_reply_chunk(
            raw_reply,
            event["text"],
            max_reply_chars,
        )
        if truncated:
            logger.warning(
                "Ответ модели обрезан по max_reply_chars=%s: label=%s",
                max_reply_chars,
                label,
            )
            finish_reason = finish_reason or "length"
            break

    if USE_RAW_MODEL_REPLY:
        full_reply = normalize_raw_model_reply(raw_reply)
    else:
        full_reply = extract_visible_reply(raw_reply, final=True)
        if brief_mode:
            full_reply = compress_brief_reply(full_reply)
    return full_reply, raw_reply, finish_reason


async def collect_model_reply(
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    brief_mode: bool = False,
    label: str = "collect",
    max_reply_chars: int | None = MAX_MODEL_REPLY_CHARS,
) -> tuple[str, str, str | None]:
    if model_lock is None:
        raise RuntimeError("Лок модели не инициализирован.")

    async with acquire_model_slot(label):
        return await collect_model_reply_unlocked(
            messages,
            max_tokens,
            label=label,
            brief_mode=brief_mode,
            max_reply_chars=max_reply_chars,
        )


async def collect_dialog_reply(
    dialog_key: str,
    user_text: str,
    request_max_tokens: int,
    *,
    brief_mode: bool,
    label: str,
    allow_history_retry: bool = False,
) -> tuple[str, str, str | None, bool, bool]:
    had_history = bool(get_request_history_for_dialog(dialog_key, user_text))
    retried_with_clean_history = False

    while True:
        try:
            brief_fallback = get_brief_fallback_reply(user_text) if brief_mode else None
            used_brief_fallback = False
            messages = build_messages(dialog_key, user_text)
            full_reply, raw_reply, finish_reason = await collect_model_reply(
                messages,
                request_max_tokens,
                brief_mode=brief_mode,
                label=label if not retried_with_clean_history else f"{label}:retry",
            )

            if brief_mode and not USE_RAW_MODEL_REPLY:
                full_reply = compress_brief_reply(full_reply)
                if brief_fallback and (
                    not full_reply.strip()
                    or looks_like_reasoning(raw_reply)
                    or looks_like_prompt_leak(full_reply)
                    or not is_strict_final_reply_candidate(full_reply)
                ):
                    logger.info(
                        "Использую brief-fallback после первого прохода: label=%s",
                        label,
                    )
                    full_reply = brief_fallback
                    used_brief_fallback = True

            if (
                not USE_RAW_MODEL_REPLY
                and brief_mode
                and not used_brief_fallback
                and not brief_fallback
                and finish_reason == "length"
                and looks_truncated_reply(full_reply)
            ):
                logger.info("Перегенерация короткого ответа: label=%s", label)
                retry_raw_reply = ""
                retry_messages = build_brief_retry_messages(user_text)

                async for event in stream_model_reply_resilient(retry_messages, request_max_tokens):
                    if event["type"] == "done":
                        continue
                    retry_raw_reply, truncated = append_reply_chunk(
                        retry_raw_reply, event["text"]
                    )
                    if truncated:
                        logger.warning(
                            "Retry-ответ обрезан по MAX_MODEL_REPLY_CHARS=%s: label=%s",
                            MAX_MODEL_REPLY_CHARS,
                            label,
                        )
                        break

                retried_reply = extract_visible_reply(retry_raw_reply, final=True)
                retried_reply = compress_brief_reply(retried_reply)
                if retried_reply.strip():
                    raw_reply = retry_raw_reply
                    full_reply = retried_reply
                elif brief_fallback:
                    logger.info(
                        "Использую brief-fallback после retry: label=%s",
                        label,
                    )
                    full_reply = brief_fallback
                    used_brief_fallback = True

            if (not USE_RAW_MODEL_REPLY) and (not brief_mode) and needs_repair_pass(
                full_reply, raw_reply, brief_mode=brief_mode
            ):
                logger.info("Запускаю repair-pass: label=%s", label)
                repair_raw_reply = ""
                repair_messages = build_repair_messages(dialog_key, user_text)

                async for event in stream_model_reply_resilient(repair_messages, request_max_tokens):
                    if event["type"] == "done":
                        continue
                    repair_raw_reply, truncated = append_reply_chunk(
                        repair_raw_reply, event["text"]
                    )
                    if truncated:
                        logger.warning(
                            "Repair-ответ обрезан по MAX_MODEL_REPLY_CHARS=%s: label=%s",
                            MAX_MODEL_REPLY_CHARS,
                            label,
                        )
                        break

                repaired_reply = extract_visible_reply(repair_raw_reply, final=True)
                if brief_mode:
                    repaired_reply = compress_brief_reply(repaired_reply)
                    if not repaired_reply.strip() and brief_fallback:
                        logger.info(
                            "Использую brief-fallback после repair: label=%s",
                            label,
                        )
                        repaired_reply = brief_fallback
                        used_brief_fallback = True
                if repaired_reply.strip():
                    raw_reply = repair_raw_reply
                    full_reply = repaired_reply

            return (
                full_reply,
                raw_reply,
                finish_reason,
                used_brief_fallback,
                retried_with_clean_history,
            )
        except Exception as exc:
            if not allow_history_retry or not had_history or retried_with_clean_history:
                raise
            logger.warning(
                "Запрос упал на истории диалога, пробую заново с чистой памятью: label=%s error=%s",
                label,
                exc,
            )
            reset_dialog(dialog_key)
            retried_with_clean_history = True


def render_standard_waiting_text(elapsed_seconds: int) -> str:
    return render_waiting_text(THINKING_PLACEHOLDER_TEXT, elapsed_seconds)


class TelegramWaitIndicator:
    def __init__(
        self,
        message: Message,
        render_text,
        *,
        dialog_key: str | None = None,
    ) -> None:
        self.message = message
        self.render_text = render_text
        self.dialog_key = dialog_key
        self.started_at = datetime.now()
        self.rendered_text = ""
        self.task: asyncio.Task[Any] | None = None
        self.edit_lock = asyncio.Lock()

    def elapsed_seconds(self) -> int:
        return max(0, int((datetime.now() - self.started_at).total_seconds()))

    async def start(self) -> None:
        await self.refresh(force=True)
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(WAIT_INDICATOR_INTERVAL_SECONDS)
                await self.refresh()
        except asyncio.CancelledError:
            return

    async def refresh(self, force: bool = False) -> None:
        text = self.render_text(self.elapsed_seconds())
        async with self.edit_lock:
            if not force and text == self.rendered_text:
                return
            self.message = await safe_edit_message(self.message, text)
            self.rendered_text = text
            if self.dialog_key is not None:
                track_bot_message(self.dialog_key, self.message)

    async def stop(self) -> None:
        if self.task is None:
            return
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        self.task = None


class TerminalWaitIndicator:
    def __init__(self, label: str = "Ожидание") -> None:
        self.label = label
        self.started_at = datetime.now()
        self.task: asyncio.Task[Any] | None = None

    def elapsed_seconds(self) -> int:
        return max(0, int((datetime.now() - self.started_at).total_seconds()))

    async def start(self) -> None:
        self.started_at = datetime.now()
        await self._render()
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(WAIT_INDICATOR_INTERVAL_SECONDS)
                await self._render()
        except asyncio.CancelledError:
            return

    async def _render(self) -> None:
        text = render_live_banner(
            self.label,
            format_clock_duration(self.elapsed_seconds()),
            pulse_step=self.elapsed_seconds(),
        )
        update_console_live_line(text)

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        clear_console_live_line_permanently()


class StreamingTelegramEditor:
    def __init__(self, message: Message, dialog_key: str) -> None:
        self.message = message
        self.dialog_key = dialog_key
        self.current_message: Message | None = None
        self.rendered_segment = ""
        self.wait_indicator: TelegramWaitIndicator | None = None

    async def start(self) -> None:
        self.current_message = await self.message.answer(render_standard_waiting_text(0))
        self.rendered_segment = render_standard_waiting_text(0)
        track_bot_message(self.dialog_key, self.current_message)
        self.wait_indicator = TelegramWaitIndicator(
            self.current_message,
            render_standard_waiting_text,
            dialog_key=self.dialog_key,
        )
        await self.wait_indicator.start()

    async def flush(self, full_text: str, final: bool = False) -> None:
        if self.current_message is None:
            await self.start()

        assert self.current_message is not None
        if final and self.wait_indicator is not None:
            await self.wait_indicator.stop()
            self.current_message = self.wait_indicator.message
            self.wait_indicator = None

        text = full_text.strip() if final else full_text
        if not text:
            text = "Модель ничего не вернула." if final else render_standard_waiting_text(0)

        reply_markup = build_response_keyboard() if final else None

        if not final or len(text) <= TELEGRAM_SEGMENT_LIMIT:
            await self._edit_if_changed(text, reply_markup=reply_markup)
            return

        chunks = [
            text[i : i + TELEGRAM_SEGMENT_LIMIT]
            for i in range(0, len(text), TELEGRAM_SEGMENT_LIMIT)
        ]
        await self._edit_if_changed(chunks[0], reply_markup=reply_markup)
        for chunk in chunks[1:]:
            self.current_message = await self.message.answer(
                chunk, reply_markup=reply_markup
            )
            self.rendered_segment = chunk
            track_bot_message(self.dialog_key, self.current_message)

    async def _edit_if_changed(
        self,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        if self.current_message is None or text == self.rendered_segment:
            return

        try:
            result = await self.current_message.edit_text(
                text, reply_markup=reply_markup
            )
            if isinstance(result, Message):
                self.current_message = result
        except TelegramBadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise

        self.rendered_segment = text
        track_bot_message(self.dialog_key, self.current_message)

    async def show_error(self, text: str) -> None:
        if self.wait_indicator is not None:
            await self.wait_indicator.stop()
            self.current_message = self.wait_indicator.message
            self.wait_indicator = None
        if self.current_message is None:
            sent = await self.message.answer(text)
            track_bot_message(self.dialog_key, sent)
            return

        try:
            result = await self.current_message.edit_text(
                text, reply_markup=build_response_keyboard()
            )
            if isinstance(result, Message):
                self.current_message = result
        except Exception:
            sent = await self.message.answer(text, reply_markup=build_response_keyboard())
            track_bot_message(self.dialog_key, sent)
        self.rendered_segment = text
        track_bot_message(self.dialog_key, self.current_message)


def is_long_think_active_status(status: str) -> bool:
    return status in {"queued", "running", "sleeping", "finalizing"}


def get_systemd_run_context() -> tuple[list[str], str] | None:
    systemd_run = shutil.which("systemd-run")
    if not systemd_run:
        return None

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return [systemd_run], "system"

    return [systemd_run, "--user"], "user"


def is_systemd_unit_active(unit_name: str | None, scope: str | None) -> bool:
    unit = str(unit_name or "").strip()
    if not unit:
        return False

    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False

    command = [systemctl]
    if str(scope or "") == "user":
        command.append("--user")
    command.extend(["is-active", "--quiet", unit])
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def stop_systemd_unit(unit_name: str | None, scope: str | None) -> None:
    unit = str(unit_name or "").strip()
    if not unit:
        return

    systemctl = shutil.which("systemctl")
    if not systemctl:
        return

    command = [systemctl]
    if str(scope or "") == "user":
        command.append("--user")
    command.extend(["stop", unit])
    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def get_long_think_owner_key(
    mode: str,
    *,
    user_id: int | None = None,
    chat_id: int | None = None,
    terminal_session_number: int | None = None,
) -> str:
    if mode == "telegram":
        return f"telegram:{chat_id or 0}:{user_id or 0}"
    return get_terminal_session_dialog_key(terminal_session_number or 0)


def touch_long_think_job(job_id: str) -> None:
    long_think_job_order.pop(job_id, None)
    long_think_job_order[job_id] = None


def build_long_think_artifact_dir(job_id: str, request_text: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = make_safe_job_slug(request_text)
    return LONG_THINK_ROOT / f"{stamp}_{job_id[:8]}_{slug}"


def trim_long_think_context(text: str) -> str:
    return truncate_middle_text(text.strip(), LONG_THINK_CONTEXT_CHARS)


def summarize_long_think_iteration(text: str, max_chars: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    shortened = cleaned[: max_chars - 1].rsplit(" ", 1)[0].strip()
    return (shortened or cleaned[: max_chars - 1]).rstrip(".,;:-") + "…"


def long_think_job_elapsed_seconds(job: dict[str, Any]) -> int | None:
    start_dt = parse_iso_datetime(job.get("started_at") or job.get("created_at"))
    if start_dt is None:
        return None
    end_dt = parse_iso_datetime(job.get("completed_at"))
    if end_dt is None and not is_long_think_active_status(str(job.get("status") or "")):
        end_dt = parse_iso_datetime(job.get("updated_at"))
    reference_dt = end_dt or datetime.now().astimezone()
    return max(0, int((reference_dt - start_dt).total_seconds()))


def long_think_job_remaining_seconds(job: dict[str, Any]) -> int | None:
    deadline_dt = parse_iso_datetime(job.get("deadline_at"))
    if deadline_dt is None or not is_long_think_active_status(str(job.get("status") or "")):
        return None
    return max(0, int((deadline_dt - datetime.now().astimezone()).total_seconds()))


def long_think_time_until_finalization_seconds(job: dict[str, Any]) -> int | None:
    finalization_dt = parse_iso_datetime(job.get("finalization_starts_at"))
    if finalization_dt is None or not is_long_think_active_status(str(job.get("status") or "")):
        return None
    return max(0, int((finalization_dt - datetime.now().astimezone()).total_seconds()))


def serialize_long_think_job(job: dict[str, Any]) -> dict[str, Any]:
    refresh_long_think_metric_averages(job)
    refresh_long_think_progress_snapshot(job)
    payload = {
        "job_id": job["job_id"],
        "mode": job["mode"],
        "owner_key": job["owner_key"],
        "status": job["status"],
        "phase": job.get("phase"),
        "request_text": job["request_text"],
        "duration_seconds": job["duration_seconds"],
        "planned_iterations": job.get("planned_iterations", 0),
        "completed_iterations": len(job.get("iterations", [])),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "deadline_at": job.get("deadline_at"),
        "finalization_starts_at": job.get("finalization_starts_at"),
        "completed_at": job.get("completed_at"),
        "updated_at": job.get("updated_at"),
        "cancel_requested": bool(job.get("cancel_requested")),
        "artifact_dir": job.get("artifact_dir"),
        "result_path": job.get("result_path"),
        "error": job.get("error"),
        "note": job.get("note"),
        "chat_id": job.get("chat_id"),
        "user_id": job.get("user_id"),
        "process_id": job.get("process_id"),
        "systemd_unit": job.get("systemd_unit"),
        "systemd_scope": job.get("systemd_scope"),
        "final_buffer_seconds": job.get("final_buffer_seconds"),
        "work_phase_seconds": job.get("work_phase_seconds"),
        "template_outline": job.get("template_outline", ""),
        "template_ready_at": job.get("template_ready_at"),
        "latest_draft": job.get("latest_draft", ""),
        "final_answer": job.get("final_answer", ""),
        "answer_completed_fully": bool(job.get("answer_completed_fully")),
        "final_finish_reason": job.get("final_finish_reason", ""),
        "progress_percent": job.get("progress_percent", 0),
        "progress_banner": job.get("progress_banner", ""),
        "average_cpu_percent": job.get("average_cpu_percent"),
        "average_ram_percent": job.get("average_ram_percent"),
        "average_gpu_percent": job.get("average_gpu_percent"),
        "metrics": ensure_long_think_metrics(job),
        "iterations": job.get("iterations", []),
        "elapsed_seconds": long_think_job_elapsed_seconds(job),
        "remaining_seconds": long_think_job_remaining_seconds(job),
        "until_finalization_seconds": long_think_time_until_finalization_seconds(job),
    }
    return payload


def persist_long_think_job(job: dict[str, Any], *, final: bool = False) -> None:
    LONG_THINK_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(job["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "result.json"
    job["result_path"] = str(result_path)
    job["updated_at"] = iso_now()
    payload = serialize_long_think_job(job)
    (artifact_dir / "state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if final or job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def mark_interrupted_long_think_jobs() -> None:
    if not LONG_THINK_ROOT.is_dir():
        return
    for state_path in LONG_THINK_ROOT.glob("*/state.json"):
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(payload.get("status") or "")
        if not is_long_think_active_status(status):
            continue
        unit_name = str(payload.get("systemd_unit") or "").strip()
        unit_scope = str(payload.get("systemd_scope") or "").strip()
        if unit_name and is_systemd_unit_active(unit_name, unit_scope):
            continue
        process_id = int(payload.get("process_id") or 0)
        if process_id > 0:
            try:
                os.kill(process_id, 0)
                continue
            except OSError:
                pass
        payload["status"] = "interrupted"
        payload["phase"] = "interrupted"
        payload["error"] = "Процесс бота был остановлен до завершения long-think job."
        payload["updated_at"] = iso_now()
        payload["completed_at"] = payload["updated_at"]
        try:
            state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result_path = state_path.with_name("result.json")
            result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            continue


def hydrate_long_think_job(payload: dict[str, Any]) -> dict[str, Any]:
    duration_seconds = max(1, int(payload.get("duration_seconds") or 1))
    final_buffer_seconds = max(
        0,
        int(
            payload.get("final_buffer_seconds")
            or get_long_think_final_buffer_seconds(duration_seconds)
        ),
    )
    work_phase_seconds = max(
        0,
        int(
            payload.get("work_phase_seconds")
            or get_long_think_work_phase_seconds(duration_seconds)
        ),
    )
    iterations = payload.get("iterations")
    if not isinstance(iterations, list):
        iterations = []

    job = {
        "job_id": str(payload.get("job_id") or "").strip(),
        "mode": str(payload.get("mode") or "terminal").strip() or "terminal",
        "owner_key": str(payload.get("owner_key") or "").strip(),
        "status": str(payload.get("status") or "queued").strip() or "queued",
        "phase": str(payload.get("phase") or "queued").strip() or "queued",
        "request_text": str(payload.get("request_text") or "").strip(),
        "duration_seconds": duration_seconds,
        "planned_iterations": max(0, int(payload.get("planned_iterations") or 0)),
        "created_at": payload.get("created_at") or iso_now(),
        "started_at": payload.get("started_at"),
        "deadline_at": payload.get("deadline_at"),
        "finalization_starts_at": payload.get("finalization_starts_at"),
        "completed_at": payload.get("completed_at"),
        "updated_at": payload.get("updated_at") or iso_now(),
        "cancel_requested": bool(payload.get("cancel_requested")),
        "artifact_dir": str(payload.get("artifact_dir") or ""),
        "result_path": str(payload.get("result_path") or ""),
        "error": str(payload.get("error") or ""),
        "note": str(payload.get("note") or ""),
        "chat_id": payload.get("chat_id"),
        "user_id": payload.get("user_id"),
        "process_id": int(payload.get("process_id") or 0),
        "systemd_unit": str(payload.get("systemd_unit") or ""),
        "systemd_scope": str(payload.get("systemd_scope") or ""),
        "bot": None,
        "task": None,
        "final_buffer_seconds": final_buffer_seconds,
        "work_phase_seconds": work_phase_seconds,
        "template_outline": str(payload.get("template_outline") or ""),
        "template_ready_at": payload.get("template_ready_at"),
        "iterations": iterations,
        "latest_draft": str(payload.get("latest_draft") or ""),
        "final_answer": str(payload.get("final_answer") or ""),
        "answer_completed_fully": bool(payload.get("answer_completed_fully")),
        "final_finish_reason": str(payload.get("final_finish_reason") or ""),
        "progress_percent": max(0, int(payload.get("progress_percent") or 0)),
        "progress_banner": str(payload.get("progress_banner") or ""),
        "progress_message_text": "",
        "progress_message": None,
        "progress_task": None,
        "metrics": payload.get("metrics"),
        "average_cpu_percent": payload.get("average_cpu_percent"),
        "average_ram_percent": payload.get("average_ram_percent"),
        "average_gpu_percent": payload.get("average_gpu_percent"),
        "metrics_task": None,
    }
    ensure_long_think_metrics(job)
    refresh_long_think_metric_averages(job)
    refresh_long_think_progress_snapshot(job)
    return job


def load_long_think_job_from_disk(job_id: str) -> dict[str, Any] | None:
    if not LONG_THINK_ROOT.is_dir():
        return None

    normalized = str(job_id or "").strip().lower()
    if not normalized:
        return None

    for state_path in LONG_THINK_ROOT.glob("*/state.json"):
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload_job_id = str(payload.get("job_id") or "").strip().lower()
        if payload_job_id != normalized:
            continue
        return hydrate_long_think_job(payload)
    return None


def resolve_long_think_runtime_job(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = str(payload.get("job_id") or "").strip()
    runtime_job = long_think_jobs.get(job_id)
    if runtime_job is not None:
        return runtime_job
    return hydrate_long_think_job(payload)


def get_active_long_think_job(owner_key: str) -> dict[str, Any] | None:
    for payload in list_long_think_jobs_for_owner(owner_key):
        if is_long_think_active_status(str(payload.get("status") or "")):
            return resolve_long_think_runtime_job(payload)
    return None


def find_long_think_job_for_owner(
    owner_key: str,
    raw_job_id: str | None = None,
) -> dict[str, Any] | None:
    normalized = (raw_job_id or "").strip().lower()
    jobs = list_long_think_jobs_for_owner(owner_key)
    if normalized:
        for payload in jobs:
            if str(payload.get("job_id") or "").lower().startswith(normalized):
                return resolve_long_think_runtime_job(payload)
        return None
    return next(
        (
            resolve_long_think_runtime_job(payload)
            for payload in jobs
            if is_long_think_active_status(str(payload.get("status") or ""))
        ),
        None,
    )


def build_duration_syntax_text() -> str:
    return (
        "Как указывать длительность:\n"
        "1. Суффиксами: 4d, 4h, 4m, 4s, 1h30m, 2d 6h 15m 10s.\n"
        "2. Через двоеточия: 00:30:00 (часы:минуты:секунды), 05:20 (минуты:секунды).\n"
        "3. Полный clock-формат: 01:12:30:45 (дни:часы:минуты:секунды)."
    )


def build_deepthink_usage_text(prefix: str) -> str:
    return (
        f"Команда: {prefix} <длительность> <запрос>\n"
        "Что делает:\n"
        "- Запускает отдельный long-think job в фоне по сроку, который ты выбрал сам.\n"
        "- В terminal-режиме старается пережить выход из SSH и продолжить работу отдельно от твоей сессии.\n"
        "- Для long-think не режет сам ответ по символьному лимиту обычного чата.\n"
        "- Сначала строит шаблон результата, потом заполняет его проходами и доводит до финала.\n"
        "- Во время работы считает прогресс и собирает среднюю нагрузку CPU / RAM / GPU.\n"
        "- В рабочей фазе бот постепенно улучшает черновик.\n"
        "- Под конец оставляет отдельное окно на финальный вывод, чтобы не оборваться посреди ответа.\n"
        "- По завершении сохраняет результат в JSON внутри deep_think_jobs/<job_id>/result.json.\n\n"
        f"{build_duration_syntax_text()}\n\n"
        "Примеры:\n"
        f"- {prefix} 4m Напиши краткий конспект по FastAPI\n"
        f"- {prefix} 8h Собери большой план архитектуры\n"
        f"- {prefix} 00:30:00 Напиши подробный шаблон проекта\n"
        f"- {prefix} 00:01:00:00 Подготовь суточный исследовательский отчёт"
    )


def build_port_manual_text(mode: str) -> str:
    sections = [
        "Что умеет этот порт:\n"
        "- Обычный диалог с локальной GGUF-моделью через llama.cpp / llama-server.\n"
        "- Память диалога с ручным сбросом.\n"
        "- Режимы ответа chat/code.\n"
        "- Локальная БЗ, которую можно использовать опционально.\n"
        "- Работа с проектами: индекс, выбор активного проекта и поиск по нему.\n"
        "- Очередь фоновых задач для индексации БЗ и проектов.\n"
        "- Менеджер локальных GGUF-моделей.\n"
        "- Пакетный режим /ineedmore для нескольких независимых запросов.\n"
        "- Планировщик /deepplan, который сначала оценивает задачу и рекомендует срок.\n"
        "- Long-think / deepthink для длительной проработки больших задач.\n"
        "- Просмотр статуса, логов ошибок, источника и лицензии.",
        "/mode show|chat|code\n"
        "- Что делает: переключает режим текущего диалога.\n"
        "- code: инженерный режим с планом, изменениями по файлам, тестами и командами проверки; сильнее опирается на проектный и локальный контекст.",
        "/kb ...\n"
        "- Что делает: управляет локальной базой знаний.\n"
        "- add <путь>: индексирует файл или папку.\n"
        "- on/off: включает или выключает БЗ для текущего диалога.\n"
        "- search <запрос>: ищет по БЗ без вызова модели.",
        "/project ...\n"
        "- Что делает: управляет проектными индексами.\n"
        "- add <путь>: ставит проект в очередь на скан.\n"
        "- use <id>: привязывает проект к текущему диалогу.\n"
        "- search <запрос>: ищет по активному проекту или по всем индексам.",
        "/model ... и /models\n"
        "- Что делают: показывают локальные `.gguf`, переключают текущую модель и умеют удалять неактивные модели.",
        "/tasks ...\n"
        "- Что делает: показывает и чистит очередь фоновых задач.",
        "/deepplan <запрос>\n"
        "- Что делает: сначала прогоняет короткий анализ задачи, прикидывает адекватную длительность и только потом предлагает запуск.\n"
        "- Что показывает: минимум, рекомендуемое время, верхнюю границу пользы, тип задачи, сложность и что именно надо закрыть.\n"
        "- Когда полезно: когда задача жирная и ты не хочешь угадывать срок вручную.\n"
        "- Как запускать после оценки: в Telegram кнопкой, в terminal ответом 'да'.",
        "/deepthink <длительность> <запрос>\n"
        "- Что делает: запускает длинную фоновую задачу по сроку, который ты указал сам. Бот строит шаблон ответа, заполняет его проходами, считает прогресс и нагрузку на железо, потом доводит всё до финала и складывает итог в JSON.\n"
        "- В terminal-режиме long-think старается пережить выход из SSH и продолжить работу без активного подключения.\n"
        "- Ограничение символов обычного чата на long-think не распространяется.\n"
        "- Где результат: deep_think_jobs/<дата>_<id>_<slug>/result.json.\n"
        "- Где смотреть прогресс: /deepstatus.\n"
        "- Как остановить: /deepcancel [job_id].\n"
        f"- {build_duration_syntax_text()}",
        "/deepstatus\n"
        "- Что делает: показывает список последних long-think job'ов для текущего чата или terminal-сессии.\n"
        "- В terminal-режиме открывает отдельный экран со списком job'ов, подробностями и принудительной остановкой.\n"
        "- Что видно: статус, фазу, сколько прошло, сколько осталось, когда стартует финальная полировка, путь к папке job'а.",
        "/deepcancel [job_id]\n"
        "- Что делает: отменяет активный long-think job.\n"
        "- Если job_id не указать: бот попробует отменить текущий активный job.\n"
        "- Артефакты при отмене не удаляются, JSON остаётся на диске.",
        "/ineedmore\n"
        "- Что делает: собирает до трёх независимых запросов в одну пачку и затем формирует один общий ответ.\n"
        "- Важно: память обычного диалога внутри этого режима не используется, так что связи между пунктами лучше прописывать прямо текстом.",
        "/reset\n"
        "- Что делает: полностью очищает память текущего диалога.\n"
        "- Когда полезно: если модель начала тащить мусор из контекста или ты резко меняешь тему.",
        "/status\n"
        "- Что делает: показывает состояние бота, модели и llama-server.",
        "/errors\n"
        "- Что делает: показывает свежий хвост runtime.log, systemd_supervisor.log, llama_server.log и последние проблемные long-think job'ы.\n"
        "- Когда полезно: если /deepthink или обычный запрос упали и нужно быстро понять, где бот наломал дров.",
        "/source и /license\n"
        "- Что делают: показывают ссылку на исходный код и информацию по лицензии GNU AGPLv3.",
    ]

    if mode == "terminal":
        sections.extend(
            [
                "/limit show | /limit <число> | /limit off | /limit ask\n"
                "- show: показать сохранённый лимит символов.\n"
                "- <число>: закрепить лимит ответов для текущей terminal-сессии.\n"
                "- off: отключить сохранённый лимит.\n"
                "- ask: снова спрашивать лимит перед ответом.",
                "/deepplan <запрос>\n"
                "- Что делает: оценивает большую задачу и предлагает срок для deepthink до запуска.",
                "/session ...\n"
                "- Что делает: показывает terminal-сессию, умеет переименовывать и экспортировать её.",
                "/repeat | /clipboard ... | /paste\n"
                "- /repeat: повторно отправляет последний обычный запрос.\n"
                "- /clipboard user|bot|show|clear|set <текст>: внутренний буфер terminal-сессии.\n"
                "- /paste: подставляет буфер в следующую строку ввода, чтобы можно было быстро править и отправить заново.",
                "/exit\n"
                "- Что делает: завершает terminal-режим.\n"
                "- Что важно: сессия не пропадает, к ней можно вернуться из меню 'Сессии' в лаунчере.",
            ]
        )
    else:
        sections.append(
            "/start\n"
            "- Что делает: показывает стартовое сообщение и быстрый список базовых команд."
        )

    return "\n\n".join(sections)


def read_text_tail(path: Path, max_chars: int = ERROR_LOG_PREVIEW_CHARS) -> str:
    if not path.is_file():
        return "Лог пока не найден."
    return path.read_text(encoding="utf-8", errors="ignore")[-max_chars:].strip() or "Лог пуст."


def list_recent_problem_long_think_jobs(limit: int = ERROR_LOG_STATUS_LIMIT) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    if not LONG_THINK_ROOT.is_dir():
        return snapshots

    for result_path in LONG_THINK_ROOT.glob("*/result.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(payload.get("status") or "")
        if status not in {"failed", "cancelled", "interrupted"}:
            continue
        snapshots.append(payload)

    snapshots.sort(key=lambda item: str(item.get("updated_at") or item.get("completed_at") or ""), reverse=True)
    return snapshots[:limit]


def build_error_logs_text() -> str:
    lines = [
        "Логи ошибок и runtime:",
        f"runtime.log: {RUNTIME_LOG_PATH}",
        read_text_tail(RUNTIME_LOG_PATH),
        "",
        f"systemd_supervisor.log: {SUPERVISOR_LOG_PATH}",
        read_text_tail(SUPERVISOR_LOG_PATH),
        "",
        f"llama_server.log: {LOG_DIR / 'llama_server.log'}",
        read_text_tail(LOG_DIR / "llama_server.log"),
    ]

    failed_jobs = list_recent_problem_long_think_jobs()
    if failed_jobs:
        lines.extend(["", "Последние проблемные long-think job'ы:"])
        for job in failed_jobs:
            lines.extend(
                [
                    f"[{str(job.get('job_id') or '')[:8]}] статус: {job.get('status')}",
                    f"Фаза: {job.get('phase') or '-'}",
                    f"Ошибка: {job.get('error') or 'без текста'}",
                    f"JSON: {job.get('result_path') or '-'}",
                    "",
                ]
            )

    return "\n".join(lines).strip()


def list_long_think_jobs_for_owner(owner_key: str) -> list[dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}

    if LONG_THINK_ROOT.is_dir():
        for state_path in LONG_THINK_ROOT.glob("*/state.json"):
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("owner_key") != owner_key:
                continue
            snapshots[str(payload.get("job_id") or state_path.parent.name)] = payload

    for job_id, job in long_think_jobs.items():
        if job.get("owner_key") != owner_key:
            continue
        snapshots[job_id] = serialize_long_think_job(job)

    ordered = sorted(
        snapshots.values(),
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )
    return ordered[:LONG_THINK_STATUS_LIMIT]


def render_long_think_jobs_status(owner_key: str) -> str:
    jobs = list_long_think_jobs_for_owner(owner_key)
    if not jobs:
        return "Long-think job'ов пока нет."

    lines = ["Long-think job'ы:"]
    for job in jobs:
        elapsed = format_clock_duration(job.get("elapsed_seconds"))
        remaining = format_clock_duration(job.get("remaining_seconds"))
        until_final = format_clock_duration(job.get("until_finalization_seconds"))
        lines.extend(
            [
                f"[{str(job.get('job_id') or '')[:8]}] {job.get('status')} | фаза: {job.get('phase') or '-'}",
                f"Прошло: {elapsed} | Осталось: {remaining} | До финала: {until_final} | Готово: {job.get('progress_percent', 0)}%",
                f"Итерации: {job.get('completed_iterations', 0)}/{job.get('planned_iterations', 0)} | Финальное окно: {format_clock_duration(job.get('final_buffer_seconds'))}",
                f"Папка: {job.get('artifact_dir') or '-'}",
            ]
        )
        if job.get("note"):
            lines.append(f"Заметка: {job['note']}")
        if job.get("error"):
            lines.append(f"Ошибка: {job['error']}")
        lines.append("")
    return "\n".join(lines).strip()


def clear_terminal_screen() -> None:
    clear_console_live_line_permanently()
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return
    if not os.getenv("TERM"):
        return
    os.system("clear")


def get_long_think_job_snapshot_for_owner(
    owner_key: str,
    raw_job_id: str,
) -> dict[str, Any] | None:
    normalized = str(raw_job_id or "").strip().lower()
    if not normalized:
        return None
    for payload in list_long_think_jobs_for_owner(owner_key):
        if str(payload.get("job_id") or "").lower().startswith(normalized):
            return payload
    return None


def build_terminal_long_think_jobs_browser_text(jobs: list[dict[str, Any]]) -> str:
    lines = [
        "Long-think процессы:",
        "Выбери номер job, чтобы посмотреть подробности или принудительно остановить его.",
        "",
    ]
    for index, job in enumerate(jobs, start=1):
        request_summary = summarize_long_think_iteration(
            str(job.get("request_text") or ""),
            max_chars=90,
        ) or "без описания"
        lines.extend(
            [
                f"{index}. [{str(job.get('job_id') or '')[:8]}] {job.get('status')} | готово: {job.get('progress_percent', 0)}%",
                (
                    f"   Фаза: {job.get('phase') or '-'} | "
                    f"Осталось: {format_clock_duration(job.get('remaining_seconds'))} | "
                    f"Запрос: {request_summary}"
                ),
            ]
        )
    lines.extend(["", "0. Назад"])
    return "\n".join(lines)


def build_terminal_long_think_job_actions_text(job: dict[str, Any]) -> str:
    request_summary = summarize_long_think_iteration(
        str(job.get("request_text") or ""),
        max_chars=140,
    ) or "без описания"
    lines = [
        f"Long-think job [{str(job.get('job_id') or '')[:8]}]",
        f"Статус: {job.get('status')} | Фаза: {job.get('phase') or '-'} | Готово: {job.get('progress_percent', 0)}%",
        f"Запрос: {request_summary}",
        f"Папка: {job.get('artifact_dir') or '-'}",
        "",
        "1. Посмотреть подробности",
        "2. Завершить процесс принудительно",
        "0. Назад",
    ]
    return "\n".join(lines)


def build_terminal_long_think_job_detail_text(job: dict[str, Any]) -> str:
    status = str(job.get("status") or "")
    detail_lines: list[str] = []
    if is_long_think_active_status(status):
        detail_lines.append(build_terminal_running_long_think_resume_text(job))
    elif status == "completed":
        detail_lines.append(build_terminal_completed_long_think_resume_text(job))
    else:
        detail_lines.extend(
            [
                "Long-think job сейчас не в running/completed состоянии.",
                f"Статус: {status or '-'}",
                f"Фаза: {job.get('phase') or '-'}",
                f"Ошибка: {job.get('error') or 'без текста'}",
                f"Папка: {job.get('artifact_dir') or '-'}",
                f"JSON: {job.get('result_path') or '-'}",
            ]
        )

    warnings: list[str] = []
    if is_long_think_active_status(status):
        warnings.append(
            "Этот job ещё живой. Ручной stop/restart бота, смена модели или kill процесса могут его оборвать."
        )
        if job.get("systemd_unit"):
            warnings.append(
                f"Detached worker висит через systemd unit {job['systemd_unit']}."
            )
        elif job.get("process_id"):
            warnings.append(
                f"Detached worker идёт отдельным pid={job['process_id']}."
            )
    if status == "completed" and not job.get("answer_completed_fully"):
        warnings.append(
            "Ответ был сохранён как лучшая доступная версия, но не закрылся идеально до конца."
        )
    if job.get("note"):
        warnings.append(f"Заметка runtime: {job['note']}")

    if warnings:
        detail_lines.extend(["", "Предупреждения:"])
        detail_lines.extend(f"- {warning}" for warning in warnings)

    detail_lines.extend(
        [
            "",
            f"ID: {str(job.get('job_id') or '')[:8]}",
            f"Прошло: {format_clock_duration(job.get('elapsed_seconds'))}",
            f"Осталось: {format_clock_duration(job.get('remaining_seconds'))}",
            f"До финала: {format_clock_duration(job.get('until_finalization_seconds'))}",
        ]
    )
    return "\n".join(detail_lines).strip()


async def ask_terminal_yes_no(prompt: str, *, default: bool = False) -> bool:
    suffix = "[д/н]" if default else "[д/н]"
    while True:
        raw_value = (
            await async_terminal_input(f"{prompt} {suffix}: ")
        ).strip().lower()
        if not raw_value:
            return default
        if raw_value in {"д", "да", "y", "yes"}:
            return True
        if raw_value in {"н", "нет", "n", "no"}:
            return False
        print("Ответь 'да' или 'нет'.", flush=True)


async def show_terminal_long_think_job_details(job: dict[str, Any]) -> None:
    clear_terminal_screen()
    print(build_terminal_long_think_job_detail_text(job) + "\n", flush=True)
    await async_terminal_input("Нажми Enter, чтобы вернуться: ")


async def force_stop_terminal_long_think_job(job: dict[str, Any]) -> None:
    clear_terminal_screen()
    print(build_terminal_long_think_job_actions_text(job) + "\n", flush=True)
    status = str(job.get("status") or "")
    if not is_long_think_active_status(status):
        print("Этот job уже не активен. Принудительно останавливать там особо нечего.\n", flush=True)
        await async_terminal_input("Нажми Enter, чтобы вернуться: ")
        return

    confirmed = await ask_terminal_yes_no(
        f"Точно принудительно останавливаю job {str(job.get('job_id') or '')[:8]}?",
        default=False,
    )
    if not confirmed:
        print("Ок, пока не трогаю процесс.\n", flush=True)
        await async_terminal_input("Нажми Enter, чтобы вернуться: ")
        return

    runtime_job = resolve_long_think_runtime_job(job)
    runtime_job["cancel_requested"] = True
    runtime_job["error"] = "Long-think job был принудительно остановлен из deepstatus."
    request_long_think_cancel(runtime_job)
    print(
        f"Принудительно останавливаю job {str(job.get('job_id') or '')[:8]}.\n"
        f"Артефакты останутся в {job.get('artifact_dir') or '-'}.\n",
        flush=True,
    )
    await async_terminal_input("Нажми Enter, чтобы вернуться: ")


async def open_terminal_deepstatus_menu(owner_key: str) -> None:
    while True:
        jobs = list_long_think_jobs_for_owner(owner_key)
        clear_terminal_screen()
        if not jobs:
            print("Long-think job'ов пока нет.\n", flush=True)
            await async_terminal_input("Нажми Enter, чтобы вернуться: ")
            return

        print(build_terminal_long_think_jobs_browser_text(jobs) + "\n", flush=True)
        raw_value = (await async_terminal_input("Выбери job по номеру: ")).strip()
        if not raw_value or raw_value == "0":
            return
        if not raw_value.isdigit():
            print("Нужен номер из списка, а не астральная проекция.\n", flush=True)
            await async_terminal_input("Нажми Enter, чтобы продолжить: ")
            continue

        job_index = int(raw_value)
        if job_index < 1 or job_index > len(jobs):
            print("Такого номера нет. Смотри в список, а не в космос.\n", flush=True)
            await async_terminal_input("Нажми Enter, чтобы продолжить: ")
            continue

        selected_job_id = str(jobs[job_index - 1].get("job_id") or "")
        while True:
            job_snapshot = get_long_think_job_snapshot_for_owner(owner_key, selected_job_id)
            if job_snapshot is None:
                print("Этот job уже куда-то делся. Возвращаю в список.\n", flush=True)
                await async_terminal_input("Нажми Enter, чтобы продолжить: ")
                break

            clear_terminal_screen()
            print(build_terminal_long_think_job_actions_text(job_snapshot) + "\n", flush=True)
            action_raw = (await async_terminal_input("Выбор: ")).strip()
            if not action_raw or action_raw == "0":
                break
            if action_raw == "1":
                await show_terminal_long_think_job_details(job_snapshot)
                continue
            if action_raw == "2":
                await force_stop_terminal_long_think_job(job_snapshot)
                continue
            print("Не понял выбор. Жми 1, 2 или 0.\n", flush=True)
            await async_terminal_input("Нажми Enter, чтобы продолжить: ")


def build_terminal_completed_long_think_resume_text(job: dict[str, Any]) -> str:
    return (
        "С возвращением! Модель доделала ваш запрос! Ура! Вот небольшая сводка:\n"
        f"Средняя нагрузка на CPU: {format_metric_percent(job.get('average_cpu_percent'))}\n"
        f"Средняя нагрузка на RAM: {format_metric_percent(job.get('average_ram_percent'))}\n"
        f"Средняя нагрузка на GPU: {format_metric_percent(job.get('average_gpu_percent'))}\n"
        f"Закрыла ли модель ответ полностью?: {'Да' if job.get('answer_completed_fully') else 'Нет'}\n"
        f"Количество символов: {len(job.get('final_answer') or '')}\n"
        f"Ответ лежит в: {job.get('result_path') or '-'}\n"
        "Спасибо, что доверяете нашему порту. "
        "С уважением, @zzentasarc & @Default_Netion ;3"
    )


def build_terminal_running_long_think_resume_text(job: dict[str, Any]) -> str:
    refresh_long_think_progress_snapshot(job)
    return (
        "Хеееей! С возвращением! :)\n"
        "Модель пока работает над вашим запросом! "
        "Спасибо, что доверяете нашему порту. "
        "С уважением, @zzentasarc & @Default_Netion ;3\n"
        f"{job.get('progress_banner') or build_long_think_progress_banner(job)}"
    )


def build_terminal_long_think_resume_text(
    owner_key: str,
    previous_opened_at: str | None,
) -> str:
    jobs = list_long_think_jobs_for_owner(owner_key)
    if not jobs:
        return ""

    active_job = next(
        (
            job
            for job in jobs
            if is_long_think_active_status(str(job.get("status") or ""))
        ),
        None,
    )
    if active_job is not None:
        return build_terminal_running_long_think_resume_text(active_job)

    previous_dt = parse_iso_datetime(previous_opened_at)
    if previous_dt is None:
        return ""

    latest_job = max(
        jobs,
        key=lambda item: str(item.get("completed_at") or item.get("updated_at") or item.get("created_at") or ""),
    )
    completed_dt = parse_iso_datetime(
        str(latest_job.get("completed_at") or latest_job.get("updated_at") or "")
    )
    if completed_dt is None or completed_dt <= previous_dt:
        return ""
    if str(latest_job.get("status") or "") != "completed":
        return ""
    return build_terminal_completed_long_think_resume_text(latest_job)


def build_local_long_think_template(job: dict[str, Any]) -> str:
    request_excerpt = summarize_long_think_iteration(job.get("request_text", ""), max_chars=180)
    return (
        "1. Понять задачу и определить итоговый формат ответа.\n"
        "2. Собрать основной материал по теме без воды и служебки.\n"
        "3. Привести результат к цельной структуре.\n"
        "4. Закончить сильным финальным выводом без обрыва.\n"
        f"Опорная тема: {request_excerpt}"
    )


def build_long_think_template_messages(job: dict[str, Any]) -> list[dict[str, str]]:
    user_lines = [
        f"Большая задача пользователя:\n{job['request_text']}",
        "Сначала собери шаблон будущего ответа.",
        "Шаблон должен помочь потом написать полный итог без обрыва и без каши.",
        "Если задача про код, шаблон должен перечислить файлы, блоки логики и порядок реализации.",
        "Если задача про текст, шаблон должен перечислить разделы, ключевые тезисы и финальный вывод.",
        "Верни только шаблон ответа на русском языке без пояснений о процессе.",
    ]
    return [
        {"role": "system", "content": LONG_THINK_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_lines)},
    ]


def build_long_think_iteration_messages(
    job: dict[str, Any],
    iteration_index: int,
    total_iterations: int,
) -> list[dict[str, str]]:
    latest_draft = trim_long_think_context(job.get("latest_draft", ""))
    template_outline = trim_long_think_context(job.get("template_outline", ""))
    remaining_seconds = long_think_job_remaining_seconds(job) or 0
    until_finalization_seconds = long_think_time_until_finalization_seconds(job) or 0
    recent_iterations = job.get("iterations", [])[-LONG_THINK_HISTORY_ITERATIONS:]
    recent_notes = [
        f"{item['index']}. {item['summary']}"
        for item in recent_iterations
        if item.get("summary")
    ]

    instruction = (
        "Собери сильный первый рабочий вариант результата."
        if iteration_index == 1
        else "Улучши текущую рабочую версию результата, закрой слабые места и усили полноту."
    )
    if iteration_index == total_iterations:
        instruction = (
            "Это последний черновой проход перед финальной полировкой. "
            "Собери максимально сильную рабочую версию результата."
        )

    user_lines = [
        f"Большая задача пользователя:\n{job['request_text']}",
        f"Проход: {iteration_index} из {total_iterations}",
        f"Полная длительность режима: {format_clock_duration(job['duration_seconds'])}",
        f"Рабочая фаза до финала: {format_clock_duration(job.get('work_phase_seconds'))}",
        f"Отдельное окно на финальную полировку: {format_clock_duration(job.get('final_buffer_seconds'))}",
        f"Осталось до дедлайна: {format_clock_duration(remaining_seconds)}",
        f"До старта финальной полировки: {format_clock_duration(until_finalization_seconds)}",
        instruction,
        "Сейчас не нужно пытаться выдать окончательную идеальную версию любой ценой. "
        "Нужно улучшить рабочий черновик так, чтобы к старту финального окна он уже был сильным и устойчивым.",
    ]
    if recent_notes:
        user_lines.extend(
            [
                "Короткие заметки по прошлым проходам:",
                *recent_notes,
            ]
        )
    if template_outline:
        user_lines.extend(
            [
                "Шаблон ответа, которого нужно придерживаться:",
                template_outline,
            ]
        )
    if latest_draft:
        user_lines.extend(
            [
                "Текущая рабочая версия результата:",
                latest_draft,
            ]
        )
    user_lines.append(
        "Верни только обновлённую рабочую версию результата без описания процесса, без выводов про свой план и без служебных пояснений."
    )

    return [
        {"role": "system", "content": LONG_THINK_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_lines)},
    ]


def build_long_think_final_messages(job: dict[str, Any]) -> list[dict[str, str]]:
    latest_draft = trim_long_think_context(job.get("latest_draft", ""))
    template_outline = trim_long_think_context(job.get("template_outline", ""))
    remaining_seconds = long_think_job_remaining_seconds(job) or 0
    user_lines = [
        f"Большая задача пользователя:\n{job['request_text']}",
        "Финальная полировка результата.",
        f"Это финальное окно длительностью: {format_clock_duration(job.get('final_buffer_seconds'))}",
        f"До жёсткого дедлайна осталось: {format_clock_duration(remaining_seconds)}",
        "Сейчас не открывай новые большие ветки решения. "
        "Сконцентрируйся на завершённости, цельности, чистоте структуры и нормальном финальном выводе.",
    ]
    if template_outline:
        user_lines.extend(
            [
                "Шаблон ответа, который надо закрыть полностью:",
                template_outline,
            ]
        )
    if latest_draft:
        user_lines.extend(
            [
                "Текущая лучшая версия результата:",
                latest_draft,
            ]
        )
    user_lines.append(
        "Верни только окончательный результат без комментариев о процессе, проверках и скрытых шагах. "
        "Ответ должен быть завершённым, а не оборванным."
    )
    return [
        {"role": "system", "content": LONG_THINK_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_lines)},
    ]


async def update_long_think_progress_message(
    job: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    refresh_long_think_progress_snapshot(job)
    if job["mode"] != "telegram":
        return

    text = str(job.get("progress_banner") or "").strip()
    if not text:
        return
    if not force and text == str(job.get("progress_message_text") or "").strip():
        return

    bot = job.get("bot")
    chat_id = job.get("chat_id")
    if bot is None or chat_id is None:
        return

    current_message = job.get("progress_message")
    try:
        if isinstance(current_message, Message):
            updated_message = await safe_edit_message(current_message, text)
        else:
            updated_message = await bot.send_message(chat_id=chat_id, text=text)
        job["progress_message"] = updated_message
        job["progress_message_text"] = text
    except Exception:
        logger.exception("Не удалось обновить progress-сообщение long-think job=%s", job["job_id"])


async def run_long_think_progress_notifier(job_id: str) -> None:
    try:
        while True:
            await asyncio.sleep(LONG_THINK_PROGRESS_UPDATE_SECONDS)
            job = long_think_jobs.get(job_id)
            if job is None:
                return
            refresh_long_think_progress_snapshot(job)
            persist_long_think_job(job)
            await update_long_think_progress_message(job)
            if not is_long_think_active_status(str(job.get("status") or "")):
                return
    except asyncio.CancelledError:
        return


async def run_long_think_metrics_monitor(job_id: str) -> None:
    previous_cpu_snapshot = read_proc_cpu_snapshot()
    try:
        while True:
            await asyncio.sleep(LONG_THINK_METRICS_SAMPLE_SECONDS)
            job = long_think_jobs.get(job_id)
            if job is None:
                return
            current_cpu_snapshot = read_proc_cpu_snapshot()
            add_long_think_metrics_sample(
                job,
                cpu_percent=compute_cpu_percent(previous_cpu_snapshot, current_cpu_snapshot),
                ram_percent=read_ram_percent(),
                gpu_percent=read_gpu_percent(),
            )
            previous_cpu_snapshot = current_cpu_snapshot or previous_cpu_snapshot
            persist_long_think_job(job)
            if not is_long_think_active_status(str(job.get("status") or "")):
                return
    except asyncio.CancelledError:
        return


async def notify_long_think_job(job: dict[str, Any], text: str) -> None:
    if job["mode"] == "telegram":
        bot = job.get("bot")
        chat_id = job.get("chat_id")
        if bot is None or chat_id is None:
            return
        try:
            await bot.send_message(chat_id=chat_id, text=text)
        except Exception:
            logger.exception("Не удалось отправить уведомление long-think в Telegram.")
        return

    write_console_output(f"\n[LONG-THINK {job['job_id'][:8]}]\n{text}\n", end="", flush=True)


def create_long_think_job(
    *,
    mode: str,
    owner_key: str,
    request_text: str,
    duration_seconds: int,
    chat_id: int | None = None,
    user_id: int | None = None,
    bot: Bot | None = None,
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    artifact_dir = build_long_think_artifact_dir(job_id, request_text)
    final_buffer_seconds = get_long_think_final_buffer_seconds(duration_seconds)
    work_phase_seconds = max(60, duration_seconds - final_buffer_seconds)
    job = {
        "job_id": job_id,
        "mode": mode,
        "owner_key": owner_key,
        "status": "queued",
        "phase": "queued",
        "request_text": request_text.strip(),
        "duration_seconds": duration_seconds,
        "planned_iterations": choose_long_think_iteration_count(work_phase_seconds),
        "created_at": iso_now(),
        "started_at": None,
        "deadline_at": None,
        "finalization_starts_at": None,
        "completed_at": None,
        "updated_at": iso_now(),
        "cancel_requested": False,
        "artifact_dir": str(artifact_dir),
        "result_path": "",
        "error": "",
        "note": "",
        "chat_id": chat_id,
        "user_id": user_id,
        "process_id": os.getpid(),
        "systemd_unit": "",
        "systemd_scope": "",
        "bot": bot,
        "task": None,
        "final_buffer_seconds": final_buffer_seconds,
        "work_phase_seconds": work_phase_seconds,
        "template_outline": "",
        "template_ready_at": None,
        "iterations": [],
        "latest_draft": "",
        "final_answer": "",
        "answer_completed_fully": False,
        "final_finish_reason": "",
        "progress_percent": 0,
        "progress_banner": "",
        "progress_message_text": "",
        "progress_message": None,
        "progress_task": None,
        "metrics": build_long_think_metrics_payload(),
        "average_cpu_percent": None,
        "average_ram_percent": None,
        "average_gpu_percent": None,
        "metrics_task": None,
    }
    refresh_long_think_progress_snapshot(job)
    long_think_jobs[job_id] = job
    touch_long_think_job(job_id)
    persist_long_think_job(job)
    return job


def launch_detached_long_think_worker(job: dict[str, Any]) -> None:
    script_path = Path(__file__).resolve()
    env = os.environ.copy()
    env["HEYMATE_DETACHED_LONG_THINK"] = "1"
    env["SHOW_MODEL_RAW"] = "0"
    systemd_context = get_systemd_run_context()
    if systemd_context is not None:
        command_prefix, scope = systemd_context
        unit_name = f"heymate-long-think-{job['job_id'][:8]}"
        command = [
            *command_prefix,
            f"--unit={unit_name}",
            "--collect",
            "--quiet",
            f"--property=WorkingDirectory={PROJECT_ROOT}",
            "--property=KillMode=control-group",
            "--property=StandardOutput=null",
            "--property=StandardError=null",
            "--setenv=PYTHONUNBUFFERED=1",
            "--setenv=SHOW_MODEL_RAW=0",
            "--setenv=HEYMATE_DETACHED_LONG_THINK=1",
            sys.executable,
            str(script_path),
            "--long-think-worker",
            job["job_id"],
        ]
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            job["process_id"] = 0
            job["systemd_unit"] = unit_name
            job["systemd_scope"] = scope
            persist_long_think_job(job)
            long_think_jobs.pop(job["job_id"], None)
            long_think_job_order.pop(job["job_id"], None)
            return

        logger.warning(
            "Не удалось отцепить long-think через systemd-run, откатываюсь на setsid: job_id=%s rc=%s stderr=%s",
            job["job_id"],
            completed.returncode,
            (completed.stderr or completed.stdout or "").strip()[:500],
        )

    child = subprocess.Popen(
        [sys.executable, str(script_path), "--long-think-worker", job["job_id"]],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    job["process_id"] = child.pid
    job["systemd_unit"] = ""
    job["systemd_scope"] = ""
    persist_long_think_job(job)
    long_think_jobs.pop(job["job_id"], None)
    long_think_job_order.pop(job["job_id"], None)


def start_long_think_job(
    *,
    mode: str,
    owner_key: str,
    request_text: str,
    duration_seconds: int,
    chat_id: int | None = None,
    user_id: int | None = None,
    bot: Bot | None = None,
) -> dict[str, Any]:
    job = create_long_think_job(
        mode=mode,
        owner_key=owner_key,
        request_text=request_text,
        duration_seconds=duration_seconds,
        chat_id=chat_id,
        user_id=user_id,
        bot=bot,
    )
    if mode == "terminal":
        launch_detached_long_think_worker(job)
    else:
        job["task"] = asyncio.create_task(run_long_think_job(job["job_id"]))
    return job


def start_long_think_from_plan(
    plan: dict[str, Any],
    *,
    bot: Bot | None = None,
) -> dict[str, Any]:
    return start_long_think_job(
        mode=str(plan.get("mode") or "terminal"),
        owner_key=str(plan.get("owner_key") or ""),
        request_text=str(plan.get("request_text") or ""),
        duration_seconds=int(plan.get("recommended_duration_seconds") or LONG_THINK_MIN_DURATION_SECONDS),
        chat_id=int(plan["chat_id"]) if plan.get("chat_id") is not None else None,
        user_id=int(plan["user_id"]) if plan.get("user_id") is not None else None,
        bot=bot,
    )


def is_process_alive(process_id: int | None) -> bool:
    pid = int(process_id or 0)
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def request_long_think_cancel(job: dict[str, Any]) -> None:
    job["cancel_requested"] = True
    if not job.get("error"):
        job["error"] = "Long-think job был отменён пользователем."
    persist_long_think_job(job)

    task = job.get("task")
    if isinstance(task, asyncio.Task) and not task.done():
        task.cancel()
        return

    unit_name = str(job.get("systemd_unit") or "").strip()
    unit_scope = str(job.get("systemd_scope") or "").strip()
    if unit_name:
        stop_systemd_unit(unit_name, unit_scope)

    process_id = int(job.get("process_id") or 0)
    if process_id > 0 and process_id != os.getpid() and is_process_alive(process_id):
        try:
            os.kill(process_id, signal.SIGTERM)
        except OSError:
            return


def has_active_detached_terminal_long_think_worker() -> bool:
    if not LONG_THINK_ROOT.is_dir():
        return False

    for state_path in LONG_THINK_ROOT.glob("*/state.json"):
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("mode") or "") != "terminal":
            continue
        if not is_long_think_active_status(str(payload.get("status") or "")):
            continue
        unit_name = str(payload.get("systemd_unit") or "").strip()
        unit_scope = str(payload.get("systemd_scope") or "").strip()
        if unit_name and is_systemd_unit_active(unit_name, unit_scope):
            return True
        process_id = int(payload.get("process_id") or 0)
        if process_id > 0 and process_id != os.getpid() and is_process_alive(process_id):
            return True
    return False


async def collect_long_think_reply_with_timeout(
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    label: str,
    timeout_seconds: int,
) -> tuple[str, str, str | None]:
    timeout_seconds = max(1, int(timeout_seconds))
    return await asyncio.wait_for(
        collect_model_reply(
            messages,
            max_tokens,
            brief_mode=False,
            label=label,
            max_reply_chars=(
                LONG_THINK_MAX_MODEL_REPLY_CHARS
                if LONG_THINK_MAX_MODEL_REPLY_CHARS > 0
                else None
            ),
        ),
        timeout=timeout_seconds,
    )


async def run_long_think_job(job_id: str) -> None:
    job = long_think_jobs.get(job_id)
    if job is None:
        return

    progress_task: asyncio.Task[Any] | None = None
    metrics_task: asyncio.Task[Any] | None = None

    try:
        start_dt = datetime.now().astimezone()
        total_iterations = int(job["planned_iterations"])
        safety_seconds = get_long_think_iteration_safety_seconds(job["duration_seconds"])
        hard_deadline_dt = start_dt + timedelta(seconds=job["duration_seconds"])
        finalization_start_dt = hard_deadline_dt - timedelta(
            seconds=int(job.get("final_buffer_seconds") or 0)
        )
        latest_iteration_start_dt = max(
            start_dt,
            finalization_start_dt - timedelta(seconds=safety_seconds),
        )
        job["status"] = "running"
        job["phase"] = "planning"
        job["started_at"] = start_dt.isoformat()
        job["deadline_at"] = hard_deadline_dt.isoformat()
        job["finalization_starts_at"] = finalization_start_dt.isoformat()
        job["note"] = ""
        job["error"] = ""
        job["answer_completed_fully"] = False
        job["final_finish_reason"] = ""
        refresh_long_think_progress_snapshot(job)
        persist_long_think_job(job)

        progress_task = asyncio.create_task(run_long_think_progress_notifier(job_id))
        metrics_task = asyncio.create_task(run_long_think_metrics_monitor(job_id))
        job["progress_task"] = progress_task
        job["metrics_task"] = metrics_task

        add_long_think_metrics_sample(
            job,
            cpu_percent=None,
            ram_percent=read_ram_percent(),
            gpu_percent=read_gpu_percent(),
        )
        persist_long_think_job(job)

        time_until_finalization = max(
            0,
            int((finalization_start_dt - datetime.now().astimezone()).total_seconds()),
        )
        template_timeout_seconds = max(
            3,
            min(
                max(15, int(job["duration_seconds"]) // 5),
                max(1, time_until_finalization - safety_seconds),
            ),
        )
        job["template_outline"] = build_local_long_think_template(job)
        if time_until_finalization > max(2, safety_seconds + 1):
            try:
                template_reply, template_raw_reply, _ = await collect_long_think_reply_with_timeout(
                    build_long_think_template_messages(job),
                    choose_long_think_token_budget(
                        max(1, time_until_finalization),
                        LONG_THINK_TEMPLATE_MAX_TOKENS,
                    ),
                    label=f"deepthink-template:{job_id[:8]}",
                    timeout_seconds=template_timeout_seconds,
                )
                template_outline = (
                    sanitize_assistant_reply_text(template_reply, template_raw_reply)
                    or template_reply.strip()
                    or template_raw_reply.strip()
                )
                if template_outline:
                    job["template_outline"] = template_outline
            except asyncio.TimeoutError:
                logger.warning(
                    "Построение шаблона long-think упёрлось во время: job_id=%s timeout=%s",
                    job_id,
                    template_timeout_seconds,
                )
                job["note"] = (
                    "Шаблон ответа пришлось собрать в локальном fallback-режиме, "
                    "потому что модель слишком долго его строила."
                )
            except Exception as exc:
                logger.warning(
                    "Не удалось построить шаблон long-think через модель: job_id=%s error=%s",
                    job_id,
                    exc,
                )
                job["note"] = (
                    "Шаблон ответа собрал локально, потому что отдельный проход планирования не взлетел."
                )

        job["template_ready_at"] = iso_now()
        job["phase"] = "template_ready"
        refresh_long_think_progress_snapshot(job)
        persist_long_think_job(job)
        await notify_long_think_job(
            job,
            "Модель построила шаблон для ответа! Мы будем держать вас в курсе!",
        )
        await update_long_think_progress_message(job, force=True)

        iteration_index = 0
        while True:
            if job.get("cancel_requested"):
                raise asyncio.CancelledError

            now_dt = datetime.now().astimezone()
            if now_dt >= latest_iteration_start_dt:
                logger.info(
                    "Останавливаю черновые проходы long-think перед финальным окном: job_id=%s iteration=%s",
                    job_id,
                    iteration_index,
                )
                break

            iteration_index += 1
            if iteration_index > int(job.get("planned_iterations") or 0):
                job["planned_iterations"] = iteration_index

            job["phase"] = f"iteration_{iteration_index}"
            refresh_long_think_progress_snapshot(job)
            persist_long_think_job(job)

            display_total_iterations = max(
                iteration_index,
                int(job.get("planned_iterations") or 0),
                total_iterations,
            )
            messages = build_long_think_iteration_messages(
                job,
                iteration_index,
                display_total_iterations,
            )
            time_until_finalization = max(
                0,
                int((finalization_start_dt - now_dt).total_seconds()),
            )
            iteration_timeout_seconds = max(1, time_until_finalization - safety_seconds)
            try:
                full_reply, raw_reply, _ = await collect_long_think_reply_with_timeout(
                    messages,
                    choose_long_think_token_budget(
                        max(1, time_until_finalization),
                        LONG_THINK_STEP_MAX_TOKENS,
                    ),
                    label=f"deepthink:{job_id[:8]}:{iteration_index}",
                    timeout_seconds=iteration_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Черновой проход long-think упёрся в дедлайн: job_id=%s iteration=%s timeout=%s",
                    job_id,
                    iteration_index,
                    iteration_timeout_seconds,
                )
                job["note"] = (
                    "Один из черновых проходов упёрся во временное окно. "
                    "Перешёл к финализации с лучшей доступной версией."
                )
                persist_long_think_job(job)
                break
            sanitized_reply = sanitize_assistant_reply_text(full_reply, raw_reply)
            job["latest_draft"] = sanitized_reply or full_reply.strip()
            job["iterations"].append(
                {
                    "index": iteration_index,
                    "timestamp": iso_now(),
                    "summary": summarize_long_think_iteration(job["latest_draft"]),
                    "chars": len(job["latest_draft"]),
                }
            )
            refresh_long_think_progress_snapshot(job)
            persist_long_think_job(job)

        if job.get("cancel_requested"):
            raise asyncio.CancelledError

        job["phase"] = "finalizing"
        refresh_long_think_progress_snapshot(job)
        persist_long_think_job(job)
        await update_long_think_progress_message(job, force=True)
        final_window_seconds = max(
            0,
            int((hard_deadline_dt - datetime.now().astimezone()).total_seconds()),
        )
        if final_window_seconds <= 0:
            if not job.get("latest_draft", "").strip():
                raise RuntimeError("Long-think упёрся в дедлайн до получения финального ответа.")
            job["final_answer"] = job["latest_draft"].strip()
            job["note"] = (
                "Дедлайн закончился до финальной полировки. "
                "Сохранил лучшую рабочую версию без дополнительного прохода."
            )
            job["answer_completed_fully"] = False
            job["final_finish_reason"] = "deadline"
        else:
            final_messages = build_long_think_final_messages(job)
            try:
                final_reply, final_raw_reply, final_finish_reason = await collect_long_think_reply_with_timeout(
                    final_messages,
                    choose_long_think_token_budget(
                        final_window_seconds,
                        LONG_THINK_FINAL_MAX_TOKENS,
                    ),
                    label=f"deepthink-final:{job_id[:8]}",
                    timeout_seconds=final_window_seconds,
                )
                job["final_answer"] = sanitize_assistant_reply_text(final_reply, final_raw_reply)
                if not job["final_answer"]:
                    job["final_answer"] = final_reply.strip() or final_raw_reply.strip()
                job["final_finish_reason"] = str(final_finish_reason or "")
                job["answer_completed_fully"] = (
                    bool(job["final_answer"].strip())
                    and str(final_finish_reason or "") != "length"
                )
            except asyncio.TimeoutError:
                if not job.get("latest_draft", "").strip():
                    raise RuntimeError("Финальная полировка не успела завершиться до дедлайна.")
                logger.warning(
                    "Финальная полировка long-think упёрлась в дедлайн: job_id=%s timeout=%s",
                    job_id,
                    final_window_seconds,
                )
                job["final_answer"] = job["latest_draft"].strip()
                job["note"] = (
                    "Финальная полировка не успела завершиться до дедлайна. "
                    "Сохранил лучшую рабочую версию."
                )
                job["answer_completed_fully"] = False
                job["final_finish_reason"] = "timeout"

        if not job.get("final_answer", "").strip():
            if job.get("latest_draft", "").strip():
                job["final_answer"] = job["latest_draft"].strip()
                job["note"] = (
                    job.get("note")
                    or "Финальный ответ пустой, поэтому сохранил лучшую рабочую версию."
                )
                job["answer_completed_fully"] = False
                job["final_finish_reason"] = job.get("final_finish_reason") or "empty-final"
            else:
                raise RuntimeError("Long-think завершился без итогового ответа.")

        job["status"] = "completed"
        job["phase"] = "completed"
        job["completed_at"] = iso_now()
        refresh_long_think_metric_averages(job)
        refresh_long_think_progress_snapshot(job)
        job["progress_percent"] = 100
        job["progress_banner"] = build_long_think_progress_banner(job)
        await update_long_think_progress_message(job, force=True)
        persist_long_think_job(job, final=True)
        append_jsonl(
            {
                "timestamp": iso_now(),
                "event": "long_think_completed",
                "job_id": job["job_id"],
                "mode": job["mode"],
                "owner_key": job["owner_key"],
                "duration_seconds": job["duration_seconds"],
                "artifact_dir": job["artifact_dir"],
                "result_path": job["result_path"],
                "text": summarize_long_think_iteration(job["request_text"], max_chars=500),
            }
        )
        completion_note_line = (
            f"Заметка: {job['note']}\n"
            if job.get("note")
            else ""
        )
        await notify_long_think_job(
            job,
            (
                f"Long-think job завершён.\n"
                f"ID: {job['job_id'][:8]}\n"
                f"Прошло: {format_clock_duration(long_think_job_elapsed_seconds(job))}\n"
                f"Ответ закрыт полностью: {'Да' if job.get('answer_completed_fully') else 'Нет'}\n"
                f"Символов: {len(job.get('final_answer') or '')}\n"
                f"{completion_note_line}"
                f"JSON: {job['result_path']}"
            ),
        )
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        job["phase"] = "cancelled"
        job["completed_at"] = iso_now()
        if not job.get("error"):
            job["error"] = "Long-think job был отменён пользователем."
        refresh_long_think_metric_averages(job)
        refresh_long_think_progress_snapshot(job)
        persist_long_think_job(job, final=True)
        append_jsonl(
            {
                "timestamp": iso_now(),
                "event": "long_think_cancelled",
                "job_id": job["job_id"],
                "mode": job["mode"],
                "owner_key": job["owner_key"],
                "artifact_dir": job["artifact_dir"],
                "result_path": job["result_path"],
                "error": job["error"],
            }
        )
        await notify_long_think_job(
            job,
            (
                f"Long-think job остановлен.\n"
                f"ID: {job['job_id'][:8]}\n"
                f"JSON: {job['result_path']}"
            ),
        )
    except Exception as exc:
        logger.exception("Long-think job упал: job_id=%s", job_id)
        job["status"] = "failed"
        job["phase"] = "failed"
        job["error"] = str(exc)
        job["completed_at"] = iso_now()
        refresh_long_think_metric_averages(job)
        refresh_long_think_progress_snapshot(job)
        persist_long_think_job(job, final=True)
        append_jsonl(
            {
                "timestamp": iso_now(),
                "event": "long_think_failed",
                "job_id": job["job_id"],
                "mode": job["mode"],
                "owner_key": job["owner_key"],
                "artifact_dir": job["artifact_dir"],
                "result_path": job["result_path"],
                "error": str(exc),
            }
        )
        await notify_long_think_job(
            job,
            (
                f"Long-think job упал.\n"
                f"ID: {job['job_id'][:8]}\n"
                f"Ошибка: {exc}\n"
                f"JSON: {job['result_path']}"
            ),
        )
    finally:
        for task_key in ("progress_task", "metrics_task"):
            task = job.get(task_key)
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            job[task_key] = None


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    dialog_key = get_dialog_key(message)
    touch_dialog_state(dialog_key)
    logger.info(
        "Пользователь активировал бота: chat_id=%s user_id=%s username=%s",
        message.chat.id,
        message.from_user.id if message.from_user else None,
        message.from_user.username if message.from_user else None,
    )
    append_jsonl(
        {
            "timestamp": iso_now(),
            "event": "user_activated",
            "chat": chat_payload(message),
            "user": user_payload(message),
            "text": message.text,
        }
    )
    sent = await message.answer(
        "Бот запущен. Пиши текст.\n"
        "/help - подробная справка по возможностям порта.\n"
        "/mode - режим диалога chat/code; code mode тянет план, файлы, тесты и команды.\n"
        "/kb - локальная база знаний.\n"
        "/project - работа с проектами.\n"
        "/model и /models - локальные модели.\n"
        "/tasks - очередь фоновых задач.\n"
        "/reset - сбросить память диалога.\n"
        "/ineedmore - собрать несколько запросов в один пакет.\n"
        "/deepplan - оценить задачу и подобрать срок для long-think.\n"
        "/deepthink - запустить долгий режим размышления.\n"
        "/deepstatus - показать статус long-think job'ов.\n"
        "/deepcancel - отменить long-think job.\n"
        "/errors - показать свежие логи ошибок.\n"
        "/status - показать статус бота, модели и llama-server.\n"
        "/license - показать лицензию GNU AGPLv3.\n"
        "/source - показать ссылку на исходный код.",
        reply_markup=build_start_keyboard(),
    )
    track_bot_message(dialog_key, sent)


@router.message(Command("license"))
async def handle_license(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    sent = await message.answer(build_license_notice_text())
    track_bot_message(get_dialog_key(message), sent)


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    await answer_long(message, build_port_manual_text("telegram"), get_dialog_key(message))


@router.message(Command("errors"))
async def handle_errors(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    await answer_long(message, build_error_logs_text(), get_dialog_key(message))


@router.message(Command("mode"))
async def handle_mode(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    dialog_key = get_dialog_key(message)
    parts = (message.text or "").split(maxsplit=1)
    argument = parts[1] if len(parts) > 1 else ""
    sent = await message.answer(await execute_mode_command(argument, dialog_key))
    track_bot_message(dialog_key, sent)


@router.message(Command("kb"))
async def handle_kb(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    dialog_key = get_dialog_key(message)
    owner_key = get_long_think_owner_key(
        "telegram",
        user_id=message.from_user.id if message.from_user else None,
        chat_id=message.chat.id,
    )
    parts = (message.text or "").split(maxsplit=1)
    argument = parts[1] if len(parts) > 1 else ""
    sent = await message.answer(await execute_kb_command(argument, dialog_key, owner_key))
    track_bot_message(dialog_key, sent)


@router.message(Command("project"))
async def handle_project(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    dialog_key = get_dialog_key(message)
    owner_key = get_long_think_owner_key(
        "telegram",
        user_id=message.from_user.id if message.from_user else None,
        chat_id=message.chat.id,
    )
    parts = (message.text or "").split(maxsplit=1)
    argument = parts[1] if len(parts) > 1 else ""
    sent = await message.answer(await execute_project_command(argument, dialog_key, owner_key))
    track_bot_message(dialog_key, sent)


@router.message(Command("model"))
async def handle_model(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    dialog_key = get_dialog_key(message)
    parts = (message.text or "").split(maxsplit=1)
    argument = parts[1] if len(parts) > 1 else ""
    sent = await message.answer(await execute_model_command(argument))
    track_bot_message(dialog_key, sent)


@router.message(Command("models"))
async def handle_models(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    sent = await message.answer(render_models_text())
    track_bot_message(get_dialog_key(message), sent)


@router.message(Command("tasks"))
async def handle_tasks(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    dialog_key = get_dialog_key(message)
    owner_key = get_long_think_owner_key(
        "telegram",
        user_id=message.from_user.id if message.from_user else None,
        chat_id=message.chat.id,
    )
    parts = (message.text or "").split(maxsplit=1)
    argument = parts[1] if len(parts) > 1 else ""
    sent = await message.answer(await execute_tasks_command(argument, owner_key))
    track_bot_message(dialog_key, sent)


@router.message(Command("deepplan"))
async def handle_deepplan(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return

    owner_key = get_long_think_owner_key(
        "telegram",
        user_id=message.from_user.id if message.from_user else None,
        chat_id=message.chat.id,
    )
    active_job = get_active_long_think_job(owner_key)
    if active_job is not None:
        sent = await message.answer(
            "У тебя уже есть активный long-think job.\n"
            f"ID: {active_job['job_id'][:8]}\n"
            f"Статус: {active_job['status']}\n"
            f"Папка: {active_job['artifact_dir']}"
        )
        track_bot_message(get_dialog_key(message), sent)
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        sent = await message.answer(build_deepplan_usage_text("/deepplan"))
        track_bot_message(get_dialog_key(message), sent)
        return

    request_text = parts[1].strip()
    dialog_key = get_dialog_key(message)
    waiting_message = await message.answer(
        render_waiting_text(
            "Изучаю задачу для deepthink",
            0,
            ["Считаю объём работы и прикидываю адекватный срок."],
        )
    )
    track_bot_message(dialog_key, waiting_message)
    wait_indicator = TelegramWaitIndicator(
        waiting_message,
        lambda elapsed_seconds: render_waiting_text(
            "Изучаю задачу для deepthink",
            elapsed_seconds,
            ["Считаю объём работы и прикидываю адекватный срок."],
        ),
        dialog_key=dialog_key,
    )
    await wait_indicator.start()
    try:
        plan = await create_pending_long_think_plan(
            mode="telegram",
            owner_key=owner_key,
            request_text=request_text,
            chat_id=message.chat.id,
            user_id=message.from_user.id if message.from_user else None,
        )
    except Exception as exc:
        await wait_indicator.stop()
        waiting_message = wait_indicator.message
        waiting_message = await safe_edit_message(
            waiting_message,
            f"Не удалось собрать deepplan: {exc}\n\n{build_deepplan_usage_text('/deepplan')}",
        )
        track_bot_message(dialog_key, waiting_message)
        return

    await wait_indicator.stop()
    waiting_message = wait_indicator.message
    append_jsonl(
        {
            "timestamp": iso_now(),
            "event": "long_think_planned",
            "chat": chat_payload(message),
            "user": user_payload(message),
            "mode": "telegram",
            "owner_key": owner_key,
            "plan_id": plan["plan_id"],
            "source": plan.get("source"),
            "recommended_duration_seconds": plan.get("recommended_duration_seconds"),
            "request_text": request_text,
        }
    )
    waiting_message = await safe_edit_message(
        waiting_message,
        build_long_think_plan_text(plan, terminal_mode=False),
        reply_markup=build_deepplan_keyboard(
            str(plan["plan_id"]),
            int(plan["recommended_duration_seconds"]),
        ),
    )
    track_bot_message(dialog_key, waiting_message)


@router.callback_query(F.data == LICENSE_CALLBACK)
async def handle_license_callback(callback: CallbackQuery) -> None:
    if await reject_if_blocked_callback(callback):
        return
    if callback.message is not None:
        sent = await callback.message.answer(build_license_notice_text())
        track_bot_message(get_callback_dialog_key(callback), sent)
    await callback.answer()


@router.message(Command("source"))
async def handle_source(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    text = (
        f"Исходный код: {SOURCE_URL}"
        if SOURCE_URL
        else "SOURCE_URL не задан. Укажи ссылку на репозиторий в переменной окружения."
    )
    sent = await message.answer(text)
    track_bot_message(get_dialog_key(message), sent)


@router.message(Command("status"))
async def handle_status(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    sent = await message.answer(await build_runtime_status_text())
    track_bot_message(get_dialog_key(message), sent)


@router.message(Command("deepstatus"))
async def handle_deepstatus(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    owner_key = get_long_think_owner_key(
        "telegram",
        user_id=message.from_user.id if message.from_user else None,
        chat_id=message.chat.id,
    )
    sent = await message.answer(render_long_think_jobs_status(owner_key))
    track_bot_message(get_dialog_key(message), sent)


@router.message(Command("deepcancel"))
async def handle_deepcancel(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    owner_key = get_long_think_owner_key(
        "telegram",
        user_id=message.from_user.id if message.from_user else None,
        chat_id=message.chat.id,
    )
    parts = (message.text or "").split(maxsplit=1)
    requested_job_id = parts[1].strip() if len(parts) > 1 else ""
    job = find_long_think_job_for_owner(owner_key, requested_job_id or None)
    if job is None or not is_long_think_active_status(str(job.get("status") or "")):
        sent = await message.answer("Активный long-think job не найден.")
        track_bot_message(get_dialog_key(message), sent)
        return

    job["cancel_requested"] = True
    job["error"] = "Long-think job был отменён пользователем."
    request_long_think_cancel(job)
    sent = await message.answer(
        f"Останавливаю long-think job {job['job_id'][:8]}. "
        f"Артефакты останутся в {job['artifact_dir']}."
    )
    track_bot_message(get_dialog_key(message), sent)


@router.message(Command("deepthink"))
async def handle_deepthink(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return

    owner_key = get_long_think_owner_key(
        "telegram",
        user_id=message.from_user.id if message.from_user else None,
        chat_id=message.chat.id,
    )
    active_job = get_active_long_think_job(owner_key)
    if active_job is not None:
        sent = await message.answer(
            "У тебя уже есть активный long-think job.\n"
            f"ID: {active_job['job_id'][:8]}\n"
            f"Статус: {active_job['status']}\n"
            f"Папка: {active_job['artifact_dir']}"
        )
        track_bot_message(get_dialog_key(message), sent)
        return

    parts = (message.text or "").split(maxsplit=2)
    if len(parts) < 3:
        sent = await message.answer(build_deepthink_usage_text("/deepthink"))
        track_bot_message(get_dialog_key(message), sent)
        return

    try:
        duration_seconds = parse_duration_spec(parts[1])
    except ValueError as exc:
        sent = await message.answer(f"{exc}\n\n{build_deepthink_usage_text('/deepthink')}")
        track_bot_message(get_dialog_key(message), sent)
        return

    request_text = parts[2].strip()
    if not request_text:
        sent = await message.answer(build_deepthink_usage_text("/deepthink"))
        track_bot_message(get_dialog_key(message), sent)
        return

    job = start_long_think_job(
        mode="telegram",
        owner_key=owner_key,
        request_text=request_text,
        duration_seconds=duration_seconds,
        chat_id=message.chat.id,
        user_id=message.from_user.id if message.from_user else None,
        bot=message.bot,
    )
    append_long_think_started_event(
        job=job,
        request_text=request_text,
        source="manual",
        chat=chat_payload(message),
        user=user_payload(message),
    )
    sent = await message.answer(
        build_long_think_started_text(job)
        + "\nПроверять статус можно командой /deepstatus, остановить - /deepcancel."
    )
    track_bot_message(get_dialog_key(message), sent)


@router.callback_query(F.data.startswith(DEEPPLAN_START_CALLBACK_PREFIX))
async def handle_deepplan_start_callback(callback: CallbackQuery) -> None:
    if await reject_if_blocked_callback(callback):
        return
    if callback.message is None:
        await callback.answer("Сообщение с планом уже куда-то делось.", show_alert=True)
        return

    owner_key = get_long_think_owner_key(
        "telegram",
        user_id=callback.from_user.id if callback.from_user else None,
        chat_id=callback.message.chat.id,
    )
    plan_id = str(callback.data or "")[len(DEEPPLAN_START_CALLBACK_PREFIX) :].strip()
    plan = get_pending_long_think_plan(plan_id, owner_key=owner_key)
    if plan is None:
        await callback.answer("План уже протух или его уже дёрнули.", show_alert=True)
        return

    active_job = get_active_long_think_job(owner_key)
    if active_job is not None:
        await callback.answer("У тебя уже есть активный long-think job.", show_alert=True)
        return

    plan = pop_pending_long_think_plan(plan_id, owner_key=owner_key) or plan
    job = start_long_think_from_plan(plan, bot=callback.message.bot)
    append_long_think_started_event(
        job=job,
        request_text=str(plan.get("request_text") or ""),
        source="plan",
        chat={
            "id": callback.message.chat.id,
            "type": callback.message.chat.type,
            "title": getattr(callback.message.chat, "title", None),
        },
        user={
            "id": callback.from_user.id if callback.from_user else None,
            "username": callback.from_user.username if callback.from_user else None,
            "first_name": callback.from_user.first_name if callback.from_user else None,
            "last_name": callback.from_user.last_name if callback.from_user else None,
            "full_name": callback.from_user.full_name if callback.from_user else None,
            "language_code": callback.from_user.language_code if callback.from_user else None,
            "is_bot": callback.from_user.is_bot if callback.from_user else None,
        },
        plan_id=plan_id,
    )
    updated = await safe_edit_message(
        callback.message,
        build_long_think_started_text(
            job,
            intro_line="Запускаю long-think по рассчитанному плану.",
        )
        + "\nПроверять статус можно командой /deepstatus, остановить - /deepcancel.",
        reply_markup=None,
    )
    track_bot_message(get_callback_dialog_key(callback), updated)
    await callback.answer("Погнали. Long-think уже поехал.")


@router.callback_query(F.data.startswith(DEEPPLAN_CANCEL_CALLBACK_PREFIX))
async def handle_deepplan_cancel_callback(callback: CallbackQuery) -> None:
    if await reject_if_blocked_callback(callback):
        return
    if callback.message is None:
        await callback.answer("Сообщение с планом уже пропало.", show_alert=True)
        return

    owner_key = get_long_think_owner_key(
        "telegram",
        user_id=callback.from_user.id if callback.from_user else None,
        chat_id=callback.message.chat.id,
    )
    plan_id = str(callback.data or "")[len(DEEPPLAN_CANCEL_CALLBACK_PREFIX) :].strip()
    discard_pending_long_think_plan(plan_id, owner_key=owner_key)
    updated = await safe_edit_message(
        callback.message,
        "Ок, этот deepplan выкинул. Если передумаешь, запусти /deepplan заново.",
        reply_markup=None,
    )
    track_bot_message(get_callback_dialog_key(callback), updated)
    await callback.answer("Ок, не запускаю.")


@router.callback_query(F.data == SOURCE_CODE_CALLBACK)
async def handle_source_callback(callback: CallbackQuery) -> None:
    if await reject_if_blocked_callback(callback):
        return
    if callback.message is not None:
        text = (
            f"Исходный код: {SOURCE_URL}"
            if SOURCE_URL
            else "SOURCE_URL не задан. Укажи ссылку на репозиторий в переменной окружения."
        )
        sent = await callback.message.answer(text)
        track_bot_message(get_callback_dialog_key(callback), sent)
    await callback.answer()


@router.callback_query(F.data == RESET_DIALOG_CALLBACK)
async def handle_reset_dialog_callback(callback: CallbackQuery) -> None:
    if await reject_if_blocked_callback(callback):
        return
    dialog_key = get_callback_dialog_key(callback)
    reset_dialog(dialog_key)
    multi_request_sessions.pop(dialog_key, None)
    message_ids = forget_tracked_bot_messages(dialog_key)

    append_jsonl(
        {
            "timestamp": iso_now(),
            "event": "dialog_reset_callback",
            "chat_id": callback.message.chat.id if callback.message is not None else 0,
            "user_id": callback.from_user.id if callback.from_user else None,
            "message_ids": message_ids,
        }
    )

    await callback.answer("Диалог сброшен. Сообщения бота удаляю.")

    chat_id = callback.message.chat.id if callback.message is not None else None
    if chat_id is None:
        return

    for message_id in sorted(set(message_ids), reverse=True):
        try:
            await callback.bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception:
            continue


@router.message(Command("reset"))
async def handle_reset(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    dialog_key = get_dialog_key(message)
    reset_dialog(dialog_key)
    multi_request_sessions.pop(dialog_key, None)
    forget_tracked_bot_messages(dialog_key)
    append_jsonl(
        {
            "timestamp": iso_now(),
            "event": "dialog_reset",
            "chat": chat_payload(message),
            "user": user_payload(message),
            "text": message.text,
        }
    )
    sent = await message.answer("Память диалога очищена.")
    track_bot_message(dialog_key, sent)


@router.message(Command("ineedmore"))
async def handle_ineedmore(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    dialog_key = get_dialog_key(message)
    touch_dialog_state(dialog_key)
    session = create_multi_request_session()
    multi_request_sessions[dialog_key] = session
    append_jsonl(
        {
            "timestamp": iso_now(),
            "event": "multi_request_opened",
            "chat": chat_payload(message),
            "user": user_payload(message),
        }
    )
    sent = await message.answer(INEEDMORE_WELCOME_TEXT)
    track_bot_message(dialog_key, sent)
    sent = await message.answer(
        render_multi_request_form(session["topic"], session["queries"]),
        reply_markup=build_ineedmore_keyboard(len(session["queries"])),
    )
    track_bot_message(dialog_key, sent)


async def process_multi_request(
    status_message: Message,
    dialog_key: str,
    request_id: str,
    topic: str,
    queries: list[str],
    user_id: int | None,
) -> None:
    chat_lock = get_chat_lock(status_message.chat.id)
    statuses = ["pending"] * len(queries)
    answers = [""] * len(queries)
    current_extra_line: str | None = None

    def render_status_text(elapsed_seconds: int) -> str:
        return render_multi_request_status(
            topic,
            queries,
            statuses,
            elapsed_seconds=elapsed_seconds,
            extra_line=current_extra_line,
        )

    wait_indicator = TelegramWaitIndicator(
        status_message,
        render_status_text,
        dialog_key=dialog_key,
    )
    await wait_indicator.start()

    async def refresh_status(extra_line: str | None = None) -> None:
        nonlocal status_message
        nonlocal current_extra_line
        current_extra_line = extra_line
        await wait_indicator.refresh(force=True)
        status_message = wait_indicator.message

    append_jsonl(
        {
            "timestamp": iso_now(),
            "event": "multi_request_started",
            "request_id": request_id,
            "chat_id": status_message.chat.id,
            "user_id": user_id,
            "topic": topic,
            "queries": queries,
        }
    )

    async def run_item(index: int, query: str) -> None:
        messages = build_multi_request_item_messages(topic, query)
        if model_lock is None:
            raise RuntimeError("Лок модели не инициализирован.")

        async with acquire_model_slot(f"ineedmore:{status_message.chat.id}:{index}"):
            statuses[index] = "thinking"
            await refresh_status()
            reply, raw_reply, _ = await collect_model_reply_unlocked(messages, INEEDMORE_ITEM_MAX_TOKENS)
            if needs_repair_pass(reply, raw_reply, brief_mode=False):
                repair_messages = [
                    {
                        "role": "system",
                        "content": build_system_message_content(
                            query, REPAIR_SYSTEM_PROMPT, base_prompt=SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Тема: {topic}\n{query}" if topic.strip() else query
                        ),
                    },
                ]
                repaired_reply, _, _ = await collect_model_reply_unlocked(repair_messages, INEEDMORE_ITEM_MAX_TOKENS)
                if repaired_reply.strip():
                    reply = repaired_reply

        answers[index] = (
            reply.strip() or "Не удалось получить внятный ответ по этому пункту."
        )
        statuses[index] = "done"
        await refresh_status()

    try:
        async with chat_lock:
            await refresh_status()
            tasks = [
                asyncio.create_task(run_item(index, query))
                for index, query in enumerate(queries)
            ]
            await asyncio.gather(*tasks)

            await refresh_status("Собираю общий итог 🧩")
            intro_messages = build_multi_request_intro_messages(topic, queries, answers)
            intro, _, _ = await collect_model_reply(
                intro_messages,
                INEEDMORE_INTRO_MAX_TOKENS,
                brief_mode=True,
                label=f"ineedmore-intro:{status_message.chat.id}",
            )
            intro = intro.strip()
            if not intro:
                intro = (
                    f'Собрал ответы по теме "{topic}".'
                    if topic.strip()
                    else "Собрал ответы по всем шаблонам."
                )
    finally:
        await wait_indicator.stop()
        status_message = wait_indicator.message

    final_lines: list[str] = []
    if topic.strip():
        final_lines.append(f"Тема: {topic}")
        final_lines.append("")
    final_lines.append(intro)
    final_lines.append("")
    for index, (query, answer) in enumerate(zip(queries, answers), start=1):
        final_lines.append(f"{index}. {query}")
        final_lines.append(answer)
        final_lines.append("")
    final_text = "\n".join(line for line in final_lines).strip()

    await refresh_status("Все шаблоны готовы ✅")
    await answer_long(status_message, final_text, dialog_key=dialog_key)

    append_jsonl(
        {
            "timestamp": iso_now(),
            "event": "multi_request_response",
            "request_id": request_id,
            "chat_id": status_message.chat.id,
            "user_id": user_id,
            "topic": topic,
            "queries": queries,
            "answers": answers,
            "final_text": final_text,
        }
    )


@router.callback_query(F.data.startswith(INEEDMORE_CALLBACK_PREFIX))
async def handle_ineedmore_callback(callback: CallbackQuery) -> None:
    if await reject_if_blocked_callback(callback):
        return
    if callback.message is None or callback.data is None:
        await callback.answer()
        return

    user_id = callback.from_user.id if callback.from_user else 0
    dialog_key = f"{callback.message.chat.id}:{user_id}"
    session = get_multi_request_session(dialog_key)
    if session is None:
        await callback.answer("Сессия /ineedmore не найдена.", show_alert=True)
        return

    queries = session["queries"]
    topic = session["topic"]

    if callback.data == INEEDMORE_ACTION_EDIT:
        session["edit_target"] = None
        await safe_edit_message(
            callback.message,
            render_multi_request_edit_menu(topic, queries),
            reply_markup=build_ineedmore_edit_keyboard(len(queries)),
        )
        await callback.answer("Выбери, что редактировать.")
        return

    if callback.data == INEEDMORE_ACTION_EDIT_BACK:
        session["edit_target"] = None
        await safe_edit_message(
            callback.message,
            render_multi_request_form(topic, queries),
            reply_markup=build_ineedmore_keyboard(len(queries)),
        )
        await callback.answer()
        return

    if callback.data == INEEDMORE_ACTION_EDIT_TOPIC:
        session["edit_target"] = "topic"
        await safe_edit_message(
            callback.message,
            render_multi_request_edit_prompt("topic", queries),
            reply_markup=build_ineedmore_edit_keyboard(len(queries)),
        )
        await callback.answer("Жду новое значение темы.")
        return

    if callback.data.startswith(INEEDMORE_ACTION_EDIT_QUERY_PREFIX):
        raw_index = callback.data.removeprefix(INEEDMORE_ACTION_EDIT_QUERY_PREFIX)
        if not raw_index.isdigit():
            await callback.answer()
            return
        query_index = int(raw_index)
        if query_index < 0 or query_index >= len(queries):
            await callback.answer("Такого пункта уже нет.", show_alert=True)
            return
        session["edit_target"] = f"query:{query_index}"
        await safe_edit_message(
            callback.message,
            render_multi_request_edit_prompt(session["edit_target"], queries),
            reply_markup=build_ineedmore_edit_keyboard(len(queries)),
        )
        await callback.answer(f"Жду новое значение для запроса {query_index + 1}.")
        return

    if callback.data == INEEDMORE_ACTION_ADD:
        if len(queries) >= MAX_MULTI_REQUEST_ITEMS:
            await callback.answer("Больше трёх пунктов пока не делаю.", show_alert=True)
            return
        queries.append("")
        session["queries"] = queries
        await safe_edit_message(
            callback.message,
            render_multi_request_form(topic, queries),
            reply_markup=build_ineedmore_keyboard(len(queries)),
        )
        await callback.answer("Пункт добавлен.")
        return

    if callback.data == INEEDMORE_ACTION_REMOVE:
        if len(queries) <= 1:
            await callback.answer("Ниже одного пункта не опускаю.", show_alert=True)
            return
        queries.pop()
        session["queries"] = queries
        edit_target = session.get("edit_target")
        if isinstance(edit_target, str) and edit_target.startswith("query:"):
            edit_index = int(edit_target.split(":", 1)[1])
            if edit_index >= len(queries):
                session["edit_target"] = None
        await safe_edit_message(
            callback.message,
            render_multi_request_form(topic, queries),
            reply_markup=build_ineedmore_keyboard(len(queries)),
        )
        await callback.answer("Пункт убран.")
        return

    if callback.data == INEEDMORE_ACTION_CANCEL:
        multi_request_sessions.pop(dialog_key, None)
        await safe_edit_message(callback.message, "Режим /ineedmore отменён.")
        await callback.answer("Отменено.")
        return

    if callback.data == INEEDMORE_ACTION_CONFIRM:
        prepared_queries = [query.strip() for query in queries]
        if not prepared_queries or not all(prepared_queries):
            await callback.answer(
                "Сначала заполни все пункты шаблона и отправь его в чат.",
                show_alert=True,
            )
            return

        multi_request_sessions.pop(dialog_key, None)
        await callback.answer("Запускаю обработку.")
        await process_multi_request(
            callback.message,
            dialog_key,
            uuid.uuid4().hex,
            topic.strip(),
            prepared_queries,
            user_id,
        )
        return

    await callback.answer()


@router.message(F.text)
async def handle_text(message: Message, bot: Bot) -> None:
    global model_lock

    if await reject_if_blocked_message(message):
        return

    text = (message.text or "").strip()
    if not text:
        await message.answer("Пустое сообщение не обрабатываю.")
        return

    dialog_key = get_dialog_key(message)
    user_id = message.from_user.id if message.from_user else None
    chat_lock = get_chat_lock(message.chat.id)
    multi_session = get_multi_request_session(dialog_key)

    if multi_session is not None:
        too_long_error = get_text_limit_error(
            text, MAX_MULTI_REQUEST_TEXT_CHARS, "Текст для /ineedmore"
        )
        if too_long_error:
            await message.answer(too_long_error)
            return

        edit_target = multi_session.get("edit_target")
        if isinstance(edit_target, str):
            if edit_target == "topic":
                multi_session["topic"] = text
            elif edit_target.startswith("query:"):
                query_index = int(edit_target.split(":", 1)[1])
                if 0 <= query_index < len(multi_session["queries"]):
                    multi_session["queries"][query_index] = text
            multi_session["edit_target"] = None
            append_jsonl(
                {
                    "timestamp": iso_now(),
                    "event": "multi_request_field_updated",
                    "chat": chat_payload(message),
                    "user": user_payload(message),
                    "target": edit_target,
                    "topic": multi_session["topic"],
                    "queries": multi_session["queries"],
                }
            )
            await message.answer(
                render_multi_request_form(
                    multi_session["topic"], multi_session["queries"]
                ),
                reply_markup=build_ineedmore_keyboard(len(multi_session["queries"])),
            )
            return

        parsed_form = parse_multi_request_form(text, len(multi_session["queries"]))
        if parsed_form is None:
            await message.answer(
                render_multi_request_form(
                    multi_session["topic"], multi_session["queries"]
                ),
                reply_markup=build_ineedmore_keyboard(len(multi_session["queries"])),
            )
            return

        topic, queries = parsed_form
        topic_error = get_text_limit_error(topic, MAX_MULTI_REQUEST_TEXT_CHARS, "Тема")
        if topic_error:
            await message.answer(topic_error)
            return
        for index, query in enumerate(queries, start=1):
            query_error = get_text_limit_error(
                query, MAX_MULTI_REQUEST_TEXT_CHARS, f"Запрос {index}"
            )
            if query_error:
                await message.answer(query_error)
                return
        multi_session["topic"] = topic
        multi_session["queries"] = queries
        append_jsonl(
            {
                "timestamp": iso_now(),
                "event": "multi_request_form_updated",
                "chat": chat_payload(message),
                "user": user_payload(message),
                "topic": topic,
                "queries": queries,
            }
        )
        await message.answer(
            render_multi_request_form(topic, queries),
            reply_markup=build_ineedmore_keyboard(len(queries)),
        )
        return

    user_text_too_long = get_text_limit_error(text, MAX_USER_TEXT_CHARS, "Сообщение")
    if user_text_too_long:
        await message.answer(user_text_too_long)
        return

    if chat_lock.locked():
        await message.answer("Подожди, я еще не закончил предыдущий ответ.")
        return

    request_id = uuid.uuid4().hex
    request_started_at = iso_now()

    logger.info(
        "Новое сообщение: request_id=%s chat_id=%s user_id=%s text=%r",
        request_id,
        message.chat.id,
        message.from_user.id if message.from_user else None,
        text,
    )
    print(f"\n[USER {message.chat.id}] {text}", flush=True)

    append_jsonl(
        {
            "timestamp": request_started_at,
            "event": "user_message",
            "request_id": request_id,
            "chat": chat_payload(message),
            "user": user_payload(message),
            "text": text,
        }
    )

    editor = StreamingTelegramEditor(message, dialog_key)
    raw_reply = ""
    full_reply = ""
    brief_mode = should_answer_briefly_for_dialog(dialog_key, text)
    brief_fallback = get_brief_fallback_reply(text) if brief_mode else None
    used_brief_fallback = False
    request_max_tokens = get_request_max_tokens_for_dialog(dialog_key, text)
    reply_finish_reason: str | None = None

    async with chat_lock:
        try:
            async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
                await editor.start()
                logger.info(
                    "Режим ответа: request_id=%s mode=%s max_tokens=%s",
                    request_id,
                    "brief" if brief_mode else "detailed",
                    request_max_tokens,
                )
                (
                    full_reply,
                    raw_reply,
                    reply_finish_reason,
                    used_brief_fallback,
                    retried_with_clean_history,
                ) = await collect_dialog_reply(
                    dialog_key,
                    text,
                    request_max_tokens,
                    brief_mode=brief_mode,
                    label=f"chat:{message.chat.id}:{request_id}",
                    allow_history_retry=True,
                )

                logger.info(
                    "Генерация завершена: request_id=%s finish_reason=%s retried_with_clean_history=%s used_brief_fallback=%s",
                    request_id,
                    reply_finish_reason,
                    retried_with_clean_history,
                    used_brief_fallback,
                )

                await editor.flush(full_reply, final=True)

            if not full_reply.strip():
                if USE_RAW_MODEL_REPLY and raw_reply.strip():
                    full_reply = normalize_raw_model_reply(raw_reply)
                    await editor.flush(full_reply, final=True)

            if not full_reply.strip():
                salvaged_reply = extract_relaxed_visible_reply(raw_reply)
                if brief_mode:
                    salvaged_reply = compress_brief_reply(salvaged_reply)
                    if brief_fallback and (
                        not salvaged_reply.strip()
                        or looks_like_prompt_leak(salvaged_reply)
                        or not is_final_reply_candidate(salvaged_reply)
                    ):
                        logger.info(
                            "Использую brief-fallback перед финальной заглушкой: request_id=%s",
                            request_id,
                        )
                        salvaged_reply = brief_fallback
                        used_brief_fallback = True
                if salvaged_reply.strip():
                    full_reply = salvaged_reply
                    await editor.flush(full_reply, final=True)

            if not full_reply.strip():
                full_reply = (
                    "Не удалось выделить финальный ответ модели."
                    if raw_reply.strip()
                    else "Модель ничего не вернула."
                )
                await editor.flush(full_reply, final=True)

            remember_turn(
                dialog_key,
                text,
                sanitize_assistant_reply_for_history(full_reply, raw_reply),
            )

            append_jsonl(
                {
                    "timestamp": iso_now(),
                    "event": "bot_response",
                    "request_id": request_id,
                    "chat": chat_payload(message),
                    "user": user_payload(message),
                    "user_text": text,
                    "bot_text": full_reply,
                }
            )
            logger.info(
                "Ответ отправлен: request_id=%s symbols=%s",
                request_id,
                len(full_reply),
            )
        except Exception as exc:
            logger.exception("Ошибка обработки request_id=%s", request_id)
            error_text = f"Ошибка: {exc}"
            append_jsonl(
                {
                    "timestamp": iso_now(),
                    "event": "error",
                    "request_id": request_id,
                    "chat": chat_payload(message),
                    "user": user_payload(message),
                    "user_text": text,
                    "error": str(exc),
                }
            )

            if editor.current_message is not None:
                await editor.show_error(error_text)
            else:
                await message.answer(error_text)


@router.message()
async def handle_other(message: Message) -> None:
    if await reject_if_blocked_message(message):
        return
    await message.answer("Я сейчас принимаю только обычный текст.")


async def bot_worker_main() -> None:
    global model_lock

    ensure_stdout_utf8()
    setup_logging()
    bootstrap_from_interactions(INTERACTIONS_LOG_PATH)
    mark_interrupted_long_think_jobs()
    ensure_feature_roots()
    load_background_tasks_from_disk()
    validate_config()

    model_lock = asyncio.Lock()
    ensure_background_task_worker_running()

    session = AiohttpSession(
        timeout=aiohttp.ClientTimeout(
            total=TELEGRAM_REQUEST_TIMEOUT,
            connect=TELEGRAM_REQUEST_TIMEOUT,
            sock_connect=TELEGRAM_REQUEST_TIMEOUT,
            sock_read=TELEGRAM_REQUEST_TIMEOUT,
        )
    )
    bot = Bot(token=BOT_TOKEN, session=session)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    await ensure_telegram_ready(bot)

    await ensure_llama_server_running()
    logger.info("Бот запущен и готов к polling.")
    try:
        await run_polling_forever(bot, dispatcher)
    finally:
        await bot.session.close()
        stop_llama_server()


async def async_terminal_input(prompt: str) -> str:
    return strip_terminal_control_sequences(
        await asyncio.to_thread(read_console_input, prompt)
    )


def should_reject_terminal_input(text: str) -> bool:
    normalized = strip_terminal_control_sequences(text).strip()
    if not normalized:
        return False
    if not looks_broken_console_text(normalized):
        return False
    score = score_console_text_quality(normalized)[0]
    return score < 0


def render_terminal_prompt(saved_char_limit: int | None) -> str:
    if saved_char_limit is None:
        return "Запрос [лимит спросим перед ответом]: "
    return f"Запрос [лимит {saved_char_limit}]: "


def render_terminal_help_text() -> str:
    return (
        "Terminal-справка:\n"
        "Сессия сохраняется автоматически после каждого ответа и основных команд.\n\n"
        f"{build_port_manual_text('terminal')}\n\n"
        "Короткий список terminal-команд:\n"
        "/help - показать эту справку\n"
        "/mode show|chat|code - режим текущего диалога; code mode даёт план, файлы, тесты, команды\n"
        "/kb ... - локальная БЗ\n"
        "/project ... - работа с проектами\n"
        "/model ... и /models - менеджер моделей\n"
        "/tasks ... - очередь фоновых задач\n"
        "/session ... - управление текущей сессией\n"
        "/reset - сбросить память диалога\n"
        "/deepplan <запрос> - оценить задачу и предложить срок для deepthink\n"
        "/limit show - показать текущий сохранённый лимит\n"
        "/limit <число> - сохранить лимит символов\n"
        "/limit off - убрать сохранённый лимит\n"
        "/limit ask - снова спрашивать лимит перед ответом\n"
        "/repeat - повторно отправить последний обычный запрос\n"
        "/clipboard show|user|bot|clear|set <текст> - буфер terminal-сессии\n"
        "/paste - подставить буфер в следующую строку ввода\n"
        "/errors - показать свежие логи ошибок\n"
        "/exit - выйти"
    )


def render_terminal_clipboard_text(session: dict[str, Any]) -> str:
    return (
        "Буфер terminal-сессии:\n"
        f"- clipboard: {build_terminal_clipboard_preview(session.get('clipboard_text'))}\n"
        f"- last user: {build_terminal_clipboard_preview(session.get('last_user_text'))}\n"
        f"- last bot: {build_terminal_clipboard_preview(session.get('last_bot_text'))}"
    )


async def read_terminal_char_limit() -> int:
    while True:
        raw_value = (await async_terminal_input("Максимум символов в ответе [2000]: ")).strip()
        if not raw_value:
            return 2000
        if raw_value.isdigit():
            value = int(raw_value)
            if 1 <= value <= 10000:
                return value
        print("Нужно число от 1 до 10000.", flush=True)


async def ask_to_save_terminal_char_limit(char_limit: int) -> bool:
    while True:
        raw_value = (
            await async_terminal_input(
                f"Сохранить лимит {char_limit} символов для следующих запросов? [д/н]: "
            )
        ).strip().lower()
        if raw_value in {"д", "да", "y", "yes"}:
            return True
        if raw_value in {"н", "нет", "n", "no", ""}:
            return False
        print("Ответь 'да' или 'нет'.", flush=True)


async def ask_terminal_yes_no(prompt: str, *, default: bool = False) -> bool:
    while True:
        raw_value = (await async_terminal_input(prompt)).strip().lower()
        if raw_value in {"д", "да", "y", "yes"}:
            return True
        if raw_value in {"н", "нет", "n", "no"}:
            return False
        if raw_value == "":
            return default
        print("Ответь 'да' или 'нет'.", flush=True)


async def collect_terminal_reply(
    dialog_key: str,
    user_text: str,
    request_max_tokens: int,
    *,
    brief_mode: bool,
    label: str,
) -> tuple[str, str, str | None, bool]:
    full_reply, raw_reply, finish_reason, _, retried_with_clean_history = await collect_dialog_reply(
        dialog_key,
        user_text,
        request_max_tokens,
        brief_mode=brief_mode,
        label=label,
        allow_history_retry=True,
    )
    return full_reply, raw_reply, finish_reason, retried_with_clean_history


async def handle_terminal_command(
    user_text: str,
    terminal_session: dict[str, Any],
    dialog_key: str,
    terminal_owner_key: str,
    saved_char_limit: int | None,
    char_limit_save_choice_made: bool,
) -> tuple[bool, int | None, bool, str | None]:
    stripped = user_text.strip()
    if not stripped.startswith("/"):
        return False, saved_char_limit, char_limit_save_choice_made, None

    if stripped in {"/help", "/h"}:
        print(render_terminal_help_text() + "\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped == "/errors":
        print(build_error_logs_text() + "\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/mode"):
        parts = stripped.split(maxsplit=1)
        argument = parts[1] if len(parts) > 1 else ""
        print(await execute_mode_command(argument, dialog_key) + "\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/kb"):
        parts = stripped.split(maxsplit=1)
        argument = parts[1] if len(parts) > 1 else ""
        print(await execute_kb_command(argument, dialog_key, terminal_owner_key) + "\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/project"):
        parts = stripped.split(maxsplit=1)
        argument = parts[1] if len(parts) > 1 else ""
        print(await execute_project_command(argument, dialog_key, terminal_owner_key) + "\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped == "/models":
        print(render_models_text() + "\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/model"):
        parts = stripped.split(maxsplit=1)
        argument = parts[1] if len(parts) > 1 else ""
        print(await execute_model_command(argument) + "\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/tasks"):
        parts = stripped.split(maxsplit=1)
        argument = parts[1] if len(parts) > 1 else ""
        print(await execute_tasks_command(argument, terminal_owner_key) + "\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/session"):
        parts = stripped.split(maxsplit=1)
        argument = parts[1] if len(parts) > 1 else ""
        print(await execute_session_command(argument, terminal_session, dialog_key) + "\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped == "/reset":
        reset_dialog(dialog_key)
        print("Память терминального диалога очищена.\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped == "/repeat":
        last_user_text = normalize_terminal_clipboard_text(
            terminal_session.get("last_user_text")
        )
        if not last_user_text:
            print("Повторять пока нечего: обычного прошлого запроса нет.\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        print(
            "Повторяю последний обычный запрос из terminal-сессии.\n",
            flush=True,
        )
        return True, saved_char_limit, char_limit_save_choice_made, last_user_text

    if stripped == "/paste":
        clipboard_text = normalize_terminal_clipboard_text(
            terminal_session.get("clipboard_text")
        )
        if not clipboard_text:
            print("Буфер пустой. Сначала заполни его через /clipboard user|bot|set.\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        set_readline_prefill_text(clipboard_text)
        print("Ок, подставлю буфер в следующую строку ввода.\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/clipboard"):
        parts = stripped.split(maxsplit=2)
        argument = parts[1].strip().lower() if len(parts) > 1 else "show"
        if argument in {"", "show"}:
            print(render_terminal_clipboard_text(terminal_session) + "\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        if argument == "user":
            last_user_text = normalize_terminal_clipboard_text(
                terminal_session.get("last_user_text")
            )
            if not last_user_text:
                print("Последнего обычного запроса пока нет.\n", flush=True)
                return True, saved_char_limit, char_limit_save_choice_made, None
            terminal_session["clipboard_text"] = last_user_text
            print("Последний user-запрос скопировал в буфер.\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        if argument == "bot":
            last_bot_text = normalize_terminal_clipboard_text(
                terminal_session.get("last_bot_text")
            )
            if not last_bot_text:
                print("Последнего bot-ответа пока нет.\n", flush=True)
                return True, saved_char_limit, char_limit_save_choice_made, None
            terminal_session["clipboard_text"] = last_bot_text
            print("Последний bot-ответ скопировал в буфер.\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        if argument == "clear":
            terminal_session["clipboard_text"] = ""
            print("Буфер terminal-сессии очищен.\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        if argument == "set":
            clipboard_text = normalize_terminal_clipboard_text(parts[2] if len(parts) > 2 else "")
            if not clipboard_text:
                print("Использование: /clipboard set <текст>\n", flush=True)
                return True, saved_char_limit, char_limit_save_choice_made, None
            terminal_session["clipboard_text"] = clipboard_text
            print("Текст сохранил в буфер terminal-сессии.\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        print(
            "Использование: /clipboard show | /clipboard user | /clipboard bot | "
            "/clipboard clear | /clipboard set <текст>\n",
            flush=True,
        )
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/limit"):
        parts = stripped.split(maxsplit=1)
        argument = parts[1].strip().lower() if len(parts) > 1 else ""
        if not argument or argument == "show":
            if saved_char_limit is None:
                print("Сохранённого лимита нет. Сейчас он спрашивается перед ответом.\n", flush=True)
            else:
                print(f"Сохранённый лимит: {saved_char_limit} символов.\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        if argument == "off":
            print("Сохранённый лимит отключён. Теперь лимит будет спрашиваться перед каждым ответом.\n", flush=True)
            return True, None, True, None
        if argument == "ask":
            print("Ок, на следующем запросе снова спрошу лимит и предложу сохранить его.\n", flush=True)
            return True, None, False, None
        if argument.isdigit():
            value = int(argument)
            if 1 <= value <= 10000:
                print(f"Сохранил лимит {value} символов для этой сессии.\n", flush=True)
                return True, value, True, None
        print("Использование: /limit show | /limit <число> | /limit off | /limit ask\n", flush=True)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/deepstatus"):
        await open_terminal_deepstatus_menu(terminal_owner_key)
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/deepcancel"):
        parts = stripped.split(maxsplit=1)
        requested_job_id = parts[1].strip() if len(parts) > 1 else ""
        job = find_long_think_job_for_owner(terminal_owner_key, requested_job_id or None)
        if job is None or not is_long_think_active_status(str(job.get("status") or "")):
            print("Активный long-think job не найден.\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        job["cancel_requested"] = True
        job["error"] = "Long-think job был отменён пользователем."
        request_long_think_cancel(job)
        print(
            f"Останавливаю long-think job {job['job_id'][:8]}. "
            f"Артефакты останутся в {job['artifact_dir']}.\n",
            flush=True,
        )
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/deepplan"):
        active_job = get_active_long_think_job(terminal_owner_key)
        if active_job is not None:
            print(
                "У тебя уже есть активный long-think job.\n"
                f"ID: {active_job['job_id'][:8]}\n"
                f"Папка: {active_job['artifact_dir']}\n",
                flush=True,
            )
            return True, saved_char_limit, char_limit_save_choice_made, None

        parts = stripped.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            print(build_deepplan_usage_text("/deepplan") + "\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None

        request_text = parts[1].strip()
        indicator = TerminalWaitIndicator(label="Оцениваю задачу для deepthink")
        await indicator.start()
        try:
            plan = await create_pending_long_think_plan(
                mode="terminal",
                owner_key=terminal_owner_key,
                request_text=request_text,
            )
        except Exception as exc:
            await indicator.stop()
            print(f"Не удалось собрать deepplan: {exc}\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        await indicator.stop()

        append_jsonl(
            {
                "timestamp": iso_now(),
                "event": "long_think_planned",
                "mode": "terminal",
                "owner_key": terminal_owner_key,
                "plan_id": plan["plan_id"],
                "source": plan.get("source"),
                "recommended_duration_seconds": plan.get("recommended_duration_seconds"),
                "request_text": request_text,
            }
        )
        print(build_long_think_plan_text(plan, terminal_mode=True) + "\n", flush=True)

        confirmed = await ask_terminal_yes_no(
            (
                "Запустить deepthink на "
                f"{format_clock_duration(plan['recommended_duration_seconds'])}? [д/н]: "
            ),
            default=False,
        )
        if not confirmed:
            discard_pending_long_think_plan(str(plan["plan_id"]), owner_key=terminal_owner_key)
            print("Ок, план посчитал, но запускать deepthink не стал.\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None

        active_job = get_active_long_think_job(terminal_owner_key)
        if active_job is not None:
            discard_pending_long_think_plan(str(plan["plan_id"]), owner_key=terminal_owner_key)
            print(
                "Пока ты думал, уже появился активный long-think job.\n"
                f"ID: {active_job['job_id'][:8]}\n"
                f"Папка: {active_job['artifact_dir']}\n",
                flush=True,
            )
            return True, saved_char_limit, char_limit_save_choice_made, None

        plan = pop_pending_long_think_plan(str(plan["plan_id"]), owner_key=terminal_owner_key) or plan
        job = start_long_think_from_plan(plan)
        append_long_think_started_event(
            job=job,
            request_text=request_text,
            source="plan",
            plan_id=str(plan.get("plan_id") or ""),
        )
        print(
            build_long_think_started_text(
                job,
                intro_line="Запускаю long-think по рассчитанному плану.",
            )
            + "\nЕсли отвалишься по SSH, long-think всё равно должен жить дальше отдельно от сессии.\n",
            flush=True,
        )
        return True, saved_char_limit, char_limit_save_choice_made, None

    if stripped.startswith("/deepthink"):
        active_job = get_active_long_think_job(terminal_owner_key)
        if active_job is not None:
            print(
                "У тебя уже есть активный long-think job.\n"
                f"ID: {active_job['job_id'][:8]}\n"
                f"Папка: {active_job['artifact_dir']}\n",
                flush=True,
            )
            return True, saved_char_limit, char_limit_save_choice_made, None

        parts = stripped.split(maxsplit=2)
        if len(parts) < 3:
            print(build_deepthink_usage_text("/deepthink") + "\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None
        try:
            duration_seconds = parse_duration_spec(parts[1])
        except ValueError as exc:
            print(f"{exc}\n\n{build_deepthink_usage_text('/deepthink')}\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None

        request_text = parts[2].strip()
        if not request_text:
            print(build_deepthink_usage_text("/deepthink") + "\n", flush=True)
            return True, saved_char_limit, char_limit_save_choice_made, None

        job = start_long_think_job(
            mode="terminal",
            owner_key=terminal_owner_key,
            request_text=request_text,
            duration_seconds=duration_seconds,
        )
        append_long_think_started_event(
            job=job,
            request_text=request_text,
            source="manual",
        )
        print(
            build_long_think_started_text(job)
            + "\n"
            "Если отвалишься по SSH, long-think всё равно должен жить дальше отдельно от сессии.\n",
            flush=True,
        )
        return True, saved_char_limit, char_limit_save_choice_made, None

    return False, saved_char_limit, char_limit_save_choice_made, None


async def terminal_worker_main(session_number: int | None = None) -> None:
    global model_lock
    global SHOW_MODEL_RAW

    ensure_stdout_utf8()
    SHOW_MODEL_RAW = False
    setup_logging(console_enabled=False)
    bootstrap_from_interactions(INTERACTIONS_LOG_PATH)
    mark_interrupted_long_think_jobs()
    ensure_feature_roots()
    load_background_tasks_from_disk()
    validate_runtime_config()

    model_lock = asyncio.Lock()
    ensure_background_task_worker_running()
    await ensure_llama_server_running()
    logger.info("Терминальный режим запущен.")

    terminal_session, created_session, history_cleared = open_terminal_session(session_number)
    dialog_key = get_terminal_session_dialog_key(int(terminal_session["session_number"]))
    terminal_owner_key = get_long_think_owner_key(
        "terminal",
        terminal_session_number=int(terminal_session["session_number"]),
    )
    restore_terminal_session_runtime(terminal_session, dialog_key)
    resume_text = build_terminal_long_think_resume_text(
        terminal_owner_key,
        str(terminal_session.get("previous_opened_at") or ""),
    )
    saved_char_limit = clamp_terminal_session_char_limit(terminal_session.get("saved_char_limit"))
    char_limit_save_choice_made = bool(terminal_session.get("char_limit_save_choice_made"))
    print(
        build_terminal_session_welcome_text(
            terminal_session,
            created=created_session,
            history_cleared=history_cleared,
            resume_text=resume_text,
        ),
        flush=True,
    )

    try:
        while True:
            try:
                user_text = (
                    await async_terminal_input(render_terminal_prompt(saved_char_limit))
                ).strip()
            except EOFError:
                print("", flush=True)
                break

            if not user_text:
                continue
            if user_text.lower() in {"/exit", "exit", "quit"}:
                break
            if should_reject_terminal_input(user_text):
                print(
                    "\nКодировка терминала опять поехала и превратила запрос в мусор. "
                    "Повтори ввод, не отправляю эту кашу в модель.\n",
                    flush=True,
                )
                continue

            command_handled, saved_char_limit, char_limit_save_choice_made, replay_user_text = await handle_terminal_command(
                user_text,
                terminal_session,
                dialog_key,
                terminal_owner_key,
                saved_char_limit,
                char_limit_save_choice_made,
            )
            if replay_user_text is not None:
                user_text = replay_user_text
            elif command_handled:
                persist_terminal_session_runtime(
                    terminal_session,
                    dialog_key,
                    saved_char_limit,
                    char_limit_save_choice_made,
                )
                continue

            if saved_char_limit is None:
                char_limit = await read_terminal_char_limit()
                if not char_limit_save_choice_made:
                    char_limit_save_choice_made = True
                    if await ask_to_save_terminal_char_limit(char_limit):
                        saved_char_limit = char_limit
                        print(
                            f"Ок, запомнил лимит {saved_char_limit} символов для этой сессии.\n",
                            flush=True,
                        )
                    else:
                        print(
                            "Ок, тогда буду спрашивать лимит перед каждым запросом.\n",
                            flush=True,
                        )
            else:
                char_limit = saved_char_limit
                print(
                    f"Использую сохранённый лимит: {char_limit} символов.\n",
                    flush=True,
                )

            persist_terminal_session_runtime(
                terminal_session,
                dialog_key,
                saved_char_limit,
                char_limit_save_choice_made,
            )

            request_max_tokens = estimate_max_tokens_for_dialog_char_limit(
                dialog_key,
                user_text,
                char_limit,
            )
            brief_mode = should_answer_briefly_for_dialog(dialog_key, user_text)
            request_id = uuid.uuid4().hex

            logger.info(
                "Терминальный запрос: request_id=%s mode=%s max_tokens=%s char_limit=%s text=%r",
                request_id,
                "brief" if brief_mode else "detailed",
                request_max_tokens,
                char_limit,
                user_text,
            )
            print(f"\n[TERMINAL USER] {user_text}\n", flush=True)

            try:
                wait_indicator = TerminalWaitIndicator("Ожидание ответа")
                await wait_indicator.start()
                try:
                    full_reply, raw_reply, _, retried_with_clean_history = await collect_terminal_reply(
                        dialog_key,
                        user_text,
                        request_max_tokens,
                        brief_mode=brief_mode,
                        label=f"terminal:{request_id}",
                    )
                finally:
                    await wait_indicator.stop()

                if not full_reply.strip():
                    if USE_RAW_MODEL_REPLY and raw_reply.strip():
                        full_reply = normalize_raw_model_reply(raw_reply)
                    if not full_reply.strip():
                        salvaged_reply = extract_relaxed_visible_reply(raw_reply)
                        if salvaged_reply.strip():
                            full_reply = salvaged_reply

                if len(full_reply) > char_limit:
                    full_reply = full_reply[:char_limit].rstrip()

                if not full_reply.strip():
                    full_reply = (
                        "Не удалось выделить финальный ответ модели."
                        if raw_reply.strip()
                        else "Модель ничего не вернула."
                    )

                remember_turn(
                    dialog_key,
                    user_text,
                    sanitize_assistant_reply_for_history(full_reply, raw_reply),
                )
                remember_terminal_session_turn(
                    terminal_session,
                    dialog_key,
                    user_text,
                    saved_char_limit,
                    char_limit_save_choice_made,
                    full_reply,
                )

                if retried_with_clean_history:
                    print(
                        "Память терминального диалога пришлось сбросить после ошибки. "
                        "Ответ получил уже на чистом контексте.\n",
                        flush=True,
                    )

                print(
                    f"\n[TERMINAL BOT | {len(full_reply)} chars]\n{full_reply}\n",
                    flush=True,
                )
            except Exception as exc:
                logger.exception(
                    "Ошибка терминального запроса: request_id=%s text=%r",
                    request_id,
                    user_text,
                )
                print(
                    f"\nОшибка обработки запроса: {exc}\n"
                    "Терминальный режим не закрываю, можешь отправить следующий запрос.\n",
                    flush=True,
                )
    finally:
        persist_terminal_session_runtime(
            terminal_session,
            dialog_key,
            saved_char_limit,
            char_limit_save_choice_made,
        )
        if not has_active_detached_terminal_long_think_worker():
            stop_llama_server()


async def long_think_worker_main(job_id: str) -> None:
    global model_lock
    global SHOW_MODEL_RAW

    ensure_stdout_utf8()
    SHOW_MODEL_RAW = False
    setup_logging(console_enabled=False)
    validate_runtime_config()

    model_lock = asyncio.Lock()
    job = load_long_think_job_from_disk(job_id)
    if job is None:
        raise RuntimeError(f"Long-think job {job_id[:8]} не найден.")

    long_think_jobs[job_id] = job
    touch_long_think_job(job_id)
    job["process_id"] = os.getpid()
    persist_long_think_job(job)

    worker_task = asyncio.create_task(run_long_think_job(job_id))
    job["task"] = worker_task

    def request_stop() -> None:
        runtime_job = long_think_jobs.get(job_id)
        if runtime_job is None:
            return
        runtime_job["cancel_requested"] = True
        if not runtime_job.get("error"):
            runtime_job["error"] = "Long-think job был остановлен."
        persist_long_think_job(runtime_job)
        task = runtime_job.get("task")
        if isinstance(task, asyncio.Task) and not task.done():
            task.cancel()

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signum, request_stop)
        except (NotImplementedError, RuntimeError):
            signal.signal(signum, lambda _sig, _frame: request_stop())

    try:
        await worker_task
    finally:
        stop_llama_server()


def run_supervisor_worker(child_mode: str = "--bot-worker") -> None:
    ensure_stdout_utf8()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if child_mode not in {"--bot-worker", "--server-worker"}:
        raise RuntimeError(
            "Supervisor умеет запускать только --bot-worker или --server-worker."
        )

    script_path = Path(__file__).resolve()
    recent_failures: deque[float] = deque()
    append_supervisor_log(
        "INFO",
        f"systemd supervisor запущен. child_mode={child_mode} pid={os.getpid()}",
    )

    while True:
        started_at = time.monotonic()
        env = os.environ.copy()
        env["HEYMATE_SUPERVISED"] = "1"
        child = subprocess.Popen(
            [sys.executable, str(script_path), child_mode],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        append_supervisor_log(
            "INFO",
            f"Поднял worker pid={child.pid} mode={child_mode}",
        )
        exit_code = child.wait()
        uptime_seconds = max(0, int(time.monotonic() - started_at))

        now_monotonic = time.monotonic()
        if uptime_seconds >= SUPERVISOR_STABLE_UPTIME_SECONDS:
            recent_failures.clear()
        recent_failures.append(now_monotonic)
        while recent_failures and now_monotonic - recent_failures[0] > SUPERVISOR_RESTART_WINDOW_SECONDS:
            recent_failures.popleft()

        recent_failures_count = len(recent_failures)
        restart_delay_seconds = get_supervisor_restart_delay_seconds(recent_failures_count)
        status_text = (
            "worker завершился без ошибки, но для фонового systemd-режима это всё равно ненормально"
            if exit_code == 0
            else f"worker упал с exit_code={exit_code}"
        )
        if recent_failures_count >= SUPERVISOR_STORM_THRESHOLD:
            status_text += (
                f"; словил бурю рестартов: {recent_failures_count} за "
                f"{SUPERVISOR_RESTART_WINDOW_SECONDS}с"
            )

        append_supervisor_log(
            "WARNING" if exit_code == 0 else "ERROR",
            (
                f"{status_text}. uptime={uptime_seconds}с. "
                f"Перезапуск через {restart_delay_seconds}с."
            ),
        )
        time.sleep(restart_delay_seconds)


def run_ai_worker() -> None:
    ensure_stdout_utf8()
    setup_logging()
    bootstrap_from_interactions(INTERACTIONS_LOG_PATH)
    validate_runtime_config()
    logger.info("AI worker запущен.")
    start_llama_server()
    if LLAMA_SERVER_PROCESS is None:
        raise RuntimeError("Не удалось запустить llama-server.")
    try:
        exit_code = LLAMA_SERVER_PROCESS.wait()
        logger.info("AI worker завершён, llama-server exit_code=%s", exit_code)
    finally:
        stop_llama_server()


if __name__ == "__main__":
    try:
        mode = sys.argv[1] if len(sys.argv) > 1 else "--bot-worker"
        if mode == "--bot-worker":
            asyncio.run(bot_worker_main())
        elif mode == "--supervisor-worker":
            extra_args = sys.argv[2:]
            child_mode = extra_args[0] if extra_args else "--bot-worker"
            run_supervisor_worker(child_mode)
        elif mode == "--terminal-worker":
            extra_args = sys.argv[2:]
            session_number: int | None = None
            if extra_args:
                if len(extra_args) == 2 and extra_args[0] == "--session-number":
                    session_number = parse_positive_int(extra_args[1])
                if session_number is None:
                    raise RuntimeError(
                        "Для терминального режима используй --terminal-worker [--session-number <номер>]."
                    )
            asyncio.run(terminal_worker_main(session_number))
        elif mode == "--long-think-worker":
            extra_args = sys.argv[2:]
            if len(extra_args) != 1:
                raise RuntimeError(
                    "Для long-think worker используй --long-think-worker <job_id>."
                )
            asyncio.run(long_think_worker_main(extra_args[0]))
        elif mode == "--server-worker":
            run_ai_worker()
        else:
            raise RuntimeError(
                "Неизвестный режим запуска. Используй без аргументов, --bot-worker, "
                "--supervisor-worker, --terminal-worker, --long-think-worker или --server-worker."
            )
    except KeyboardInterrupt:
        print("\nОстановлено вручную.", flush=True)
    except TelegramUnauthorizedError:
        print(
            "\nКритическая ошибка: Telegram отклонил токен бота. "
            "Проверь BOT_TOKEN в переменных окружения.",
            flush=True,
        )
    except Exception as exc:
        print(f"\nКритическая ошибка: {exc}", flush=True)
        raise
