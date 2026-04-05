"""Telegram bot backed by llama.cpp and a local GGUF model."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from collections import OrderedDict, deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

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
    from aiogram.exceptions import TelegramBadRequest, TelegramUnauthorizedError
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

BOT_TOKEN = env_str("BOT_TOKEN")
MODEL_PATH = env_path(
    "MODEL_PATH",
    get_setting("selected_model_path", "./models/model.gguf") or "./models/model.gguf",
)
LLAMA_CPP_DIR = env_path("LLAMA_CPP_DIR", "./llama.cpp")
LLAMA_SERVER_EXE = env_path(
    "LLAMA_SERVER_EXE",
    str(LLAMA_CPP_DIR / "llama-server.exe"),
)
SOURCE_URL = env_str(
    "SOURCE_URL",
    "https://github.com/AlbertGithot/Qwen3.5-Uncensored-But-On-TG-Bot.git",
)
LLAMA_SERVER_HOST = env_str("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = env_int("LLAMA_SERVER_PORT", 8080)
LLAMA_SERVER_BASE_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
LLAMA_SERVER_CHAT_URL = f"{LLAMA_SERVER_BASE_URL}/v1/chat/completions"
LLAMA_SERVER_HEALTH_URL = f"{LLAMA_SERVER_BASE_URL}/health"
LLAMA_SERVER_MODELS_URL = f"{LLAMA_SERVER_BASE_URL}/v1/models"
LLAMA_SERVER_START_TIMEOUT = env_int("LLAMA_SERVER_START_TIMEOUT", 180)
LLAMA_SERVER_REASONING = env_str("LLAMA_SERVER_REASONING", "off")
LLAMA_SERVER_REASONING_BUDGET = env_int("LLAMA_SERVER_REASONING_BUDGET", 0)
LLAMA_SERVER_REASONING_FORMAT = env_str("LLAMA_SERVER_REASONING_FORMAT", "deepseek")

SYSTEM_PROMPT = (
    "Ты локальный ассистент Telegram. "
    "Отвечай только на русском языке. "
    "Показывай только готовый финальный ответ пользователю. "
    "Не повторяй вопрос пользователя. "
    "Не пиши строки вроде 'User input:', 'User asks:', 'Question:' или служебные пояснения."
)

# Если модель сама не подхватывает chat template из GGUF, впиши сюда формат
# вроде 'chatml', 'llama-2', 'mistral-instruct' и т.д. Иначе оставь None.
CHAT_FORMAT: str | None = env_str("CHAT_FORMAT", "qwen") or None

MAX_HISTORY_MESSAGES = env_int("MAX_HISTORY_MESSAGES", 10)
MAX_HISTORY_ENTRY_CHARS = env_int("MAX_HISTORY_ENTRY_CHARS", 2500)
MAX_ACTIVE_DIALOGS = env_int("MAX_ACTIVE_DIALOGS", 200)
MAX_USER_TEXT_CHARS = env_int("MAX_USER_TEXT_CHARS", 6000)
MAX_MULTI_REQUEST_TEXT_CHARS = env_int("MAX_MULTI_REQUEST_TEXT_CHARS", 2000)
MAX_LOG_TEXT_CHARS = env_int("MAX_LOG_TEXT_CHARS", 6000)
MAX_MODEL_REPLY_CHARS = env_int("MAX_MODEL_REPLY_CHARS", 24000)

N_CTX = env_int("N_CTX", 32768)
N_THREADS = max(1, (os.cpu_count() or 4) - 1)
N_BATCH = env_int("N_BATCH", 512)
N_GPU_LAYERS = env_int("N_GPU_LAYERS", 0)

MAX_TOKENS = env_int("MAX_TOKENS", 3072)
BRIEF_MAX_TOKENS = env_int("BRIEF_MAX_TOKENS", 240)
TEMPERATURE = env_float("TEMPERATURE", 0.6)
TOP_P = env_float("TOP_P", 0.95)
TOP_K = env_int("TOP_K", 20)
REPEAT_PENALTY = env_float("REPEAT_PENALTY", 1.1)
STOP_STRINGS = ["<|eot_id|>", "<|end|>", "</s>"]

TELEGRAM_SEGMENT_LIMIT = env_int("TELEGRAM_SEGMENT_LIMIT", 3800)
SHOW_MODEL_RAW = env_bool("SHOW_MODEL_RAW", True)
USE_RAW_MODEL_REPLY = env_bool("USE_RAW_MODEL_REPLY", True)
ENABLE_REPAIR_PASS = env_bool("ENABLE_REPAIR_PASS", True)
AI_ENABLED = env_bool("AI_ENABLED", True)
THINKING_PLACEHOLDER_TEXT = "Подожди, я думаю...."
MAX_TRACKED_BOT_MESSAGES = env_int("MAX_TRACKED_BOT_MESSAGES", 80)
PROJECT_ROOT = Path(__file__).resolve().parent
LICENSE_FILE_PATH = PROJECT_ROOT / "LICENSE"
RESET_DIALOG_CALLBACK = "dialog:reset_clear"
SOURCE_CODE_CALLBACK = "show_source_code"
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

BRIEF_REPLY_STYLE_PROMPT = (
    "Это простой вопрос. "
    "Ответь коротко, естественно и сразу по делу. "
    "Только финальный ответ."
)
BRIEF_REPLY_MAX_CHARS = 320
PROMPT_SNAPSHOT_SIGNATURE = "v6-env-bat-launcher"

LOG_DIR = Path("bot_logs")
RUNTIME_LOG_PATH = LOG_DIR / "runtime.log"
INTERACTIONS_LOG_PATH = LOG_DIR / "interactions.jsonl"
LICENSE_CALLBACK = "show_license"
logger = logging.getLogger("telegram_llama_bot")
router = Router()

dialog_histories: dict[str, deque[dict[str, str]]] = {}
chat_locks: dict[int, asyncio.Lock] = {}
multi_request_sessions: dict[str, dict[str, Any]] = {}
bot_response_message_ids: dict[str, deque[int]] = {}
dialog_prompt_snapshots: dict[str, str] = {}
dialog_activity_order: OrderedDict[str, None] = OrderedDict()
model_lock: asyncio.Lock | None = None
LLAMA_SERVER_PROCESS: subprocess.Popen[str] | None = None
LLAMA_SERVER_LOG_HANDLE: Any | None = None


def iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def ensure_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
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

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


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


def append_reply_chunk(current: str, chunk: str) -> tuple[str, bool]:
    updated = current + chunk
    if len(updated) <= MAX_MODEL_REPLY_CHARS:
        return updated, False
    return updated[:MAX_MODEL_REPLY_CHARS], True


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
        "https://github.com/AlbertGithot/Qwen3.5-Uncensored-But-On-TG-Bot.git"
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


def render_multi_request_status(topic: str, queries: list[str], statuses: list[str]) -> str:
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

    lines = [THINKING_PLACEHOLDER_TEXT, "", "/ineedmore: статус шаблонов", ""]
    if topic.strip():
        lines.append(f"Тема: {topic}")
        lines.append("")

    for index, (query, status) in enumerate(zip(queries, statuses), start=1):
        label = shorten_status_label(query) or f"Запрос {index}"
        lines.append(f"{index}. {label} - {render_status_label(index - 1, status)}")

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


def validate_config() -> None:
    if not BOT_TOKEN or "PASTE_TELEGRAM_BOT_TOKEN_HERE" in BOT_TOKEN:
        raise RuntimeError("Вставь токен бота в BOT_TOKEN.")

    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Не найден файл модели: {MODEL_PATH}")

    if not LLAMA_SERVER_EXE.is_file():
        raise RuntimeError(f"Не найден llama-server.exe: {LLAMA_SERVER_EXE}")


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


def get_request_max_tokens(user_text: str) -> int:
    return BRIEF_MAX_TOKENS if should_answer_briefly(user_text) else MAX_TOKENS


def get_request_history(dialog_key: str, user_text: str) -> list[dict[str, str]]:
    if should_answer_briefly(user_text):
        return []
    return list(get_dialog_history(dialog_key))


def get_brief_fallback_reply(user_text: str) -> str | None:
    return None


def build_system_message_content(
    user_text: str,
    *extra_prompts: str | None,
    base_prompt: str | None = None,
) -> str:
    prompt_parts = [base_prompt or get_system_prompt_for_request(user_text)]
    if should_answer_briefly(user_text):
        prompt_parts.append(BRIEF_REPLY_STYLE_PROMPT)
    prompt_parts.extend(
        prompt.strip() for prompt in extra_prompts if isinstance(prompt, str) and prompt.strip()
    )
    return "\n\n".join(prompt_parts)


def get_system_prompt_for_request(user_text: str) -> str:
    if should_answer_briefly(user_text):
        return BRIEF_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def build_messages(dialog_key: str, user_text: str) -> list[dict[str, str]]:
    ensure_prompt_snapshot(dialog_key)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_message_content(user_text),
        }
    ]
    messages.extend(get_request_history(dialog_key, user_text))
    messages.append({"role": "user", "content": user_text})
    return messages


def build_repair_messages(dialog_key: str, user_text: str) -> list[dict[str, str]]:
    ensure_prompt_snapshot(dialog_key)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_message_content(user_text, REPAIR_SYSTEM_PROMPT),
        },
    ]
    messages.extend(get_request_history(dialog_key, user_text))
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


def build_llama_server_command() -> list[str]:
    return [
        str(LLAMA_SERVER_EXE),
        "--model",
        str(MODEL_PATH),
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
    command = build_llama_server_command()
    logger.info("Запускаю llama-server: %s", " ".join(command))
    LLAMA_SERVER_PROCESS = subprocess.Popen(
        command,
        cwd=str(LLAMA_CPP_DIR),
        stdout=LLAMA_SERVER_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        text=True,
    )


async def ensure_llama_server_running() -> None:
    if await is_llama_server_ready():
        logger.info("Использую уже запущенный llama-server: %s", LLAMA_SERVER_BASE_URL)
        return

    logger.info("Запуск локального llama-server для модели: %s", MODEL_PATH)
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

    if SHOW_MODEL_RAW:
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

                    if SHOW_MODEL_RAW:
                        print(delta, end="", flush=True)
                    yield {"type": "token", "text": delta}
    finally:
        if SHOW_MODEL_RAW:
            print("", flush=True)

    yield {"type": "done", "finish_reason": finish_reason}


async def collect_model_reply_unlocked(
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    brief_mode: bool = False,
) -> tuple[str, str, str | None]:
    raw_reply = ""
    finish_reason: str | None = None

    async for event in stream_model_reply(messages, max_tokens):
        if event["type"] == "done":
            finish_reason = event.get("finish_reason")
            continue
        raw_reply, truncated = append_reply_chunk(raw_reply, event["text"])
        if truncated:
            logger.warning("Ответ модели обрезан по MAX_MODEL_REPLY_CHARS=%s", MAX_MODEL_REPLY_CHARS)
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
) -> tuple[str, str, str | None]:
    if model_lock is None:
        raise RuntimeError("Лок модели не инициализирован.")

    async with model_lock:
        return await collect_model_reply_unlocked(
            messages, max_tokens, brief_mode=brief_mode
        )


class StreamingTelegramEditor:
    def __init__(self, message: Message, dialog_key: str) -> None:
        self.message = message
        self.dialog_key = dialog_key
        self.current_message: Message | None = None
        self.rendered_segment = ""

    async def start(self) -> None:
        self.current_message = await self.message.answer(THINKING_PLACEHOLDER_TEXT)
        self.rendered_segment = THINKING_PLACEHOLDER_TEXT
        track_bot_message(self.dialog_key, self.current_message)

    async def flush(self, full_text: str, final: bool = False) -> None:
        if self.current_message is None:
            await self.start()

        assert self.current_message is not None

        text = full_text.strip() if final else full_text
        if not text:
            text = "Модель ничего не вернула." if final else THINKING_PLACEHOLDER_TEXT

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
        "/reset - сбросить память диалога.\n"
        "/ineedmore - собрать несколько запросов в один пакет.\n"
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
    status_edit_lock = asyncio.Lock()

    async def refresh_status(extra_line: str | None = None) -> None:
        text = render_multi_request_status(topic, queries, statuses)
        if extra_line:
            text = f"{text}\n\n{extra_line}"
        async with status_edit_lock:
            await safe_edit_message(status_message, text)

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

        async with model_lock:
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
        )
        intro = intro.strip()
        if not intro:
            intro = (
                f'Собрал ответы по теме "{topic}".'
                if topic.strip()
                else "Собрал ответы по всем шаблонам."
            )

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
    brief_mode = should_answer_briefly(text)
    brief_fallback = get_brief_fallback_reply(text) if brief_mode else None
    used_brief_fallback = False
    request_max_tokens = get_request_max_tokens(text)
    reply_finish_reason: str | None = None

    async with chat_lock:
        try:
            async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
                await editor.start()
                messages = build_messages(dialog_key, text)
                logger.info(
                    "Режим ответа: request_id=%s mode=%s max_tokens=%s",
                    request_id,
                    "brief" if brief_mode else "detailed",
                    request_max_tokens,
                )

                if model_lock is None:
                    raise RuntimeError("Лок модели не инициализирован.")

                async with model_lock:
                    async for event in stream_model_reply(messages, request_max_tokens):
                        if event["type"] == "done":
                            reply_finish_reason = event.get("finish_reason")
                            continue

                        chunk = event["text"]
                        raw_reply, truncated = append_reply_chunk(raw_reply, chunk)
                        if truncated:
                            logger.warning(
                                "Ответ модели обрезан по MAX_MODEL_REPLY_CHARS=%s: request_id=%s",
                                MAX_MODEL_REPLY_CHARS,
                                request_id,
                            )
                            reply_finish_reason = reply_finish_reason or "length"
                            break

                    if USE_RAW_MODEL_REPLY:
                        full_reply = normalize_raw_model_reply(raw_reply)
                    else:
                        full_reply = extract_visible_reply(raw_reply, final=True)
                    if brief_mode and not USE_RAW_MODEL_REPLY:
                        full_reply = compress_brief_reply(full_reply)
                        if brief_fallback and (
                            not full_reply.strip()
                            or looks_like_reasoning(raw_reply)
                            or looks_like_prompt_leak(full_reply)
                            or not is_strict_final_reply_candidate(full_reply)
                        ):
                            logger.info(
                                "Использую brief-fallback после первого прохода: request_id=%s",
                                request_id,
                            )
                            full_reply = brief_fallback
                            used_brief_fallback = True

                    logger.info(
                        "Генерация завершена: request_id=%s finish_reason=%s",
                        request_id,
                        reply_finish_reason,
                    )

                    if (
                        not USE_RAW_MODEL_REPLY
                        and
                        brief_mode
                        and not used_brief_fallback
                        and not brief_fallback
                        and reply_finish_reason == "length"
                        and looks_truncated_reply(full_reply)
                    ):
                        logger.info(
                            "Перегенерация короткого ответа: request_id=%s",
                            request_id,
                        )
                        retry_raw_reply = ""
                        retry_messages = build_brief_retry_messages(text)

                        async for event in stream_model_reply(retry_messages, request_max_tokens):
                            if event["type"] == "done":
                                continue
                            retry_raw_reply, truncated = append_reply_chunk(
                                retry_raw_reply, event["text"]
                            )
                            if truncated:
                                logger.warning(
                                    "Retry-ответ обрезан по MAX_MODEL_REPLY_CHARS=%s: request_id=%s",
                                    MAX_MODEL_REPLY_CHARS,
                                    request_id,
                                )
                                break

                        retried_reply = extract_visible_reply(retry_raw_reply, final=True)
                        retried_reply = compress_brief_reply(retried_reply)
                        if retried_reply.strip():
                            raw_reply = retry_raw_reply
                            full_reply = retried_reply
                        elif brief_fallback:
                            logger.info(
                                "Использую brief-fallback после retry: request_id=%s",
                                request_id,
                            )
                            full_reply = brief_fallback
                            used_brief_fallback = True

                    if (not USE_RAW_MODEL_REPLY) and (not brief_mode) and needs_repair_pass(
                        full_reply, raw_reply, brief_mode=brief_mode
                    ):
                        logger.info(
                            "Запускаю repair-pass: request_id=%s",
                            request_id,
                        )
                        repair_raw_reply = ""
                        repair_messages = build_repair_messages(dialog_key, text)

                        async for event in stream_model_reply(repair_messages, request_max_tokens):
                            if event["type"] == "done":
                                continue
                            repair_raw_reply, truncated = append_reply_chunk(
                                repair_raw_reply, event["text"]
                            )
                            if truncated:
                                logger.warning(
                                    "Repair-ответ обрезан по MAX_MODEL_REPLY_CHARS=%s: request_id=%s",
                                    MAX_MODEL_REPLY_CHARS,
                                    request_id,
                                )
                                break

                        repaired_reply = extract_visible_reply(
                            repair_raw_reply, final=True
                        )
                        if brief_mode:
                            repaired_reply = compress_brief_reply(repaired_reply)
                            if not repaired_reply.strip() and brief_fallback:
                                logger.info(
                                    "Использую brief-fallback после repair: request_id=%s",
                                    request_id,
                                )
                                repaired_reply = brief_fallback
                                used_brief_fallback = True
                        if repaired_reply.strip():
                            raw_reply = repair_raw_reply
                            full_reply = repaired_reply

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

            remember_turn(dialog_key, text, full_reply)

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
    validate_config()

    model_lock = asyncio.Lock()

    bot = Bot(token=BOT_TOKEN)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    logger.info("Бот запущен и готов к polling.")
    try:
        await dispatcher.start_polling(
            bot,
            allowed_updates=dispatcher.resolve_used_update_types(),
        )
    finally:
        await bot.session.close()
        stop_llama_server()


def run_ai_worker() -> None:
    ensure_stdout_utf8()
    setup_logging()
    bootstrap_from_interactions(INTERACTIONS_LOG_PATH)
    validate_config()
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
        elif mode == "--server-worker":
            run_ai_worker()
        else:
            raise RuntimeError(
                "Неизвестный режим запуска. Используй без аргументов, --bot-worker или --server-worker."
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
