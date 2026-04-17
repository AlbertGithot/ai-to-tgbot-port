from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "bot_control.db"


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                full_name TEXT,
                language_code TEXT,
                is_bot INTEGER DEFAULT 0,
                last_chat_id INTEGER,
                last_seen TEXT,
                blocked INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                event TEXT NOT NULL,
                chat_id INTEGER,
                user_id INTEGER,
                direction TEXT,
                text TEXT,
                created_at TEXT NOT NULL,
                extra_json TEXT,
                UNIQUE(request_id, event, direction)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_user_time
            ON messages(user_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_messages_chat_time
            ON messages(chat_id, created_at DESC);
            """
        )


def get_setting(key: str, default: str | None = None) -> str | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT value FROM settings WHERE key = ?",
            (key,),
        ).fetchone()
    if row is None:
        return default
    return str(row["value"])


def set_setting(key: str, value: Any) -> None:
    payload = "" if value is None else str(value)
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO settings(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, payload),
        )


def _normalize_user(raw_user: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw_user:
        return None
    user_id = raw_user.get("id")
    if user_id in (None, 0):
        return None
    return {
        "user_id": int(user_id),
        "username": raw_user.get("username"),
        "first_name": raw_user.get("first_name"),
        "last_name": raw_user.get("last_name"),
        "full_name": raw_user.get("full_name"),
        "language_code": raw_user.get("language_code"),
        "is_bot": 1 if raw_user.get("is_bot") else 0,
    }


def upsert_user(
    raw_user: dict[str, Any] | None,
    raw_chat: dict[str, Any] | None = None,
    *,
    seen_at: str | None = None,
) -> None:
    user = _normalize_user(raw_user)
    if user is None:
        return

    chat_id = raw_chat.get("id") if raw_chat else None
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO users(
                user_id, username, first_name, last_name, full_name,
                language_code, is_bot, last_chat_id, last_seen
            )
            VALUES(
                :user_id, :username, :first_name, :last_name, :full_name,
                :language_code, :is_bot, :last_chat_id, :last_seen
            )
            ON CONFLICT(user_id) DO UPDATE SET
                username = excluded.username,
                first_name = excluded.first_name,
                last_name = excluded.last_name,
                full_name = excluded.full_name,
                language_code = excluded.language_code,
                is_bot = excluded.is_bot,
                last_chat_id = COALESCE(excluded.last_chat_id, users.last_chat_id),
                last_seen = COALESCE(excluded.last_seen, users.last_seen)
            """,
            {
                **user,
                "last_chat_id": chat_id,
                "last_seen": seen_at,
            },
        )


def _insert_message(
    *,
    request_id: str | None,
    event: str,
    chat_id: int | None,
    user_id: int | None,
    direction: str,
    text: str | None,
    created_at: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = json.dumps(extra or {}, ensure_ascii=False)
    with _connect() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO messages(
                request_id, event, chat_id, user_id, direction, text, created_at, extra_json
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                event,
                chat_id,
                user_id,
                direction,
                text,
                created_at,
                payload,
            ),
        )


def record_event(record: dict[str, Any]) -> None:
    event = str(record.get("event") or "")
    timestamp = str(record.get("timestamp") or "")
    chat = record.get("chat") or {}
    user = record.get("user") or {}
    request_id = record.get("request_id")

    upsert_user(user, chat, seen_at=timestamp)

    if event == "user_message":
        _insert_message(
            request_id=request_id,
            event=event,
            chat_id=chat.get("id"),
            user_id=user.get("id"),
            direction="user",
            text=record.get("text"),
            created_at=timestamp,
        )
        return

    if event == "bot_response":
        user_text = record.get("user_text")
        if isinstance(user_text, str):
            _insert_message(
                request_id=request_id,
                event="user_message_from_response",
                chat_id=chat.get("id"),
                user_id=user.get("id"),
                direction="user",
                text=user_text,
                created_at=timestamp,
            )
        _insert_message(
            request_id=request_id,
            event=event,
            chat_id=chat.get("id"),
            user_id=user.get("id"),
            direction="assistant",
            text=record.get("bot_text"),
            created_at=timestamp,
        )
        return

    if event == "error":
        _insert_message(
            request_id=request_id,
            event=event,
            chat_id=chat.get("id"),
            user_id=user.get("id"),
            direction="error",
            text=record.get("error"),
            created_at=timestamp,
            extra={"user_text": record.get("user_text")},
        )
        return

    if event in {
        "dialog_reset",
        "dialog_reset_callback",
        "multi_request_response",
        "multi_request_form_updated",
        "multi_request_field_updated",
        "multi_request_started",
        "multi_request_opened",
        "user_activated",
    }:
        _insert_message(
            request_id=request_id,
            event=event,
            chat_id=(chat or {}).get("id") or record.get("chat_id"),
            user_id=(user or {}).get("id") or record.get("user_id"),
            direction="system",
            text=record.get("text") or record.get("final_text"),
            created_at=timestamp,
            extra=record,
        )


def bootstrap_from_interactions(log_path: str | Path) -> None:
    path = Path(log_path)
    if not path.is_file():
        return

    already_bootstrapped = get_setting("bootstrap_done", "0")
    if already_bootstrapped == "1":
        return

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                record_event(record)
            except Exception:
                continue

    set_setting("bootstrap_done", "1")


def get_users() -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT
                user_id,
                username,
                first_name,
                last_name,
                full_name,
                language_code,
                is_bot,
                last_chat_id,
                last_seen,
                blocked
            FROM users
            ORDER BY COALESCE(last_seen, '') DESC, user_id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_user(user_id: int) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return None if row is None else dict(row)


def set_user_blocked(user_id: int, blocked: bool) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE users SET blocked = ? WHERE user_id = ?",
            (1 if blocked else 0, user_id),
        )


def is_user_blocked(user_id: int | None) -> bool:
    if user_id is None:
        return False
    with _connect() as connection:
        row = connection.execute(
            "SELECT blocked FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return bool(row["blocked"]) if row is not None else False


def get_dialog_messages(user_id: int, limit: int = 200) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT request_id, event, chat_id, user_id, direction, text, created_at, extra_json
            FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    messages: list[dict[str, Any]] = []
    for row in reversed(rows):
        item = dict(row)
        try:
            item["extra_json"] = json.loads(item["extra_json"] or "{}")
        except json.JSONDecodeError:
            item["extra_json"] = {}
        messages.append(item)
    return messages
