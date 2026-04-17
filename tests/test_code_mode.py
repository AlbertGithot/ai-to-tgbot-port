import asyncio
import tempfile
import unittest
from pathlib import Path

import bot


class CodeModeRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_root = Path(self.temp_dir.name)

        self.original_project_context_root = bot.PROJECT_CONTEXT_ROOT
        self.original_knowledge_base_root = bot.KNOWLEDGE_BASE_ROOT
        self.original_task_queue_root = bot.TASK_QUEUE_ROOT
        self.original_dialog_runtime_settings = {
            key: dict(value) for key, value in bot.dialog_runtime_settings.items()
        }
        self.original_dialog_histories = dict(bot.dialog_histories)
        self.original_dialog_prompt_snapshots = dict(bot.dialog_prompt_snapshots)

        bot.PROJECT_CONTEXT_ROOT = temp_root / "project_contexts"
        bot.KNOWLEDGE_BASE_ROOT = temp_root / "local_kb"
        bot.TASK_QUEUE_ROOT = temp_root / "task_queue"
        bot.dialog_runtime_settings.clear()
        bot.dialog_histories.clear()
        bot.dialog_prompt_snapshots.clear()
        bot.ensure_feature_roots()

    def tearDown(self) -> None:
        bot.PROJECT_CONTEXT_ROOT = self.original_project_context_root
        bot.KNOWLEDGE_BASE_ROOT = self.original_knowledge_base_root
        bot.TASK_QUEUE_ROOT = self.original_task_queue_root

        bot.dialog_runtime_settings.clear()
        bot.dialog_runtime_settings.update(self.original_dialog_runtime_settings)

        bot.dialog_histories.clear()
        bot.dialog_histories.update(self.original_dialog_histories)

        bot.dialog_prompt_snapshots.clear()
        bot.dialog_prompt_snapshots.update(self.original_dialog_prompt_snapshots)

        self.temp_dir.cleanup()

    def _write_project_record(self) -> dict:
        project = {
            "project_id": "proj1234abcd",
            "title": "demo-service",
            "project_path": "/tmp/demo-service",
            "created_at": "2026-04-18T00:00:00+00:00",
            "updated_at": "2026-04-18T00:00:00+00:00",
            "file_count": 3,
            "total_chars": 840,
            "files": [
                {
                    "path": "src/app.py",
                    "absolute_path": "/tmp/demo-service/src/app.py",
                    "chars": 420,
                    "chunk_count": 2,
                    "preview": "FastAPI app startup, middleware wiring and route registration.",
                },
                {
                    "path": "src/auth.py",
                    "absolute_path": "/tmp/demo-service/src/auth.py",
                    "chars": 280,
                    "chunk_count": 2,
                    "preview": "JWT validation and user extraction helpers.",
                },
                {
                    "path": "tests/test_auth.py",
                    "absolute_path": "/tmp/demo-service/tests/test_auth.py",
                    "chars": 140,
                    "chunk_count": 1,
                    "preview": "Regression tests for auth failures and invalid tokens.",
                },
            ],
            "chunks": [
                {
                    "chunk_id": "1-1",
                    "path": "src/app.py",
                    "text": "from fastapi import FastAPI\nfrom .auth import validate_token\n\ndef create_app():\n    app = FastAPI()\n    return app\n",
                },
                {
                    "chunk_id": "2-1",
                    "path": "src/auth.py",
                    "text": "def validate_token(token: str) -> dict:\n    if not token:\n        raise ValueError('empty token')\n    return {'sub': 'demo'}\n",
                },
            ],
        }
        bot.atomic_write_json(bot.project_record_file_path(project["project_id"]), project)
        return project

    def test_analyze_code_request_detects_fix_patch_and_tests(self) -> None:
        analysis = bot.analyze_code_request(
            "Исправь баг в auth middleware, покажи patch diff и добавь регрессионные тесты."
        )

        self.assertTrue(analysis["wants_fix"])
        self.assertTrue(analysis["wants_patch"])
        self.assertTrue(analysis["wants_tests"])
        self.assertIn(analysis["complexity"], {"medium", "large"})

    def test_build_messages_in_code_mode_includes_contract_and_project_overview(self) -> None:
        project = self._write_project_record()
        dialog_key = "dlg:code"
        bot.set_dialog_response_mode(dialog_key, "code")
        bot.set_dialog_active_project(dialog_key, project["project_id"])

        messages = bot.build_messages(
            dialog_key,
            "Исправь баг авторизации, покажи patch diff и добавь регрессионные тесты.",
        )

        system_message = messages[0]["content"]
        self.assertIn("unified diff", system_message)
        self.assertIn("регрессионные тесты", system_message)
        self.assertIn("Обзор активного проекта", system_message)
        self.assertIn("src/app.py", system_message)
        self.assertIn("Есть активный проект", system_message)

    def test_build_local_context_prompt_in_code_mode_keeps_project_overview_without_search_hits(self) -> None:
        project = self._write_project_record()
        dialog_key = "dlg:overview"
        bot.set_dialog_response_mode(dialog_key, "code")
        bot.set_dialog_active_project(dialog_key, project["project_id"])

        context_prompt = bot.build_local_context_prompt(
            dialog_key,
            "Собери rate limiter и троттлинг для внешнего API шлюза.",
        )

        self.assertIn("Обзор активного проекта", context_prompt)
        self.assertIn("src/app.py", context_prompt)
        self.assertIn("стек по файлам", context_prompt)

    def test_code_mode_token_budget_scales_with_complexity(self) -> None:
        dialog_key = "dlg:tokens"
        bot.set_dialog_response_mode(dialog_key, "code")

        small_budget = bot.get_request_max_tokens_for_dialog(dialog_key, "Объясни dict в Python.")
        large_budget = bot.get_request_max_tokens_for_dialog(
            dialog_key,
            "Напиши большой backend-сервис с REST API, очередью задач, миграциями БД, Docker Compose и регрессионными тестами.",
        )

        self.assertGreaterEqual(large_budget, small_budget)
        self.assertLessEqual(large_budget, bot.MAX_TOKENS)
        self.assertGreaterEqual(small_budget, bot.BRIEF_MAX_TOKENS)

    def test_execute_mode_command_reports_stronger_code_profile(self) -> None:
        result = asyncio.run(bot.execute_mode_command("code", "dlg:command"))

        self.assertIn("Режим переключил на `code`.", result)
        self.assertIn("план, изменения по файлам, тесты и команды проверки", result)


if __name__ == "__main__":
    unittest.main()
