import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bot


class LongThinkAndSessionRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.temp_dir.name)
        self.original_long_think_root = bot.LONG_THINK_ROOT
        self.original_terminal_sessions_root = bot.TERMINAL_SESSIONS_ROOT
        self.original_long_think_jobs = dict(bot.long_think_jobs)
        self.original_long_think_job_order = dict(bot.long_think_job_order)

        bot.LONG_THINK_ROOT = self.temp_root / "deep_think_jobs"
        bot.TERMINAL_SESSIONS_ROOT = self.temp_root / "terminal_sessions"
        bot.long_think_jobs.clear()
        bot.long_think_job_order.clear()

    def tearDown(self) -> None:
        bot.LONG_THINK_ROOT = self.original_long_think_root
        bot.TERMINAL_SESSIONS_ROOT = self.original_terminal_sessions_root
        bot.long_think_jobs.clear()
        bot.long_think_jobs.update(self.original_long_think_jobs)
        bot.long_think_job_order.clear()
        bot.long_think_job_order.update(self.original_long_think_job_order)
        self.temp_dir.cleanup()

    def test_normalize_terminal_session_tags_deduplicates_and_caps(self) -> None:
        tags = bot.normalize_terminal_session_tags(["python", "Python", "tls 1.3", "", "flask"])

        self.assertEqual(tags, ["python", "tls 1.3", "flask"])

    def test_execute_session_command_supports_tags(self) -> None:
        session = bot.create_terminal_session_payload(1)

        add_result = asyncio.run(bot.execute_session_command("tags add python, flask", session, "dlg:test"))
        remove_result = asyncio.run(bot.execute_session_command("tags remove python", session, "dlg:test"))

        self.assertIn("python", add_result)
        self.assertNotIn("python", session["tags"])
        self.assertIn("flask", session["tags"])
        self.assertIn("flask", remove_result)

    def test_persist_long_think_job_writes_markdown_artifacts(self) -> None:
        artifact_dir = self.temp_root / "deep_think_jobs" / "job_demo"
        job = {
            "job_id": "deadbeef1234",
            "mode": "terminal",
            "owner_key": "terminal:session:1",
            "status": "completed",
            "phase": "completed",
            "request_text": "Напиши большой отчёт по ESP32",
            "duration_seconds": 3600,
            "planned_iterations": 2,
            "created_at": "2026-04-20T00:00:00+00:00",
            "started_at": "2026-04-20T00:00:01+00:00",
            "deadline_at": "2026-04-20T01:00:01+00:00",
            "finalization_starts_at": "2026-04-20T00:45:01+00:00",
            "completed_at": "2026-04-20T00:58:01+00:00",
            "updated_at": "2026-04-20T00:58:01+00:00",
            "cancel_requested": False,
            "artifact_dir": str(artifact_dir),
            "result_path": "",
            "error": "",
            "note": "Финальный прогон завершён.",
            "chat_id": None,
            "user_id": None,
            "process_id": 0,
            "systemd_unit": "",
            "systemd_scope": "",
            "bot": None,
            "task": None,
            "final_buffer_seconds": 900,
            "work_phase_seconds": 2700,
            "template_outline": "1. Введение\n2. Архитектура\n3. Вывод",
            "template_ready_at": "2026-04-20T00:05:00+00:00",
            "iterations": [{"index": 1, "summary": "черновик", "chars": 1200}],
            "latest_draft": "Черновой текст ответа",
            "final_answer": "Финальный текст ответа",
            "continued_from_job_id": "",
            "seed_answer": "",
            "answer_completed_fully": True,
            "final_finish_reason": "stop",
            "progress_percent": 100,
            "progress_banner": "banner",
            "progress_message_text": "",
            "progress_message": None,
            "progress_task": None,
            "draft_markdown_path": "",
            "final_markdown_path": "",
            "template_markdown_path": "",
            "metrics": bot.build_long_think_metrics_payload(),
            "average_cpu_percent": 12.5,
            "average_ram_percent": 34.2,
            "average_gpu_percent": None,
            "metrics_task": None,
        }

        bot.persist_long_think_job(job, final=True)

        self.assertTrue((artifact_dir / "state.json").is_file())
        self.assertTrue((artifact_dir / "result.json").is_file())
        self.assertTrue((artifact_dir / "template_outline.md").is_file())
        self.assertTrue((artifact_dir / "current_draft.md").is_file())
        self.assertTrue((artifact_dir / "final_answer.md").is_file())
        self.assertTrue(any((artifact_dir / "snapshots").glob("*.md")))

    def test_start_continued_long_think_job_passes_seed_answer(self) -> None:
        previous_job = {
            "job_id": "cafebabe1234",
            "request_text": "Подготовь архитектурный план",
            "final_answer": "Готовый сильный черновик",
        }

        with mock.patch("bot.start_long_think_job", return_value={"job_id": "newjob"}) as start_mock:
            result = bot.start_continued_long_think_job(
                previous_job,
                mode="terminal",
                owner_key="terminal:session:7",
                duration_seconds=7200,
            )

        self.assertEqual(result["job_id"], "newjob")
        start_mock.assert_called_once()
        self.assertEqual(start_mock.call_args.kwargs["seed_answer"], "Готовый сильный черновик")
        self.assertEqual(start_mock.call_args.kwargs["continued_from_job_id"], "cafebabe1234")


if __name__ == "__main__":
    unittest.main()
