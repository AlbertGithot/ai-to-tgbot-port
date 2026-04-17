import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import site_dashboard


class SiteDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        (self.project_root / "site_dashboard_templates").mkdir(parents=True, exist_ok=True)
        (self.project_root / "site_dashboard_static").mkdir(parents=True, exist_ok=True)

        source_templates = Path("site_dashboard_templates")
        source_static = Path("site_dashboard_static")
        for path in source_templates.iterdir():
            (self.project_root / "site_dashboard_templates" / path.name).write_text(
                path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        for path in source_static.iterdir():
            (self.project_root / "site_dashboard_static" / path.name).write_text(
                path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        self.state_path = self.project_root / "web_panel_runtime" / "panel_state.json"
        self.state, self.generated_code = site_dashboard.ensure_panel_state(self.state_path)
        self.app = site_dashboard.create_app(project_root=self.project_root, state_path=self.state_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_long_think_result(self) -> None:
        job_dir = self.project_root / "deep_think_jobs" / "job_demo"
        job_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": "deadbeef1234",
            "request_text": "Напиши большой отчёт по ESP32",
            "status": "completed",
            "phase": "completed",
            "progress_percent": 100,
            "planned_iterations": 4,
            "completed_iterations": 4,
            "remaining_seconds": 0,
            "elapsed_seconds": 3600,
            "updated_at": "2026-04-18T01:00:00+07:00",
            "result_path": str(job_dir / "result.json"),
            "artifact_dir": str(job_dir),
            "final_answer": "Готовый JSON-ответ от модели",
            "answer_completed_fully": True,
            "error": "",
        }
        (job_dir / "state.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        (job_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _write_task(self) -> None:
        task_dir = self.project_root / "task_queue"
        task_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": "task1234",
            "kind": "project_scan",
            "status": "running",
            "description": "Скан проекта demo",
            "progress_current": 3,
            "progress_total": 7,
            "progress_label": "Сканирую файлы",
            "updated_at": "2026-04-18T01:05:00+07:00",
            "error": "",
        }
        (task_dir / "task_task1234.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_ensure_panel_state_generates_access_code_and_secret(self) -> None:
        self.assertTrue(self.generated_code)
        self.assertTrue(self.state["session_secret"])
        self.assertTrue(self.state["access_code_hash"])
        self.assertTrue(self.state["site_enabled"])

    def test_build_overview_payload_collects_jobs_tasks_and_files(self) -> None:
        self._write_long_think_result()
        self._write_task()
        with mock.patch("site_dashboard.collect_relevant_processes", return_value=[{"pid": 100, "elapsed": "00:10", "cpu_percent": "2.0", "memory_percent": "1.0", "command": "python3 bot.py"}]):
            payload = site_dashboard.build_overview_payload(self.project_root)

        self.assertEqual(payload["stats"]["active_job_count"], 0)
        self.assertEqual(payload["stats"]["result_file_count"], 1)
        self.assertEqual(payload["stats"]["running_task_count"], 1)
        self.assertEqual(payload["stats"]["process_count"], 1)
        self.assertEqual(payload["result_files"][0]["job_id"], "deadbeef1234")
        self.assertEqual(payload["background_tasks"][0]["task_id"], "task1234")

    def test_dashboard_api_requires_login_and_returns_overview(self) -> None:
        self._write_long_think_result()
        unauthorized = self.client.get("/api/overview")
        self.assertEqual(unauthorized.status_code, 401)

        with self.client.session_transaction() as session:
            session["panel_authenticated"] = True

        with mock.patch("site_dashboard.collect_relevant_processes", return_value=[]):
            authorized = self.client.get("/api/overview")

        self.assertEqual(authorized.status_code, 200)
        payload = authorized.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["stats"]["result_file_count"], 1)

    def test_manage_toggle_changes_site_state(self) -> None:
        with self.client.session_transaction() as session:
            session["panel_authenticated"] = True

        response = self.client.post("/manage", data={"action": "toggle-site"}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        updated = site_dashboard.load_panel_state(self.state_path)
        self.assertFalse(updated["site_enabled"])


if __name__ == "__main__":
    unittest.main()
