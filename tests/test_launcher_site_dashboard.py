import tempfile
import unittest
from pathlib import Path
from unittest import mock

import launcher_cli


class LauncherSiteDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_site_dashboard_settings_defaults_to_loopback(self) -> None:
        settings = launcher_cli.load_site_dashboard_settings(self.project_root)

        self.assertEqual(settings["host"], "127.0.0.1")
        self.assertEqual(settings["listen_url"], "http://127.0.0.1:5080")
        self.assertEqual(settings["url"], "http://127.0.0.1:5080")

    def test_load_site_dashboard_settings_uses_env_file(self) -> None:
        (self.project_root / launcher_cli.ENV_FILE_NAME).write_text(
            "\n".join(
                [
                    "SITE_DASHBOARD_HOST=0.0.0.0",
                    "SITE_DASHBOARD_PORT=6060",
                    "SITE_DASHBOARD_REFRESH_SECONDS=9",
                ]
            ),
            encoding="utf-8",
        )

        settings = launcher_cli.load_site_dashboard_settings(self.project_root)

        self.assertEqual(settings["host"], "0.0.0.0")
        self.assertEqual(settings["port"], 6060)
        self.assertEqual(settings["refresh_seconds"], 9)
        self.assertEqual(settings["listen_url"], "http://0.0.0.0:6060")

    def test_load_site_dashboard_settings_prefers_server_ip_for_external_access(self) -> None:
        (self.project_root / launcher_cli.ENV_FILE_NAME).write_text(
            "SITE_DASHBOARD_HOST=0.0.0.0\nSITE_DASHBOARD_PORT=5080\n",
            encoding="utf-8",
        )

        with mock.patch.dict(
            "os.environ",
            {"SSH_CONNECTION": "203.0.113.10 51422 45.93.200.129 22"},
            clear=False,
        ), mock.patch(
            "launcher_cli.run_capture_command",
            return_value=None,
        ), mock.patch(
            "launcher_cli.socket.getaddrinfo",
            return_value=[],
        ):
            settings = launcher_cli.load_site_dashboard_settings(self.project_root)

        self.assertEqual(settings["listen_url"], "http://0.0.0.0:5080")
        self.assertEqual(settings["url"], "http://45.93.200.129:5080")
        self.assertIn("http://45.93.200.129:5080", settings["access_urls"])

    def test_extract_site_dashboard_access_code_reads_first_launch_log(self) -> None:
        log_text = (
            "[site_dashboard] Первый запуск. Код доступа создан: test-code-123\n"
            "[site_dashboard] Состояние панели лежит тут: /tmp/panel_state.json"
        )

        self.assertEqual(
            launcher_cli.extract_site_dashboard_access_code(log_text),
            "test-code-123",
        )

    def test_get_site_dashboard_status_recovers_running_process(self) -> None:
        with mock.patch(
            "launcher_cli.find_running_site_dashboard_process",
            return_value={
                "pid": 4321,
                "command": f"python3 {self.project_root / launcher_cli.SITE_DASHBOARD_ENTRYPOINT}",
            },
        ):
            status = launcher_cli.get_site_dashboard_status(self.project_root)

        self.assertTrue(status["running"])
        self.assertTrue(status["recovered"])
        self.assertEqual(status["pid"], 4321)
        self.assertTrue(launcher_cli.site_dashboard_process_state_path(self.project_root).is_file())

    def test_get_site_dashboard_status_cleans_stale_process_file(self) -> None:
        process_state_path = launcher_cli.site_dashboard_process_state_path(self.project_root)
        process_state_path.parent.mkdir(parents=True, exist_ok=True)
        launcher_cli.save_json(process_state_path, {"pid": 999999})

        with mock.patch("launcher_cli.read_process_command", return_value=""), mock.patch(
            "launcher_cli.find_running_site_dashboard_process",
            return_value=None,
        ):
            status = launcher_cli.get_site_dashboard_status(self.project_root)

        self.assertFalse(status["running"])
        self.assertFalse(process_state_path.exists())

    def test_build_site_dashboard_systemd_service_text_contains_exec_and_env(self) -> None:
        (self.project_root / launcher_cli.ENV_FILE_NAME).write_text(
            "\n".join(
                [
                    "SITE_DASHBOARD_HOST=127.0.0.1",
                    "SITE_DASHBOARD_PORT=6061",
                    "SITE_DASHBOARD_REFRESH_SECONDS=11",
                ]
            ),
            encoding="utf-8",
        )

        with mock.patch("launcher_cli.python_command", return_value=["python3"]), mock.patch(
            "launcher_cli.shutil.which",
            side_effect=lambda value: "/usr/bin/python3" if value == "python3" else None,
        ):
            service_text = launcher_cli.build_site_dashboard_systemd_service_text(
                self.project_root,
                "system",
            )

        self.assertIn('ExecStart="/usr/bin/python3" "', service_text)
        self.assertIn("site_dashboard.py", service_text)
        self.assertIn("Environment=SITE_DASHBOARD_HOST=127.0.0.1", service_text)
        self.assertIn("Environment=SITE_DASHBOARD_PORT=6061", service_text)
        self.assertIn("Environment=SITE_DASHBOARD_REFRESH_SECONDS=11", service_text)
        self.assertIn("WantedBy=multi-user.target", service_text)


if __name__ == "__main__":
    unittest.main()
