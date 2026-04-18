import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bot
import launcher_cli


class ModelDiscoveryRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project_root = self.root / "project"
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.system_root = self.root / "system"
        self.system_root.mkdir(parents=True, exist_ok=True)
        self.external_model = self.system_root / "mnt" / "vault" / "outside-common-roots.gguf"
        self.external_model.parent.mkdir(parents=True, exist_ok=True)
        self.external_model.write_bytes(b"gguf-test")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_launcher_finds_models_in_system_wide_scan(self) -> None:
        with mock.patch("launcher_cli.iter_common_model_roots", return_value=[]), mock.patch(
            "launcher_cli.iter_system_model_roots",
            return_value=[self.system_root],
        ):
            models = launcher_cli.find_external_model_paths(self.project_root)

        self.assertIn(self.external_model.resolve(), models)

    def test_bot_finds_models_in_system_wide_scan(self) -> None:
        with mock.patch("bot.iter_common_model_roots", return_value=[]), mock.patch(
            "bot.iter_system_model_roots",
            return_value=[self.system_root],
        ):
            model = bot.find_external_model_path()
            models = bot.list_available_model_paths()

        self.assertEqual(model, self.external_model.resolve())
        self.assertIn(self.external_model.resolve(), models)


if __name__ == "__main__":
    unittest.main()
