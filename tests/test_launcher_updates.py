import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import launcher_cli


@unittest.skipUnless(shutil.which("git"), "git is required for launcher update tests")
class LauncherUpdateRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.remote_repo = self.root / "remote.git"
        self.source_repo = self.root / "source"
        self.local_repo = self.root / "local"

        self.run_git(self.root, "init", "--bare", str(self.remote_repo))
        self.run_git(self.root, "clone", str(self.remote_repo), str(self.source_repo))
        self.configure_repo(self.source_repo)

        (self.source_repo / "README.md").write_text("old history\n", encoding="utf-8")
        self.run_git(self.source_repo, "add", "README.md")
        self.run_git(self.source_repo, "commit", "-m", "initial linux commit")
        self.run_git(self.source_repo, "branch", "-M", "linux")
        self.run_git(self.source_repo, "push", "-u", "origin", "linux")

        self.run_git(self.root, "clone", "--branch", "linux", str(self.remote_repo), str(self.local_repo))
        self.configure_repo(self.local_repo)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    def configure_repo(self, repo_path: Path) -> None:
        self.run_git(repo_path, "config", "user.name", "Launcher Test")
        self.run_git(repo_path, "config", "user.email", "launcher-test@example.com")

    def rewrite_remote_branch_history(self) -> None:
        self.run_git(self.source_repo, "checkout", "--orphan", "rewrite-linux")
        self.run_git(self.source_repo, "rm", "-rf", "--ignore-unmatch", ".")
        (self.source_repo / "README.md").write_text("rewritten history\n", encoding="utf-8")
        (self.source_repo / "VERSION.txt").write_text("34ee80d\n", encoding="utf-8")
        self.run_git(self.source_repo, "add", "README.md", "VERSION.txt")
        self.run_git(self.source_repo, "commit", "-m", "rewrite linux history")
        self.run_git(self.source_repo, "branch", "-M", "linux")
        self.run_git(self.source_repo, "push", "--force", "origin", "linux")

    def test_install_project_update_realigns_after_force_pushed_branch(self) -> None:
        self.rewrite_remote_branch_history()

        updated = launcher_cli.install_project_update(
            self.local_repo,
            {"branch": "linux"},
        )

        self.assertTrue(updated)
        self.assertEqual(
            launcher_cli.current_git_head_sha(self.local_repo),
            launcher_cli.current_git_head_sha(self.local_repo, "origin/linux"),
        )
        self.assertEqual(
            (self.local_repo / "README.md").read_text(encoding="utf-8"),
            "rewritten history\n",
        )
        self.assertEqual(
            (self.local_repo / "VERSION.txt").read_text(encoding="utf-8"),
            "34ee80d\n",
        )


if __name__ == "__main__":
    unittest.main()
