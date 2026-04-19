# ai-to-tgbot-port (Linux Branch)

Local Telegram bot runtime for GGUF models on Linux, built around `llama.cpp`.

This branch is the Linux-specific package. It is intended for terminal-first usage, SSH sessions, and long-running deployments on desktop Linux or server Linux.

## What This Branch Is

This branch contains a Linux-focused package for running a Telegram bot on top of a local GGUF model.

The target use cases are:

* running the bot on a Linux workstation;
* running the bot on a VPS or dedicated server;
* controlling setup over SSH;
* keeping the model local;
* using `llama.cpp` / `llama-server` instead of a cloud API;
* managing the bot through `systemd` once installation is complete.

This is not a single script drop. It includes:

* a terminal-first launcher;
* environment setup logic;
* model discovery and selection;
* `llama.cpp` discovery or download;
* the Telegram bot runtime;
* an optional local Flask dashboard for Linux administration;
* a local SQLite helper layer;
* `systemd` integration for production-style Linux usage.

## What Is Inside

This branch keeps the Linux package directly in the repository root.

Files:

* `HeyMate`
  Main Linux launcher entrypoint. This is the preferred terminal / SSH entrypoint.
  It can be used in two ways:
  * inside an already checked out Linux package;
  * as a standalone bootstrap launcher that downloads the Linux branch and then runs the installer.
  If the package is a git clone, it also tries to update the local Linux branch automatically before launching.

* `install_and_launch.sh`
  Thin compatibility wrapper that delegates to `./HeyMate`.

* `launcher_cli.py`
  Interactive Linux installer / launcher. It can:
  * check Python availability;
  * install missing Python dependencies;
  * find local GGUF models;
  * find local `llama-server`;
  * download `llama.cpp` if needed;
  * inspect Linux hardware and derive safe runtime defaults;
  * generate and update `.env`;
  * launch the bot in the foreground;
  * launch and manage the local web dashboard;
  * install, start, stop, restart, inspect, and remove a `systemd` service.

* `site_dashboard.py`
  Optional local Flask control dashboard. It is intended for Linux administration,
  long-think job visibility, and process inspection. By default it binds to
  `127.0.0.1`, not to all interfaces.

* `bot.py`
  Main Telegram bot runtime. It:
  * validates configuration;
  * starts and monitors `llama-server`;
  * communicates with the local model over HTTP;
  * serializes model access through a queue;
  * stores local runtime state;
  * exposes Telegram commands such as `/status`, `/reset`, `/license`, `/source`, and `/ineedmore`.

* `bot_control_db.py`
  SQLite helper layer used by the runtime.

* `.env.example`
  Example environment template.

* `requirements.txt`
  Python dependencies.

* `LICENSE`
  Project license.

* `.gitignore`
  Ignore rules for runtime state, models, logs, databases, and local artifacts.

## How It Works

The runtime is made of four practical layers:

1. Launcher layer
   Handles setup, Python checks, dependency installation, model selection, `llama.cpp` discovery, Linux hardware inspection, and `.env` generation.

2. Telegram layer
   Built on top of `aiogram`, receives updates and sends replies.

3. Model layer
   A local `llama-server` process serves the GGUF model through an OpenAI-compatible HTTP API.

4. Local state layer
   SQLite and local log files hold runtime state and diagnostics.

Request flow:

1. A Telegram user sends a message.
2. The bot validates state and limits.
3. The bot builds the prompt and recent dialog context.
4. The request enters a serialized model queue.
5. The runtime ensures `llama-server` is alive.
6. The bot sends an HTTP request to the local model server.
7. The streamed reply is processed.
8. The final response is sent back to Telegram.
9. Logs and local dialog state are updated.

Why Linux should prefer `systemd`:

* it survives SSH disconnects;
* it survives reboots;
* it provides restart policies;
* it gives proper service status;
* it integrates with `journalctl`;
* it is more reliable than keeping the bot alive in a random shell session.

## Installation

Requirements:

* Linux
* Python 3.11+
* a Telegram bot token from BotFather
* a GGUF model
* `llama.cpp` with `llama-server`, or allow the launcher to download it

Steps:

```bash
git clone -b linux https://github.com/AlbertGithot/ai-to-tgbot-port.git
cd ai-to-tgbot-port
chmod +x HeyMate install_and_launch.sh
./HeyMate
```

If `HeyMate` is used as a standalone launcher outside an existing package checkout, it will bootstrap the Linux branch into `./ai-to-tgbot-port` by default and then continue with the normal setup flow.

If `HeyMate` is started inside an already installed git-based Linux package, it will try to pull the latest fast-forward updates from the `linux` branch before continuing.

What the launcher does:

* checks Python;
* checks required Python packages;
* tries to find a local GGUF model;
* tries to find a local `llama-server`;
* can download a recommended model if you want;
* can download `llama.cpp` if needed;
* inspects CPU, RAM, model size, GPU, and available `llama.cpp` GPU backend support on Linux;
* derives conservative runtime values such as `N_THREADS`, `N_CTX`, `N_BATCH`, `N_GPU_LAYERS`, `MAX_TOKENS`, and startup timeout;
* creates or updates `.env`;
* can install a `systemd` service;
* can launch the bot directly.
* keeps the web dashboard on loopback by default unless you explicitly expose it.

## Configuration

The runtime reads `.env` from the project root.

Important variables:

* `BOT_TOKEN`
  Telegram bot token.

* `MODEL_PATH`
  Path to the `.gguf` model.

* `LLAMA_CPP_DIR`
  Path to the `llama.cpp` directory.

* `LLAMA_SERVER_EXE`
  Path to the `llama-server` binary.

* `SOURCE_URL`
  Repository URL shown by the bot.

* `MAX_HISTORY_MESSAGES`
  Chat memory depth. `-1` means unlimited.

* `N_CTX`
  Context size for the model runtime.

* `N_THREADS`
  CPU thread count used by `llama-server`.

* `N_BATCH`
  Batch size used by `llama-server`.

* `N_GPU_LAYERS`
  Number of layers offloaded to GPU, if a compatible Linux backend is available.

* `MAX_TOKENS`
  Main generation token budget.

* `BRIEF_MAX_TOKENS`
  Short-answer token budget.

* `LLAMA_SERVER_AUTO_RESTART`
  Enables automatic restart attempts when the runtime fails.

* `LLAMA_SERVER_MAX_RESTART_ATTEMPTS`
  Number of restart attempts per failure window.

* `LLAMA_SERVER_RESTART_DELAY_SECONDS`
  Delay between restart attempts.

* `SITE_DASHBOARD_HOST`
  Flask dashboard bind address. The safe default is `127.0.0.1`.

* `SITE_DASHBOARD_PORT`
  Flask dashboard port.

* `SITE_DASHBOARD_REFRESH_SECONDS`
  Dashboard refresh interval.

## Operational Notes

Runtime artifacts are local and should not be committed:

* `.env`
* `.launcher_state.json`
* `bot_logs/`
* `bot_control.db`
* `*.gguf`
* `llama.cpp/`
* `build/`
* `dist/`
* `__pycache__/`

Linux-specific note:

* the launcher expects an interactive terminal for setup;
* `HeyMate` can bootstrap the full Linux package by itself, but the actual setup flow still expects an interactive shell or SSH TTY;
* if the package is installed as a git clone, `HeyMate` attempts a safe auto-update before launch;
* once setup is complete, use the `systemd` options from the launcher for long-running service management;
* for server deployments, `systemd` is the intended path instead of `screen`.
* the dashboard binds to loopback by default; expose it to the network only deliberately and only behind sane firewall rules.
