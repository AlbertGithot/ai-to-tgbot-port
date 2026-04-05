# Qwen3.5 Uncensored Telegram Bot

Telegram bot for local GGUF models on top of `llama.cpp`.

The project now uses a batch installer and launcher instead of the old default `tkinter` control-panel flow.

## Main Files

- `install_and_launch.bat` - main installer and launcher for Windows
- `launcher_cli.py` - interactive installer and launcher logic
- `bot.py` - Telegram bot entrypoint
- `bot_control_db.py` - SQLite helpers for bot activity
- `.env.example` - runtime configuration template

## What It Does

- runs a Telegram bot with `aiogram`
- starts and uses local `llama-server` from `llama.cpp`
- keeps short dialog memory
- supports `/ineedmore`
- stores bot activity in SQLite
- can auto-read settings from `.env`
- can be installed and launched through `install_and_launch.bat`

## Requirements

- Windows
- Python 3.11+
- internet access for first-time install

Python dependencies are installed automatically by the launcher when needed.

## Recommended Start

Run:

```powershell
install_and_launch.bat
```

The launcher handles:

- dependency check
- repository installation
- model selection
- `.env` generation
- later launches and basic maintenance actions

## Direct Bot Run

If you already have a filled `.env`, you can run the bot directly:

```powershell
python bot.py
```

Advanced modes:

```powershell
python bot.py --bot-worker
python bot.py --server-worker
```

Default run without arguments now starts the bot directly. It no longer opens the old `tkinter` panel.

## Configuration

The bot reads variables from:

1. current process environment
2. local `.env` file in the project root

Minimal useful variables:

```powershell
BOT_TOKEN=your_telegram_bot_token
MODEL_PATH=.\models\model.gguf
LLAMA_CPP_DIR=.\llama.cpp
LLAMA_SERVER_EXE=.\llama.cpp\llama-server.exe
SOURCE_URL=https://github.com/AlbertGithot/Qwen3.5-Uncensored-But-On-TG-Bot.git
```

See `.env.example` for the full list.

## Notes

- `MAX_HISTORY_MESSAGES=-1` means unlimited in-memory history
- runtime files such as `.env`, logs, database files and downloaded models are ignored by git
- the repository does not ship a model inside source control

## Model Used During Development

- Model: `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`
- Format: `GGUF`
- Quantization: `Q5_K_M`

Model page:

https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive

## License

This repository is published under the GNU Affero General Public License v3.0.

If you run it as a network service, you must provide users access to the corresponding source code. Set `SOURCE_URL` so the bot can point users to the repository.

## Repository

https://github.com/AlbertGithot/Qwen3.5-Uncensored-But-On-TG-Bot.git
