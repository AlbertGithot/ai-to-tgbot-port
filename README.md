# Qwen3.5 Uncensored Telegram Bot

Telegram bot for local GGUF models on top of `llama.cpp`.

The project now includes a desktop control panel, a small SQLite database for bot activity, a direct AI request window, and a Telegram bot worker managed from one entrypoint.

## What It Does

- Runs a Telegram bot with `aiogram`
- Starts and uses local `llama-server` from `llama.cpp`
- Stores basic bot activity in SQLite
- Keeps short dialog memory
- Supports `/ineedmore` for grouped requests
- Shows recent logs in a control panel
- Lets the operator:
  - enable or disable AI
  - start or stop the Telegram worker
  - choose a `.gguf` model
  - inspect users and dialogs
  - block or unblock users
  - send direct prompts to the model

## Project Files

- `bot.py` - main entrypoint
- `bot_control_panel.py` - desktop control panel
- `bot_control_db.py` - SQLite helpers
- `bot_control.db` - runtime database, created automatically

## Requirements

- Python 3.11+
- Windows build of `llama.cpp` with `llama-server.exe`
- Local GGUF model
- Python packages from `requirements.txt`
- `tkinter` for the control panel
- `Pillow` is optional if you want Telegram avatars in the user list

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional:

```powershell
pip install pillow
```

## Configuration

Set the environment variables before launch.

Minimum useful setup:

```powershell
$env:BOT_TOKEN="your_telegram_bot_token"
$env:MODEL_PATH="C:\Models\your-model.gguf"
$env:LLAMA_CPP_DIR="C:\Tools\llama.cpp\b8625"
$env:LLAMA_SERVER_EXE="C:\Tools\llama.cpp\b8625\llama-server.exe"
$env:SOURCE_URL="https://github.com/AlbertGithot/Qwen3.5-Uncensored-But-On-TG-Bot.git"
```

Additional runtime settings are listed in `.env.example`.

Important notes:

- `bot.py` does not depend on `telegram_llama_bot_local_config.py`
- the selected model path can also be changed from the control panel
- the bot uses `llama-server` over local HTTP, not `llama-cpp-python`

## Run

Default mode opens the control panel:

```powershell
python bot.py
```

Worker modes:

```powershell
python bot.py --server-worker
python bot.py --bot-worker
```

## Control Panel

After launch, the panel has four sections:

- `Management`
  - enable or disable AI
  - start or stop the Telegram bot
  - choose a model
  - check current status
- `Database`
  - view users
  - inspect dialogs
  - block or unblock access
- `Direct AI`
  - send prompts to the local model without Telegram
- `Terminal`
  - view recent runtime and `llama-server` logs

## Telegram Side

The bot includes:

- dialog reset
- grouped requests through `/ineedmore`
- source code link through `/source`
- license notice through `/license`
- streamed generation in Telegram
- raw model output in terminal logs when enabled

## Runtime Data

Generated locally and normally not committed:

- `bot_control.db`
- `bot_logs/`
- `__pycache__/`

## Model

The repository does not ship a model.

Example setup used during development:

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
