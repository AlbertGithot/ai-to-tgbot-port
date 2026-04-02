# 🤖 Qwen3.5 Uncensored Telegram Bot

<img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python"  alt="Python" width="40" height="40"/>&nbsp;
<img src="https://avatars.githubusercontent.com/u/33784865?s=280&v=4" title="Aiogram"  alt="Aiogram" width="40" height="40"/>&nbsp;

Telegram bot for local GGUF models on top of `llama.cpp`.

This project includes a Telegram bot, a desktop control panel, a SQLite database for activity tracking, and a direct AI interaction interface — all managed from a single entrypoint.

---

## 📌 Description

This project provides a fully local AI-powered Telegram bot system with:

* local LLM inference via `llama.cpp`
* desktop control panel for management
* built-in SQLite database
* direct interaction with the model outside Telegram

Designed for full control, experimentation, and uncensored AI usage.

---

## ⚙️ Features

* `aiogram` Telegram bot
* Automatic `llama-server` startup
* Local GGUF model support
* Desktop control panel (tkinter)
* SQLite database for bot activity
* Dialog memory
* `/ineedmore` grouped requests
* Streamed responses in Telegram
* Raw model output in terminal logs

### 🧩 Control Panel Capabilities

* Enable / disable AI
* Start / stop Telegram worker
* Select `.gguf` model
* View users and dialogs
* Block / unblock users
* Send direct prompts to the model
* View runtime and model logs

---

## 📁 Project Structure

* `bot.py` — main entrypoint
* `bot_control_panel.py` — desktop control panel
* `bot_control_db.py` — SQLite helpers
* `bot_control.db` — runtime database (auto-created)

---

## 📦 Requirements

* Python 3.11+
* Windows build of `llama.cpp` with `llama-server.exe`
* Local GGUF model
* `tkinter` (for control panel)

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional:

```powershell
pip install pillow
```

---

## ⚙️ Configuration

Set environment variables before launch:

```powershell
$env:BOT_TOKEN="your_telegram_bot_token"
$env:MODEL_PATH="C:\Models\your-model.gguf"
$env:LLAMA_CPP_DIR="C:\Tools\llama.cpp\b8625"
$env:LLAMA_SERVER_EXE="C:\Tools\llama.cpp\b8625\llama-server.exe"
$env:SOURCE_URL="https://github.com/AlbertGithot/Qwen3.5-Uncensored-But-On-TG-Bot.git"
```

Additional settings are available in `.env.example`.

### 🔹 Notes

* `bot.py` does NOT depend on legacy config files
* Model path can be changed from the control panel
* Uses `llama-server` over HTTP (not `llama-cpp-python`)

---

## 🚀 Run

Default mode (control panel):

```powershell
python bot.py
```

Worker modes:

```powershell
python bot.py --server-worker
python bot.py --bot-worker
```

---

## 🖥️ Control Panel

Includes four main sections:

### Management

* Enable / disable AI
* Start / stop bot
* Select model
* Check status

### Database

* View users
* Inspect dialogs
* Block / unblock users

### Direct AI

* Send prompts directly to the model

### Terminal

* View runtime logs
* View `llama-server` output

---

## 🤖 Telegram Features

* Dialog reset
* `/ineedmore` grouped requests
* `/source` — source code link
* `/license` — license notice
* Streamed responses
* Optional raw output logging

---

## 📂 Runtime Data

Generated locally and usually not committed:

* `bot_control.db`
* `bot_logs/`
* `__pycache__/`

---

## 🧠 Model

Example configuration used during development:

**Model:**
HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive

* Format: GGUF
* Quantization: Q5_K_M
* Size: ~24.8 GB

Licensed under the Apache License 2.0.

---

## ⚠️ Recommendations

This project is designed to run with **llama.cpp**.

Using alternatives like `llama-cpp-python` is **not recommended for inexperienced users**.
It may break the inference pipeline if used incorrectly.

---

## ⚠️ Disclaimer

`bot.py` and parts of the project were generated and adapted with the help of AI.
The code is not fully reviewed and may contain issues.

---

## 💸 Support the Author

If you like this project, you can support the author:

* **USDT (TON):**
  UQDCmDuVnhwvZ6EW6qv0J_nV1_7U9-IPCKYo3L279JId0qbU

* **TON:**
  UQBN03z--f0LsRcODzNLukobgrGIt6lo-MAn-FX7l1KBKZZI

* **BTC:**
  bc1q4tpzte0efcn9xsf67dcuzw6e749dahnm0j7f8m

---

## 📬 Contact

Telegram: @Default_Netion

---

## 📜 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

If you run this software as a network service, you must provide access to the source code.

---

## 🔗 Repository

https://github.com/AlbertGithot/Qwen3.5-Uncensored-But-On-TG-Bot.git

