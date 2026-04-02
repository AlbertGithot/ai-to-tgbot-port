# 🤖 Telegram Bot with Local AI

A Telegram bot powered by a local GGUF language model using `llama.cpp`.

---

## 📌 Description

This project is a Telegram bot that runs a locally hosted LLM for generating responses.
It is designed for full control over inference, customization, and uncensored behavior.

---

## ⚙️ Features

* `aiogram` Telegram bot
* Local GGUF model support
* Automatic `llama-server` startup from `llama.cpp`
* Dialog memory
* `/ineedmore` command
* Streamed raw model output in terminal

---

## 👤 Author

Copyright (C) 2026 NoNameUnderl1ner (aka Default_Netion)

---

## 📜 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

If you run this bot as a network service, you must provide access to the source code.

Set `SOURCE_URL` so users can retrieve the source via `/source`.

---

## 🧠 Model

This project uses the following model:

HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive

* Quantization: Q5_K_M (5-bit)
* Size: ~24.8 GB
* Format: GGUF

The model is licensed under the Apache License 2.0.

---

## ⚠️ Recommendations

This project is designed to run with **llama.cpp**.

Using alternatives like `llama-cpp-python` is **not recommended for inexperienced users**.
You may break the entire pipeline if you don’t understand how inference and bindings work.

The Telegram bot is built using the `aiogram` library.

---

## 📦 Requirements

* Python 3.11+
* `llama.cpp` binaries with `llama-server.exe`
* Local GGUF model

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Set environment variables before launch:

```powershell
$env:BOT_TOKEN="your_bot_token"
$env:MODEL_PATH="C:\Models\your-model.gguf"
$env:LLAMA_CPP_DIR="C:\Tools\llama.cpp\b8625"
$env:LLAMA_SERVER_EXE="C:\Tools\llama.cpp\b8625\llama-server.exe"
$env:SOURCE_URL="https://github.com/AlbertGithot/Qwen3.5-Uncensored-But-On-TG-Bot.git"
```

You can copy values from `.env.example`.

---

## 🚀 Run

```powershell
python bot.py
```

The bot will automatically start `llama-server` and communicate with it via a local OpenAI-compatible HTTP API.

---

## ⚠️ Disclaimer

`bot.py` was generated and adapted with the help of AI.
It has not been fully reviewed or manually polished, so expect possible issues or нестабильное поведение.

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

## 📦 Notes

* Model files and logs are ignored via `.gitignore`
* `bot.py` is the main entry point
* Local machine-specific files are excluded from the repository

---

## 🔗 Source Code

https://github.com/AlbertGithot/Qwen3.5-Uncensored-But-On-TG-Bot.git

