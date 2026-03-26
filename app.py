# app.py
import requests
import os
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()
from rag.embedder import get_model # loads the model at startup to decrease latency
print("Pre-loading embedding model...")
get_model()                    
print("Embedding model ready.")

from telegram.ext import Application, CommandHandler, MessageHandler, filters
from bot.handlers import start_handler, help_handler, ask_handler, fallback_handler, history_handler

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

def check_ollama():
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        print("Ollama is running.")
    except requests.exceptions.ConnectionError:
        print("WARNING: Ollama is not running.")
        print("Start it with: ollama serve")
        print("Bot will start but /ask will fail until Ollama is running.")

async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    check_ollama()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN not set in .env")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start",  start_handler))
    app.add_handler(CommandHandler("help",   help_handler))
    app.add_handler(CommandHandler("ask",    ask_handler))
    app.add_handler(CommandHandler("history", history_handler))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, fallback_handler
    ))

    print("Bot is running. Press Ctrl+C to stop.")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    # Keep running until Ctrl+C
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())