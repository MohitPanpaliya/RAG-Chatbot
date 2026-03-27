# Telegram RAG Bot

A Telegram bot that answers questions from a small local knowledge base using RAG (Retrieval-Augmented Generation). Built this as part of a DS assignment — the idea was to keep the stack as simple as possible without pulling in LangChain or similar.

---

## How it works

Documents get chunked and embedded offline (once). At query time the bot embeds the question, does a cosine similarity search against stored vectors, and feeds the top results as context to an LLM. The LLM only answers from what's in the context — so it won't hallucinate stuff that isn't in the docs.

Two separate flows:
- **Ingestion** (offline) — chunks docs, embeds them, stores in SQLite
- **Serving** (bot runtime) — embed query, retrieve chunks, call LLM, reply

---

## Stack

- Bot: `python-telegram-bot` v21
- Embeddings: `all-MiniLM-L6-v2` via sentence-transformers (384-dim, runs on CPU)
- Vector store: plain SQLite with float32 blobs — no Chroma, no FAISS needed at this scale
- LLM: `llama3.2` via Ollama — runs locally, roughly 55% CPU / 45% GPU
- Caching: in-memory LRU cache (50 entries) + per-user message history (last 3 turns)

---

## Setup

**Requirements**
- Python 3.10+
- [Ollama](https://ollama.com/download) installed
- Telegram bot token from @BotFather

```bash
git clone https://github.com/yourname/telegram-rag-bot
cd telegram-rag-bot

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your token:
```
TELEGRAM_BOT_TOKEN=your_token_here
```

Pull the model (one-time, ~2GB download):
```bash
ollama pull llama3.2
```

---

## Running locally

Two terminals needed simultaneously.

**Terminal 1 — keep Ollama running:**
```bash
ollama serve
```

**Terminal 2 — index docs, then start the bot:**
```bash
python scripts_ingest/ingest.py   # run once, or whenever docs change
python app.py
```

Should print `Bot is running.` once connected.

---

## Commands

| Command | What it does |
|---|---|
| `/start` | welcome message, resets conversation history |
| `/ask <question>` | queries the knowledge base |
| `/history` | shows your last 3 exchanges |
| `/help` | lists all commands |

Quick test after startup:
```
/ask can I work from home?
```

---

## Project structure

```
├── app.py                  entry point — starts the bot
├── requirements.txt
├── .env                    tokens (not committed)
├── knowledge_base/         source markdown documents
├── rag/
│   ├── chunker.py          splits docs into overlapping chunks
│   ├── embedder.py         wraps all-MiniLM-L6-v2
│   ├── retriever.py        sqlite store + cosine similarity search
│   ├── pipeline.py         orchestrates the full RAG flow
│   └── cache.py            query cache + per-user message history
├── bot/
│   └── handlers.py         telegram command handlers
├── scripts_ingest/
│   └── ingest.py           one-time indexing script
└── tests/
    └── test_chunker.py
```

---

## Notes and known issues

- First query after cold start is slow (~10-15s) while llama3.2 loads. After that it's faster.
- Tested on Python 3.14 — had to use `asyncio.run()` explicitly, PTB's `run_polling()` doesn't work cleanly with 3.14's event loop handling.
- Message history and query cache are in-memory only, so restarting the bot clears both. Fine for a demo, would use Redis in a real deployment.
- If Ollama isn't running when you start the bot, the bot starts fine but `/ask` will throw a 404 until you run `ollama serve`.
