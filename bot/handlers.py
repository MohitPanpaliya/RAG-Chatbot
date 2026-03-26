from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from rag.pipeline import rag_query
from rag.cache import add_to_history, clear_history, cache_stats

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    user_id = update.effective_user.id
    clear_history(user_id)
    name = update.effective_user.first_name or "there"
    await update.message.reply_text(
        f"Hi {name}! I'm a RAG bot. Use /ask to query my knowledge base.\n"
        f"Try: /ask Can I Work from Home?"
    )

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    await update.message.reply_text(
        "*Commands:*\n"
        "`/ask <question>` — search the knowledge base\n"
        "`/help` — show this message\n\n"
        "`/history` — show your recent conversation\n"
        "`/start` — reset conversation history\n\n"
        "*Example:*\n"
        "`/ask What is the leave policy?`",
        parse_mode=ParseMode.MARKDOWN,
    )

async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    user_id = update.effective_user.id
    query = " ".join(context.args).strip() if context.args else ""

        # context.args is a list of words after "/ask"
    query = " ".join(context.args).strip() if context.args else ""

    if not query:
        await update.message.reply_text(
            "Ask something like:\n`/ask Can I work from home?`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Show "typing..." while we process, this let the user feel that bot is working
    await update.message.chat.send_action("typing")

    try:
        result  = rag_query(query, user_id=user_id)
        answer  = result["answer"]
        sources = result["sources"]

        # Save this convo to history after getting the answer
        add_to_history(user_id, role="user", text=query)
        add_to_history(user_id, role="bot",  text=answer)

        src_lines = "\n".join(
            f"  • {s.replace('.md','').replace('_',' ').title()}"
            for s in sources
        )

        await update.message.reply_text(
            f"*Q: {query}*\n\n{answer}\n\n_Sources:_\n{src_lines}",
            parse_mode=ParseMode.MARKDOWN,
        )

    except Exception as e:
        await update.message.reply_text(f"Something went wrong: {str(e)}")


async def history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # shows hitory
    if not update.message:
        return
    from rag.cache import get_history
    user_id = update.effective_user.id
    history = get_history(user_id)

    if not history:
        await update.message.reply_text("No conversation history yet. Try `/ask` first!")
        return

    lines = ["*Recent conversation:*\n"]
    for msg in history:
        prefix = "You" if msg["role"] == "user" else "Bot"
        lines.append(f"*{prefix}:* {msg['text']}")

    await update.message.reply_text(
        "\n\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
    )


async def fallback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return    
    await update.message.reply_text(
        "Use /ask <question> to query me, or /help for instructions."
    )