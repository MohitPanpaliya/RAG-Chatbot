import requests
from rag.embedder import embed_query
from rag.retriever import retrieve
from rag.cache import get_cached, set_cache, format_history_for_prompt

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based
ONLY on the context provided. If the context does not have enough information,
say "I don't have that in my knowledge base." Be concise (2-4 sentences)."""


def build_prompt(query: str, chunks, user_id: int = None) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        label = chunk.source_file.replace(".md", "").replace("_", " ").title()
        blocks.append(f"[{i}. {label}]\n{chunk.content}")
    context = "\n\n---\n\n".join(blocks)
    # add chat history if available
    history_section = ""
    if user_id is not None:
        history_text = format_history_for_prompt(user_id)
        if history_text:
            history_section = f"\n\n{history_text}\n"

    return (
        f"{SYSTEM_PROMPT}"
        f"{history_section}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def rag_query(query: str, top_k=4, model="llama3.2:latest",user_id: int = None) -> dict:  #changing top_k to 2 instead of 4

    # Skip embedding + retrieval + LLM if we've seen this query before
    cached = get_cached(query)
    if cached and user_id is None:
        return cached
    # embed query
    q_vec = embed_query(query)

    # fetch relevant chunks
    chunks = retrieve(q_vec, top_k=top_k)
    if not chunks:
        return {"answer": "No relevant info found.", "sources": []}

    # Build prompt (with history if available)
    prompt = build_prompt(query, chunks, user_id=user_id)
    # call Ollama locally (no API key needed)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=90  # can be slow on first run
    )
    response.raise_for_status()
    answer = response.json()["response"].strip()

    result = {
        "answer":  answer,
        "sources": list(dict.fromkeys(c.source_file for c in chunks)),
    }

    # store only if no history (avoid wrong context reuse)
    if user_id is None:
        set_cache(query, result)
    
    return result