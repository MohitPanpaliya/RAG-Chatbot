from collections import defaultdict, OrderedDict
# history
# Stores last few messages per user
# structure:
# {
#   user_id: [
#       {"role": "user"/"bot", "text": "..."},
#       ...
#   ]
# }
MAX_HISTORY = 3  # keep last 3 exchanges (1 exchange = user+bot)

_history: dict = defaultdict(list)


def add_to_history(user_id: int, role: str, text: str) -> None:
    # add message to that user's history
    _history[user_id].append({"role": role, "text": text})

    max_messages = MAX_HISTORY * 2
    if len(_history[user_id]) > max_messages:
        _history[user_id] = _history[user_id][-max_messages:]


def get_history(user_id: int) -> list:
    return _history[user_id].copy()


def format_history_for_prompt(user_id: int) -> str:
    # convert history into a string for LLM prompt
    history = get_history(user_id)
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for msg in history:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {msg['text']}")
    return "\n".join(lines)


def clear_history(user_id: int) -> None:
    if user_id in _history:
        del _history[user_id]

MAX_CACHE_SIZE = 50  # max number of stored queries

_cache: OrderedDict = OrderedDict()


def _normalise(query: str) -> str:
    return query.lower().strip()


def get_cached(query: str) -> dict | None:
    # return cached result if exists
    key = _normalise(query)
    if key in _cache:
        _cache.move_to_end(key)   # mark as recently used
        print(f"[Cache] HIT for '{query}'")
        return _cache[key]
    return None


def set_cache(query: str, result: dict) -> None:
    # store result in cache """
    key = _normalise(query)
    _cache[key] = result
    _cache.move_to_end(key)

    if len(_cache) > MAX_CACHE_SIZE:
        evicted = _cache.popitem(last=False)   # remove oldest 
        print(f"[Cache] Evicted oldest entry: '{evicted[0]}'")
    print(f"[Cache] STORED '{query}' ({len(_cache)}/{MAX_CACHE_SIZE} entries)")


def cache_stats() -> dict:
    # debuuginfo
    return {"size": len(_cache), "max": MAX_CACHE_SIZE, "keys": list(_cache.keys())}