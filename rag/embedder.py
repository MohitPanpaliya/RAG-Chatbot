import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # load once and reuse (no need to reload every time)_

def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    # returns one embedding per text (shape: n x 384)
    return get_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

def embed_query(query: str) -> np.ndarray:
    #  single vector for a query string.
    return get_model().encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )