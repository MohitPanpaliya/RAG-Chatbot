# rag/chunker.py
import os
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    content: str


def load_documents(knowledge_dir: str) -> dict:
    docs = {}
    for fname in os.listdir(knowledge_dir):
        if fname.endswith((".md", ".txt")):
            fpath = os.path.join(knowledge_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                docs[fname] = f.read()
    print(f"  Loaded {len(docs)} documents")
    return docs


def split_into_chunks(text: str, source: str,
                      chunk_size=400, overlap=80) -> List[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks, buffer, offset = [], "", 0

    for para in paragraphs:
        if len(buffer) + len(para) <= chunk_size:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            if buffer:
                chunks.append(Chunk(f"{source}::{offset}", source, buffer))
                buffer = buffer[-overlap:] + "\n\n" + para
            else:
                buffer = para
        offset += len(para)

    if buffer.strip():
        chunks.append(Chunk(f"{source}::{offset}", source, buffer))

    return chunks


def chunk_all_documents(knowledge_dir: str) -> List[Chunk]:
    all_chunks = []
    for fname, content in load_documents(knowledge_dir).items():
        doc_chunks = split_into_chunks(content, fname)
        all_chunks.extend(doc_chunks)
        print(f"  {fname} → {len(doc_chunks)} chunks")
    return all_chunks