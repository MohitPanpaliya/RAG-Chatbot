# scripts/ingest.py
print("Script started")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.chunker  import chunk_all_documents
from rag.embedder import embed_texts
from rag.retriever import init_db, clear_db, insert_chunks, count_chunks

def ingest(knowledge_dir="./knowledge_base", db_path="rag_store.db"):
    print("Initialising database...")
    init_db(db_path)
    clear_db(db_path)   # re-run safe: wipes old data first

    print("Chunking documents...")
    chunks = chunk_all_documents(knowledge_dir)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_texts([c.content for c in chunks])

    print("Storing in SQLite...")
    insert_chunks(
        [c.chunk_id    for c in chunks],
        [c.source_file for c in chunks],
        [c.content     for c in chunks],
        embeddings,
        db_path,
    )
    print(f"Done! {count_chunks(db_path)} chunks stored.")

if __name__ == "__main__":
    ingest()