"""
Run this once (or whenever data/ changes) to (re)build the vector store:
    python -m src.ingest
"""

import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from src.chunker import chunk_document

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "potens_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # fast, local, no API key needed


def build_index():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_DIR)
    # fresh start each run so re-ingesting doesn't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    filepaths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if len(filepaths) < 5:
        print(f"WARNING: found only {len(filepaths)} documents in data/. "
              f"Requirement is 5+.")

    all_ids, all_texts, all_metadatas = [], [], []
    doc_id_counter = 0

    for filepath in filepaths:
        chunks = chunk_document(filepath)
        for c in chunks:
            uid = f"{c['source_file']}::{c['chunk_index']}"
            all_ids.append(uid)
            all_texts.append(c["text"])
            all_metadatas.append({
                "source_file": c["source_file"],
                "section_title": c["section_title"],
                "chunk_index": c["chunk_index"],
                "doc_id": os.path.splitext(c["source_file"])[0],
            })
        doc_id_counter += 1
        print(f"Chunked {filepath} -> {len(chunks)} chunks")

    print(f"Embedding {len(all_texts)} chunks...")
    embeddings = model.encode(all_texts, show_progress_bar=True).tolist()

    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_metadatas,
    )

    print(f"Done. Indexed {len(all_texts)} chunks from {doc_id_counter} documents "
          f"into '{COLLECTION_NAME}' at {DB_DIR}")


if __name__ == "__main__":
    build_index()
