import os
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "potens_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_client = None
_collection = None


def _get_resources():
    global _model, _client, _collection
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_DIR)
        _collection = _client.get_collection(COLLECTION_NAME)
    return _model, _collection


def retrieve(query: str, top_k: int = 4):
    """
    Returns a list of dicts: {text, source_file, section_title, chunk_index,
    doc_id, similarity}. Similarity is cosine similarity in [0, 1]
    (Chroma returns distance; we convert since our embeddings are normalized-ish).
    """
    model, collection = _get_resources()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    hits = []
    if not results["ids"] or not results["ids"][0]:
        return hits

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        # Chroma default is L2 distance; convert to a rough 0-1 similarity proxy.
        similarity = 1 / (1 + distance)
        hits.append({
            "text": results["documents"][0][i],
            "source_file": results["metadatas"][0][i]["source_file"],
            "section_title": results["metadatas"][0][i]["section_title"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "similarity": round(similarity, 4),
        })
    return hits


def retrieve_by_doc_id(doc_id: str, top_k: int = 50):
    """Fetch all chunks belonging to a specific document (used by /contradict)."""
    _, collection = _get_resources()
    results = collection.get(where={"doc_id": doc_id}, limit=top_k)
    chunks = []
    for i in range(len(results["ids"])):
        chunks.append({
            "text": results["documents"][i],
            "source_file": results["metadatas"][i]["source_file"],
            "section_title": results["metadatas"][i]["section_title"],
        })
    return chunks
