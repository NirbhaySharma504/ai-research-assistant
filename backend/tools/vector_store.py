"""ChromaDB vector store with per-session collections and MiniLM embeddings.

Each research session gets its own collection (session_{id}) so retrieval never
crosses session boundaries. The embedding model and Chroma client are loaded once
and reused.
"""

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from backend.config import settings

_model: SentenceTransformer | None = None
_client: chromadb.ClientAPI | None = None
_splitter: RecursiveCharacterTextSplitter | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    return _client


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _splitter


def _collection(session_id: str):
    # cosine distance for semantic retrieval
    return _get_client().get_or_create_collection(
        f"session_{session_id}", metadata={"hnsw:space": "cosine"}
    )


def store_content(text: str, metadata: dict, session_id: str) -> int:
    """Chunk, embed, and store text. Returns number of chunks stored."""
    chunks = _get_splitter().split_text(text)
    if not chunks:
        return 0
    embeddings = _get_model().encode(chunks).tolist()
    collection = _collection(session_id)
    url = metadata.get("url", "doc")
    ids = [f"{url}_chunk_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{**metadata, "chunk_index": i} for i in range(len(chunks))],
        ids=ids,
    )
    return len(chunks)


def retrieve_relevant(query: str, session_id: str, k: int | None = None) -> list[dict]:
    """Return top-k {content, metadata, distance} chunks for the query."""
    k = k or settings.RETRIEVAL_TOP_K
    query_embedding = _get_model().encode([query]).tolist()[0]
    collection = _collection(session_id)
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_embeddings=[query_embedding], n_results=min(k, count)
    )
    return [
        {"content": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def delete_session(session_id: str) -> None:
    try:
        _get_client().delete_collection(f"session_{session_id}")
    except Exception:  # noqa: BLE001
        pass
