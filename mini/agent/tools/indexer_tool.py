"""
Indexer Tool -- Embeds code chunks with MiniLM and builds a FAISS index.
Fully offline. Model downloads once (~90 MB), then runs from cache.
"""
import json
import numpy as np
import faiss

try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHUNKS_FILE, INDEX_FILE, META_FILE, DATA_DIR

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def build_index(chunks: list[dict] | None = None) -> dict:
    """
    Embed all chunks and build a FAISS inner-product index.
    Saves faiss.index and metadata.json to data/.
    """
    if chunks is None:
        if not CHUNKS_FILE.exists():
            return {"status": "error", "message": "No chunks found. Run parser first."}
        chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    if not chunks:
        return {
            "status": "error",
            "message": "Zero chunks to index. The repository appears to have no parseable source code files (.py, .js, .jsx, .ts, .tsx, .java, .go, .cpp, .html, etc.) or all files were empty.",
        }

    model = _get_model()

    # Embed the combined text field (name + docstring + code)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (inner product on normalized vectors = cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    # Save metadata (chunk id -> position mapping)
    meta = []
    for i, c in enumerate(chunks):
        meta.append({
            "idx":        i,
            "id":         c["id"],
            "file":       c["file"],
            "name":       c["name"],
            "start_line": c["start_line"],
            "end_line":   c["end_line"],
            "type":       c["type"],
            "language":   c.get("language", "python"),
        })
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Build Call Graph & Dependency Topology
    try:
        from agent.tools.graph_tool import build_dependency_graph
        build_dependency_graph(chunks)
    except Exception as e:
        print(f"[codeAtlas] Warning: Dependency graph build skipped: {e}")

    return {
        "status":    "ok",
        "indexed":   len(chunks),
        "dimension": dim,
        "message":   f"Indexed {len(chunks)} chunks ({dim}-dim vectors)",
    }


def load_index():
    """Load the FAISS index and metadata from disk."""
    if not INDEX_FILE.exists() or not META_FILE.exists():
        return None, None
    index = faiss.read_index(str(INDEX_FILE))
    meta  = json.loads(META_FILE.read_text(encoding="utf-8"))
    return index, meta
