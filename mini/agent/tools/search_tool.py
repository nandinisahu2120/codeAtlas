"""
Search Tool -- Semantic vector search over the FAISS index.
Takes a natural-language query, returns top-k matching code chunks.
"""
import json
import numpy as np
from config import TOP_K, CHUNKS_FILE
from agent.tools.indexer_tool import _get_model, load_index


def search(query: str, top_k: int = TOP_K) -> dict:
    """
    Embed the user query with MiniLM and search the FAISS index.
    Returns top-k code chunks ranked by cosine similarity.
    """
    index, meta = load_index()
    if index is None:
        return {"status": "error", "results": [], "message": "No index found. Index the repo first."}

    # Load full chunks for code content
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    chunk_by_idx = {i: c for i, c in enumerate(chunks)}

    # Embed query
    model = _get_model()
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    # Search
    scores, indices = index.search(q_vec, min(top_k, index.ntotal))

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue
        m = meta[idx]
        c = chunk_by_idx.get(idx, {})
        results.append({
            "rank":       rank + 1,
            "score":      float(score),
            "id":         m["id"],
            "file":       m["file"],
            "name":       m["name"],
            "start_line": m["start_line"],
            "end_line":   m["end_line"],
            "type":       m["type"],
            "language":   m.get("language", "unknown"),
            "code":       c.get("code", ""),
        })

    return {
        "status":  "ok",
        "query":   query,
        "results": results,
        "message": f"Found {len(results)} matches for: {query}",
    }


def get_token_contributions(query: str, code: str) -> list[dict]:
    """
    Lightweight explainability: measure each query token's individual
    cosine similarity against the code embedding in a single batch pass.
    """
    valid_tokens = [t for t in query.lower().split() if len(t) >= 2]
    if not valid_tokens:
        return []

    model = _get_model()
    # Batch encode code + all query tokens in one single forward pass
    all_vecs = model.encode([code] + valid_tokens, normalize_embeddings=True)
    code_vec = all_vecs[0]
    tok_vecs = all_vecs[1:]

    sims = np.dot(tok_vecs, code_vec)
    contributions = [
        {"token": tok, "contribution": round(float(s), 4)}
        for tok, s in zip(valid_tokens, sims)
    ]
    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    return contributions
