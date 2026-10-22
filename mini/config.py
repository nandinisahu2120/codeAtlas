import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
REPO_DIR  = BASE_DIR / "target_repo"
ACTIVE_REPO_FILE = DATA_DIR / "active_repo.txt"


def get_repo_dir() -> Path:
    if ACTIVE_REPO_FILE.exists():
        try:
            p = Path(ACTIVE_REPO_FILE.read_text(encoding="utf-8").strip())
            if p.exists() and p.is_dir():
                return p
        except Exception:
            pass
    return REPO_DIR


def set_repo_dir(new_path: str | Path) -> Path:
    p = Path(new_path).expanduser().resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_REPO_FILE.write_text(str(p), encoding="utf-8")
    return p


CHUNKS_FILE = DATA_DIR / "chunks.json"
INDEX_FILE  = DATA_DIR / "faiss.index"
META_FILE   = DATA_DIR / "metadata.json"

# Embedding model — 90 MB, downloaded once, runs offline
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Local LLM — ~1.5 GB, no API, runs fully offline on 8 GB RAM
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Search
TOP_K          = 5
MAX_NEW_TOKENS = 160
CHUNK_MAX_LINES = 60   # skip chunks longer than this (avoids noise)
RUN_TIMEOUT    = 15    # seconds before killing a subprocess
