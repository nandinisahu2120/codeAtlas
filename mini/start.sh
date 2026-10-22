#!/usr/bin/env bash
# ========================================================
# codeAtlas macOS / Linux Launch Script
# Sets threading guards and starts the Streamlit UI
# ========================================================

export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
export KMP_INIT_AT_FORK=FALSE
export PYTHONIOENCODING=utf-8

if [ -x "./venv/bin/python" ]; then
    ./venv/bin/python -m streamlit run ui/app.py --server.port 8501 --server.fileWatcherType none
else
    python3 -m streamlit run ui/app.py --server.port 8501 --server.fileWatcherType none
fi
