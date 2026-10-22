@echo off
REM ========================================================
REM codeAtlas Windows Launch Script
REM Sets threading guards and starts the Streamlit UI
REM ========================================================

set OMP_NUM_THREADS=1
set KMP_DUPLICATE_LIB_OK=TRUE
set TOKENIZERS_PARALLELISM=false
set KMP_INIT_AT_FORK=FALSE
set PYTHONIOENCODING=utf-8

if exist "venv\Scripts\python.exe" (
    echo Starting codeAtlas using venv...
    venv\Scripts\python.exe -m streamlit run ui\app.py --server.port 8501 --server.fileWatcherType none
) else (
    echo Starting codeAtlas using system Python...
    python -m streamlit run ui\app.py --server.port 8501 --server.fileWatcherType none
)

pause
