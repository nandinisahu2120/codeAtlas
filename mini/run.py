"""
Universal Cross-Platform Launcher for codeAtlas.
Works on Windows, macOS, and Linux.
Usage:
    python run.py
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Cross-platform environment configuration
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).resolve().parent
APP_SCRIPT = PROJECT_ROOT / "ui" / "app.py"


def find_python_executable():
    """Detect virtualenv Python executable across Windows, macOS, and Linux."""
    # Windows virtualenv
    win_venv = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    if win_venv.exists():
        return str(win_venv)

    # Unix (macOS / Linux) virtualenv
    unix_venv = PROJECT_ROOT / "venv" / "bin" / "python"
    unix_venv3 = PROJECT_ROOT / "venv" / "bin" / "python3"
    if unix_venv.exists():
        return str(unix_venv)
    if unix_venv3.exists():
        return str(unix_venv3)

    # Fallback to active runtime python
    return sys.executable


def main():
    python_exe = find_python_executable()
    print("=" * 60)
    print("codeAtlas: Agentic Semantic Code Search Engine")
    print(f"Operating System: {sys.platform}")
    print(f"Python Runtime:   {python_exe}")
    print(f"Project Root:     {PROJECT_ROOT}")
    print("=" * 60)

    cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        str(APP_SCRIPT),
        "--server.port",
        "8501",
        "--server.headless",
        "false",
        "--server.fileWatcherType",
        "none",
    ]

    print("Launching codeAtlas UI at http://localhost:8501 ...")
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ)
    except KeyboardInterrupt:
        print("\ncodeAtlas shut down gracefully.")


if __name__ == "__main__":
    main()
