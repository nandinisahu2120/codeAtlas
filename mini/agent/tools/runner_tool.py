"""
Runner Tool -- Safely executes a Python file in a subprocess.
Captures stdout, stderr, and exit code. Has a timeout guard.
"""
import sys
import subprocess
from pathlib import Path
from config import RUN_TIMEOUT, get_repo_dir, BASE_DIR


def run_file(file_path: str) -> dict:
    """
    Execute a Python file from the target repo or project root.
    Returns stdout, stderr, exit code, and whether it timed out.
    """
    repo_dir = get_repo_dir()
    full_path = Path(file_path)

    # If relative path given, resolve from repo dir or base dir
    if not full_path.is_absolute():
        if (repo_dir / file_path).exists():
            full_path = repo_dir / file_path
        elif (BASE_DIR / file_path).exists():
            full_path = BASE_DIR / file_path
        else:
            full_path = repo_dir / file_path

    if not full_path.exists():
        return {
            "status":    "error",
            "message":   f"File not found: {full_path}",
            "stdout":    "",
            "stderr":    "",
            "exit_code": -1,
        }

    if not full_path.suffix == ".py":
        return {
            "status":    "error",
            "message":   f"Only .py files can be executed. Got: {full_path.suffix}",
            "stdout":    "",
            "stderr":    "",
            "exit_code": -1,
        }

    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
            cwd=str(repo_dir),
        )
        return {
            "status":    "ok",
            "stdout":    result.stdout[-2000:] if result.stdout else "",
            "stderr":    result.stderr[-2000:] if result.stderr else "",
            "exit_code": result.returncode,
            "timed_out": False,
            "message":   f"Executed {full_path.name} (exit code {result.returncode})",
        }

    except subprocess.TimeoutExpired:
        return {
            "status":    "timeout",
            "stdout":    "",
            "stderr":    "",
            "exit_code": -1,
            "timed_out": True,
            "message":   f"Execution timed out after {RUN_TIMEOUT}s",
        }

    except Exception as e:
        return {
            "status":    "error",
            "stdout":    "",
            "stderr":    str(e),
            "exit_code": -1,
            "timed_out": False,
            "message":   f"Error running file: {e}",
        }
