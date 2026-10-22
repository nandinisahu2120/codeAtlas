"""
Repo Loader Tool -- Clones remote GitHub repositories or resolves local project directories.
"""
import re
import subprocess
from pathlib import Path
from config import DATA_DIR, set_repo_dir

REPOS_CACHE_DIR = DATA_DIR / "repos"


def is_github_url(source: str) -> bool:
    """Check if the string is a GitHub or Git URL."""
    s = source.strip()
    return bool(
        re.match(r"^(https?://github\.com/|git@github\.com:)", s)
        or s.startswith("github.com/")
    )


def clone_or_load_repo(source: str) -> dict:
    """
    Accepts either a GitHub repository URL or a local directory path.
    Clones GitHub repos into data/repos/<repo_name> with depth=1.
    Sets active repo and returns status dict with target Path.
    """
    cleaned = source.strip().rstrip("/")
    if not cleaned:
        return {"status": "error", "message": "Repository source cannot be empty."}

    # Normalize github.com shorthand
    if cleaned.startswith("github.com/"):
        cleaned = f"https://{cleaned}"

    # Handle GitHub repository
    if is_github_url(cleaned):
        # Extract owner and repo name
        match = re.search(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?$", cleaned)
        if match:
            owner, repo_name = match.group(1), match.group(2)
            dir_name = f"{owner}_{repo_name}"
        else:
            dir_name = re.sub(r"[^\w.-]", "_", cleaned.split("/")[-1])

        REPOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dest_dir = REPOS_CACHE_DIR / dir_name

        # If already cloned, use it or pull
        if dest_dir.exists() and (dest_dir / ".git").exists():
            set_repo_dir(dest_dir)
            return {
                "status": "ok",
                "path": dest_dir,
                "name": dest_dir.name,
                "message": f"Using cached clone of {cleaned} at {dest_dir}",
                "is_cloned": True,
            }

        # Clone fresh shallow copy
        try:
            cmd = ["git", "clone", "--depth", "1", cleaned, str(dest_dir)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode != 0:
                return {
                    "status": "error",
                    "message": f"Git clone failed: {res.stderr.strip() or res.stdout.strip()}",
                }
            set_repo_dir(dest_dir)
            return {
                "status": "ok",
                "path": dest_dir,
                "name": dest_dir.name,
                "message": f"Successfully cloned {cleaned}",
                "is_cloned": True,
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Git clone timed out after 120 seconds."}
        except Exception as e:
            return {"status": "error", "message": f"Error cloning repository: {e}"}

    # Handle local path
    local_path = Path(cleaned).expanduser().resolve()
    if not local_path.exists():
        return {"status": "error", "message": f"Local directory not found: {local_path}"}
    if not local_path.is_dir():
        return {"status": "error", "message": f"Path is not a directory: {local_path}"}

    set_repo_dir(local_path)
    return {
        "status": "ok",
        "path": local_path,
        "name": local_path.name,
        "message": f"Targeting local repository at {local_path}",
        "is_cloned": False,
    }
