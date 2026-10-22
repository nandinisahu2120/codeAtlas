"""
Evaluation script: Semantic Search (MiniLM + FAISS) vs Grep Baseline.
Measures Top-1 and Top-3 accuracy on 10 realistic code search queries.
"""
import sys
import json
import time
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
from agent.tools.search_tool import search
from config import REPO_DIR, get_repo_dir


def run_grep_search(query: str, repo_dir: Path) -> list[str]:
    """Keyword grep baseline using individual query terms with cross-platform fallback."""
    terms = [w.lower() for w in query.split() if len(w) > 3]
    if not terms:
        terms = query.split()

    matches = []
    has_grep = shutil.which("grep") is not None

    if has_grep:
        for term in terms:
            try:
                res = subprocess.run(
                    ["grep", "-ril", term, str(repo_dir)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in res.stdout.strip().splitlines():
                    if line:
                        rel = Path(line).relative_to(repo_dir).as_posix()
                        if rel not in matches:
                            matches.append(rel)
            except Exception:
                pass
    else:
        # Pure-Python fallback for Windows / environments without grep
        for file_path in repo_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in file_path.parts for part in {".git", "venv", "__pycache__", "data"}):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore").lower()
                for term in terms:
                    if term in content:
                        rel = file_path.relative_to(repo_dir).as_posix()
                        if rel not in matches:
                            matches.append(rel)
                        break
            except Exception:
                pass

    return matches


def evaluate():
    from agent.tools.parser_tool import parse_repo, load_chunks
    from agent.tools.indexer_tool import build_index
    from config import set_repo_dir

    prev_active = get_repo_dir()
    eval_file = Path(__file__).parent / "eval_queries.json"
    queries = json.loads(eval_file.read_text(encoding="utf-8"))

    # If queries target target_repo, ensure target_repo is parsed and indexed
    if queries and (REPO_DIR / queries[0]["expected_file"]).exists():
        if prev_active != REPO_DIR:
            set_repo_dir(REPO_DIR)
            chunks = parse_repo(str(REPO_DIR))
            build_index(load_chunks())

    target_eval_dir = get_repo_dir()

    semantic_top1 = 0
    semantic_top3 = 0
    grep_top1 = 0
    grep_top3 = 0

    print("Running 10-query benchmark: Semantic Search vs Grep Baseline")
    print("=" * 70)

    for item in queries:
        qid = item["id"]
        q = item["query"]
        exp_file = item["expected_file"]
        exp_name = item.get("expected_name", "")

        # Semantic Search
        t0 = time.time()
        sem_res = search(q, top_k=3)
        sem_time = (time.time() - t0) * 1000
        sem_hits = sem_res.get("results", [])

        sem_files = [h["file"] for h in sem_hits]
        sem_hit_top1 = len(sem_files) > 0 and sem_files[0] == exp_file
        sem_hit_top3 = exp_file in sem_files[:3]

        if sem_hit_top1:
            semantic_top1 += 1
        if sem_hit_top3:
            semantic_top3 += 1

        # Grep Search
        t0 = time.time()
        grep_files = run_grep_search(q, get_repo_dir())
        grep_time = (time.time() - t0) * 1000

        grep_hit_top1 = len(grep_files) > 0 and grep_files[0] == exp_file
        grep_hit_top3 = exp_file in grep_files[:3]

        if grep_hit_top1:
            grep_top1 += 1
        if grep_hit_top3:
            grep_top3 += 1

        top_match_str = f"{sem_hits[0]['file']}::{sem_hits[0]['name']}" if sem_hits else "none"
        print(f"[{qid:02d}] Query: {q}")
        print(f"     Expected: {exp_file} ({exp_name})")
        print(f"     Semantic Top-1: {top_match_str} (hit={sem_hit_top1}, {sem_time:.1f}ms)")
        print(f"     Grep Top-1:     {grep_files[0] if grep_files else 'none'} (hit={grep_hit_top1}, {grep_time:.1f}ms)")
        print("-" * 70)

    n = len(queries)
    print("\nSummary of Benchmark Results:")
    print(f"Total Queries:         {n}")
    print(f"Semantic Search Top-1: {semantic_top1}/{n} ({semantic_top1/n*100:.1f}%)")
    print(f"Semantic Search Top-3: {semantic_top3}/{n} ({semantic_top3/n*100:.1f}%)")
    print(f"Grep Baseline Top-1:   {grep_top1}/{n} ({grep_top1/n*100:.1f}%)")
    print(f"Grep Baseline Top-3:   {grep_top3}/{n} ({grep_top3/n*100:.1f}%)")

    # Restore active repository for the UI session
    if prev_active != target_eval_dir and prev_active.exists():
        set_repo_dir(prev_active)
        parse_repo(str(prev_active))
        build_index(load_chunks())


if __name__ == "__main__":
    evaluate()
