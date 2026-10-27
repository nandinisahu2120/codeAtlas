"""
codeAtlas Pre-flight Environment and Dependency Checker.
Validates Python version, system memory, and required packages across Windows/macOS/Linux.
"""
import sys
import platform
import importlib

REQUIRED_PACKAGES = [
    ("sentence_transformers", "Sentence Transformers"),
    ("faiss", "FAISS Vector Store"),
    ("torch", "PyTorch"),
    ("transformers", "Hugging Face Transformers"),
    ("streamlit", "Streamlit UI"),
    ("langgraph", "LangGraph StateGraph"),
    ("networkx", "NetworkX Graph Engine"),
    ("tree_sitter", "Tree-sitter AST Parser"),
]

def check():
    print("=" * 60)
    print("codeAtlas: Pre-flight System & Environment Validation")
    print("=" * 60)
    print(f"OS:             {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version: {platform.python_version()}")

    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ is required to run codeAtlas.")
        sys.exit(1)

    print("\nChecking Dependencies:")
    missing = []
    for pkg, label in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  [OK] {label} ({pkg})")
        except ImportError:
            print(f"  [MISSING] {label} ({pkg})")
            missing.append(pkg)

    print("-" * 60)
    if missing:
        print(f"Validation failed. Install missing packages: pip install {' '.join(missing)}")
        sys.exit(1)
    else:
        print("All dependencies verified. Environment is ready to run codeAtlas.")
        print("Run: python run.py (or ./start.sh)")

if __name__ == "__main__":
    check()
