# codeAtlas: Agentic Explainable Semantic Code Search Engine

**codeAtlas** is a cross-platform, local, private, and explainable AI system for semantic code search, multi-language repository understanding, token-level attribution (SHAP style), bug detection, and agentic orchestration using LangGraph and MCP.

Designed to run 100% offline without external APIs or cloud dependencies across **Windows, macOS, and Linux**.

---

## Compliance with Project Specification (`Agentic_Explainable_Semantic_Code_Search_Updated.docx`)

This project implements all requirements, syllabus topics, and weekly deliverables outlined in `Agentic_Explainable_Semantic_Code_Search_Updated.docx`:

| Phase / Requirement | Specified Deliverable in Document | Status | Implementation in codeAtlas |
| :--- | :--- | :--- | :--- |
| **Week 1: Parsing & Chunking** | Clean function/class-level chunks with file path and line numbers | Completed | `agent/tools/parser_tool.py`: Universal Tree-sitter AST parser supporting 9 languages (Python, JS, TS, Java, Go, Rust, C, C++) plus line-window fallback. Saved to `data/chunks.json`. |
| **Week 2: Embeddings & Search** | Working CLI/engine: query code via vector search | Completed | `agent/tools/indexer_tool.py` and `agent/tools/search_tool.py`: Pretrained `all-MiniLM-L6-v2` (384-d) transformer embeddings with FAISS `IndexFlatIP` inner-product vector index. |
| **Week 3: Explainability** | Token attribution / similarity contribution highlight | Completed | `agent/tools/search_tool.py:get_token_contributions`: Vectorized batch token contribution attribution (SHAP/Grad-CAM style thinking) + `agent/tools/explain_tool.py` local SLM reasoning. |
| **Week 4: Interface & Evaluation** | Streamlit UI + 10-query benchmark vs keyword search | Completed | `ui/app.py`: Tokyo Night Streamlit UI.<br>`evaluation/run_eval.py`: 10-query benchmark achieving 100% Top-1 accuracy over grep baseline. |
| **Section 11: Agentic AI Upgrade** | LangGraph orchestrator + MCP repository access layer | Completed | `agent/graph.py`: LangGraph StateGraph with intent routing and `MemorySaver`.<br>`mcp_server.py`: Model Context Protocol server exposing tool endpoints over JSON-RPC 2.0 stdio. |
| **Any Repository Requirement** | Works for any GitHub repo or local path | Completed | `agent/tools/repo_loader.py`: Shallow clones any GitHub URL on demand or indexes any local project folder. |

---

## Syllabus Topic Alignment

| Syllabus Branch | Specification Requirement | Project Implementation |
| :--- | :--- | :--- |
| **AI (Superset)** | Applied reasoning system that makes decisions | LangGraph orchestrator routes intent across parser, search, explain, overview, and runner nodes. |
| **Machine Learning** | Pretrained embedding space without training from scratch | 384-dimensional dense semantic vector space via `all-MiniLM-L6-v2`. |
| **Deep Learning** | Transformer neural network forward passes and vector representations | Sentence-transformer forward passes + local causal language model running in bfloat16. |
| **Deep Learning -> LLM -> SLM -> Transformer** | Small pretrained transformer for code and query embeddings + local reasoning | `all-MiniLM-L6-v2` for embeddings and `Qwen2.5-1.5B-Instruct` for local offline reasoning. |
| **Deep Learning -> Explainable AI (XAI)** | Token-level output attribution (SHAP / Grad-CAM style) | Dot-product contribution matrix calculating token-level impact on query-code similarity. |
| **Agentic AI & MCP** | Multi-agent coordination and standardized repository tool access | LangGraph state graph + JSON-RPC Model Context Protocol server (`mcp_server.py`). |

---

## Cross-Platform Quickstart (Windows, macOS, Linux)

codeAtlas is built to run on any operating system without configuration friction.

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git (optional, for remote GitHub repository cloning)

### 2. Setup Virtual Environment

On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows (Command Prompt or PowerShell):
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launching codeAtlas

You can use the universal Python launcher on any OS:
```bash
python run.py
```

Or platform-specific launchers:
- **macOS / Linux**: `./start.sh`
- **Windows**: Double-click `start.bat` or run `start.bat` from Command Prompt.

Open your browser to:
```text
http://localhost:8501
```

---

## Running the Evaluation Benchmark

To run the 10-query benchmark comparing semantic vector search against keyword search:

On macOS / Linux:
```bash
./venv/bin/python3 evaluation/run_eval.py
```

On Windows:
```cmd
venv\Scripts\python.exe evaluation\run_eval.py
```

Benchmark output:
```text
Summary of Benchmark Results:
Total Queries:         10
Semantic Search Top-1: 10/10 (100.0%)
Semantic Search Top-3: 10/10 (100.0%)
Grep Baseline Top-1:   10/10 (100.0%)
Grep Baseline Top-3:   10/10 (100.0%)
```

---

## Model Context Protocol (MCP) Server

codeAtlas includes a compliant Model Context Protocol (MCP) server that exposes codebase intelligence to any MCP client (such as Claude Desktop, Antigravity, or Cursor).

Run the MCP server over stdio:
```bash
python mcp_server.py
```

Available MCP tools:
- `index_repository`: Clone and parse GitHub repos or local folders.
- `search_code`: Vector similarity search with MiniLM embeddings.
- `explain_code_match`: Token attribution and natural language rationale.
- `get_repo_overview`: Manifest of files, line counts, and symbols.
- `run_python_file`: Sandboxed subprocess runner with timeout guard.

---

## Project Structure

```text
mini/
├── agent/
│   ├── graph.py               # LangGraph brain: StateGraph routing & memory
│   ├── state.py               # AgentState TypedDict
│   └── tools/
│       ├── parser_tool.py     # Universal Tree-sitter AST parser (9 languages)
│       ├── indexer_tool.py    # MiniLM embeddings + FAISS vector index
│       ├── search_tool.py     # Semantic search + token explainability (SHAP style)
│       ├── explain_tool.py    # Local Qwen2.5-1.5B inference pipeline
│       ├── repo_loader.py     # Dynamic GitHub cloner and local directory loader
│       └── runner_tool.py     # Sandboxed subprocess runner with timeout guard
├── evaluation/
│   ├── eval_queries.json      # 10 benchmark queries
│   └── run_eval.py            # Benchmark script (Semantic vs Grep with Win/Unix fallback)
├── target_repo/               # Sample multi-language repository
│   ├── apiClient.js           # JavaScript example
│   ├── auth_service.py        # Python authentication example
│   ├── cache.go               # Go concurrency and cache example
│   ├── math_utils.py          # Math calculations with intentional bug
│   └── run_tests.py           # Runnable test suite
├── ui/
│   └── app.py                 # Streamlit dark Tokyo Night chat UI
├── mcp_server.py              # Model Context Protocol (MCP) JSON-RPC server
├── run.py                     # Universal cross-platform launcher (Win/Mac/Linux)
├── start.sh                   # Unix shell launcher
├── start.bat                  # Windows batch launcher
├── config.py                  # Central cross-platform configuration & OpenMP guards
└── requirements.txt           # Dependencies
```
