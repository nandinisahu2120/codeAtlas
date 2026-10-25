# codeAtlas System Architecture & Design Specification

## 1. System Vision
codeAtlas is designed to solve the data sovereignty and intellectual property risks inherent in cloud-based developer tools. It provides private, air-gapped semantic retrieval, multi-language AST topology mapping, and local causal reasoning directly on developer machines.

---

## 2. Layered Architecture

```text
+------------------------------------------------------------------+
|                     User & Integration Layer                     |
|  - Streamlit UI (Tokyo Night Dark Theme)                         |
|  - Model Context Protocol (Anthropic MCP JSON-RPC Server)        |
+------------------------------------------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|               LangGraph Deterministic State Machine              |
|  - Intent Detection Router                                       |
|  - Nodes: Parser, Search, Graph, Overview, Explain, Bug, Runner  |
|  - Checkpointer: Thread-Safe MemorySaver (Multi-turn Memory)     |
+------------------------------------------------------------------+
             |                           |                     |
             v                           v                     v
+------------------------+  +----------------------+  +---------------------+
|      Vector Engine     |  |   AST Call Graph     |  | Local SLM Inference |
| - all-MiniLM-L6-v2     |  | - Tree-sitter AST    |  | - Qwen2.5-1.5B      |
| - 384-d Embeddings     |  | - NetworkX DiGraph   |  | - Quantized CPU     |
| - FAISS IndexFlatIP    |  | - Blast Radius       |  | - Platform Dtype    |
+------------------------+  +----------------------+  +---------------------+
```

---

## 3. Core Subsystems

### 3.1 Syntax-Aware AST Parsing
- Extracts functions, methods, classes, and declarations across Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, and SQL using Tree-sitter.
- Line-sliding window fallback mechanism guarantees parsing robustness for unconventional or unsupported file formats.

### 3.2 Dual Vector & Topological Storage
- **Vector Space:** 384-dimensional dense semantic vectors normalized for inner-product cosine similarity.
- **Topological Graph:** Directed graph modeling caller-to-callee dependencies and file-level import relationships.

### 3.3 Explainable AI (XAI) Scoring
- Calculates token contribution weights using vectorized cosine attribution against the query, giving developers transparent insight into why code matches were returned.

---

## 4. Hardware and Platform Compatibility
- **macOS (Apple Silicon):** Native bfloat16 vector acceleration.
- **Windows (x86 CPU):** Automatically utilizes float32 to prevent OpenMP bfloat16 software emulation bottlenecks.
- **Linux:** Fully supported with standard POSIX process management.
