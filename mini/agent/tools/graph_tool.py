"""
Call Graph & AST Dependency Topology Tool for codeAtlas.
Extracts function caller-callee relations, import dependency graphs,
and performs impact analysis using NetworkX.
"""
import json
import os
import re
from pathlib import Path
import networkx as nx

from config import CALL_GRAPH_FILE, CHUNKS_FILE, DATA_DIR, get_repo_dir

# Common language keywords to ignore when identifying function call identifiers
IGNORED_CALL_KEYWORDS = {
    # Python
    "if", "elif", "else", "for", "while", "with", "try", "except", "finally",
    "def", "class", "return", "yield", "raise", "assert", "import", "from",
    "lambda", "print", "len", "range", "enumerate", "zip", "isinstance",
    "type", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "super", "getattr", "setattr", "hasattr", "open", "input",
    # JS / TS
    "function", "const", "let", "var", "switch", "case", "catch", "throw",
    "typeof", "instanceof", "new", "delete", "void", "in", "of",
    "console", "log", "warn", "error", "require", "import", "export",
    "resolve", "reject", "Promise", "Array", "Object", "String", "Number",
    # General / C / Java / Go
    "public", "private", "protected", "static", "void", "package", "main",
    "fmt", "println", "printf", "make", "append", "panic", "recover",
}


def _extract_calls_from_text(code_text: str, caller_name: str | None = None) -> set[str]:
    """
    Extract called function identifiers from code snippet.
    Finds patterns like `func_name(` or `obj.method_name(`.
    Ignores the function's own definition header.
    """
    pattern = re.compile(r"(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
    matches = set()
    for m in pattern.finditer(code_text):
        name = m.group(1)
        if name not in IGNORED_CALL_KEYWORDS and len(name) > 1:
            matches.add(name)

    # If caller_name is in matches, check if it's truly recursive or just the definition header
    if caller_name and caller_name in matches:
        # Check if caller_name( appears more than once in the code
        signature_calls = len(re.findall(rf"\b{re.escape(caller_name)}\s*\(", code_text))
        if signature_calls <= 1:
            matches.discard(caller_name)

    return matches


def _extract_imports_from_source(source: str, file_suffix: str) -> set[str]:
    """
    Extract imported module/file names from source code.
    """
    imported = set()
    s = file_suffix.lower()

    if s == ".py":
        # from foo.bar import baz OR import foo
        for m in re.finditer(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import", source, re.M):
            mod = m.group(1).split(".")[0]
            if mod:
                imported.add(mod)
        for m in re.finditer(r"^\s*import\s+([a-zA-Z0-9_,\. ]+)", source, re.M):
            line = m.group(1)
            for item in line.split(","):
                mod = item.strip().split()[0].split(".")[0]
                if mod:
                    imported.add(mod)

    elif s in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
        # import ... from './foo' OR require('./foo')
        for m in re.finditer(r"""(?:from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\))""", source):
            target = m.group(1) or m.group(2)
            if target:
                imported.add(Path(target).name)

    elif s == ".java":
        for m in re.finditer(r"^\s*import\s+([a-zA-Z0-9_\.]+);", source, re.M):
            pkg = m.group(1)
            imported.add(pkg.split(".")[-1])

    elif s == ".go":
        for m in re.finditer(r'"([^"]+)"', source):
            target = m.group(1)
            imported.add(Path(target).name)

    elif s in (".c", ".cpp", ".cc", ".h", ".hpp"):
        for m in re.finditer(r"""#include\s*[<"]([^>"]+)[>"]""", source):
            target = m.group(1)
            imported.add(Path(target).name)

    return imported


def build_dependency_graph(chunks: list[dict] | None = None, repo_dir: Path | None = None) -> nx.DiGraph:
    """
    Build a NetworkX directed graph containing:
    - File nodes and import dependencies
    - Function/Class nodes and definitions
    - Directed caller -> callee call edges
    """
    if chunks is None:
        if not CHUNKS_FILE.exists():
            return nx.DiGraph()
        try:
            chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        except Exception:
            chunks = []

    if repo_dir is None:
        repo_dir = get_repo_dir()

    G = nx.DiGraph()

    # 1. Map of defined symbols: name -> list of chunk dicts
    symbol_definitions: dict[str, list[dict]] = {}
    files_seen: set[str] = set()

    for c in chunks:
        name = c.get("name")
        file_path = c.get("file")
        if not name or not file_path:
            continue

        files_seen.add(file_path)
        if name != "<anonymous>":
            symbol_definitions.setdefault(name, []).append(c)

        node_id = c.get("id") or f"{file_path}::{name}@{c.get('start_line', 0)}"
        G.add_node(
            node_id,
            node_type="symbol",
            symbol_type=c.get("type", "function"),
            name=name,
            file=file_path,
            start_line=c.get("start_line", 0),
            end_line=c.get("end_line", 0),
            language=c.get("language", ""),
        )

        # File defines symbol edge
        file_node = f"file::{file_path}"
        if not G.has_node(file_node):
            G.add_node(file_node, node_type="file", name=file_path, file=file_path)
        G.add_edge(file_node, node_id, relation="defines")

    # 2. Extract call edges (Caller -> Callee)
    for c in chunks:
        caller_id = c.get("id") or f"{c.get('file')}::{c.get('name')}@{c.get('start_line', 0)}"
        caller_name = c.get("name")
        code = c.get("code", "")
        calls = _extract_calls_from_text(code, caller_name=caller_name)

        for called_name in calls:
            # Check if this called symbol exists in our project definitions
            targets = symbol_definitions.get(called_name, [])
            for target_chunk in targets:
                callee_id = target_chunk.get("id")
                # Do not add self-loops unless recursive call
                if callee_id and (callee_id != caller_id or caller_name == called_name):
                    G.add_edge(
                        caller_id,
                        callee_id,
                        relation="calls",
                        caller=caller_name,
                        callee=called_name,
                    )

    # 3. Extract file-level import dependencies
    file_map_by_stem = {Path(f).stem: f for f in files_seen}
    for file_path in files_seen:
        full_path = repo_dir / file_path
        if full_path.exists() and full_path.is_file():
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
                imports = _extract_imports_from_source(content, full_path.suffix)
                src_file_node = f"file::{file_path}"

                for imp in imports:
                    imp_stem = Path(imp).stem
                    if imp_stem in file_map_by_stem and file_map_by_stem[imp_stem] != file_path:
                        target_file = file_map_by_stem[imp_stem]
                        tgt_file_node = f"file::{target_file}"
                        G.add_edge(src_file_node, tgt_file_node, relation="imports")
            except Exception:
                pass

    save_graph(G)
    return G


def save_graph(G: nx.DiGraph, path: Path = CALL_GRAPH_FILE) -> bool:
    """Save graph to JSON node-link format."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(G)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        print(f"[codeAtlas] Warning: Failed to save call graph: {e}")
        return False


def load_graph(path: Path = CALL_GRAPH_FILE) -> nx.DiGraph:
    """Load graph from JSON node-link format with fallback to build."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return nx.node_link_graph(data)
        except Exception:
            pass
    return build_dependency_graph()


# ---------------------------------------------------------------------------
# Graph Traversal & Impact Analysis Queries
# ---------------------------------------------------------------------------

def _find_symbol_nodes(G: nx.DiGraph, symbol_name: str) -> list[str]:
    """Find all node IDs matching a given function/symbol name."""
    clean_target = symbol_name.strip().strip("'\"()")
    matches = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "symbol":
            name = data.get("name", "")
            if name.lower() == clean_target.lower():
                matches.append(node)
    # If no exact match, try partial/suffix match
    if not matches:
        for node, data in G.nodes(data=True):
            if data.get("node_type") == "symbol":
                name = data.get("name", "")
                if clean_target.lower() in name.lower():
                    matches.append(node)
    return matches


def analyze_impact(symbol_name: str, G: nx.DiGraph | None = None) -> dict:
    """
    Performs impact and blast-radius analysis:
    'Which functions break if I change this signature?'
    """
    if G is None:
        G = load_graph()

    matching_nodes = _find_symbol_nodes(G, symbol_name)
    if not matching_nodes:
        return {
            "status": "not_found",
            "symbol": symbol_name,
            "message": f"Symbol '{symbol_name}' was not found in the indexed code graph.",
        }

    # Focus on the first primary match (or aggregate)
    primary_node = matching_nodes[0]
    node_data = G.nodes[primary_node]
    file_path = node_data.get("file", "unknown")
    start_line = node_data.get("start_line", 0)
    end_line = node_data.get("end_line", 0)

    # 1. Build call-only subgraph
    call_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "calls"
    ]
    call_graph = G.edge_subgraph(call_edges).copy()

    # 2. Direct callers (predecessors in call graph)
    direct_callers = []
    if primary_node in call_graph:
        for pred in call_graph.predecessors(primary_node):
            pred_data = G.nodes.get(pred, {})
            direct_callers.append({
                "id": pred,
                "name": pred_data.get("name", pred),
                "file": pred_data.get("file", "unknown"),
                "line": pred_data.get("start_line", 0),
            })

    # 3. Transitive callers (ancestors in call graph)
    transitive_callers = []
    if primary_node in call_graph:
        ancestors = nx.ancestors(call_graph, primary_node)
        for anc in ancestors:
            if anc not in [c["id"] for c in direct_callers]:
                anc_data = G.nodes.get(anc, {})
                transitive_callers.append({
                    "id": anc,
                    "name": anc_data.get("name", anc),
                    "file": anc_data.get("file", "unknown"),
                    "line": anc_data.get("start_line", 0),
                })

    # 4. Dependent files (files importing the file containing this symbol)
    dependent_files = []
    file_node = f"file::{file_path}"
    if file_node in G:
        for importer in G.predecessors(file_node):
            edge_data = G.get_edge_data(importer, file_node)
            if edge_data and edge_data.get("relation") == "imports":
                dependent_files.append(G.nodes[importer].get("name", importer))

    # 5. Risk Assessment
    total_affected = len(direct_callers) + len(transitive_callers)
    if total_affected == 0:
        risk_level = "LOW (Isolated / Entry Point)"
    elif total_affected <= 2:
        risk_level = "MEDIUM (Localized Impact)"
    else:
        risk_level = "HIGH (Architectural Hub / High Blast Radius)"

    return {
        "status": "ok",
        "symbol": node_data.get("name", symbol_name),
        "file": file_path,
        "lines": f"{start_line}-{end_line}",
        "risk_level": risk_level,
        "direct_caller_count": len(direct_callers),
        "transitive_caller_count": len(transitive_callers),
        "direct_callers": direct_callers,
        "transitive_callers": transitive_callers,
        "dependent_files": dependent_files,
    }


def query_callers(symbol_name: str, G: nx.DiGraph | None = None) -> dict:
    """Find who calls symbol_name, and what symbol_name calls."""
    if G is None:
        G = load_graph()

    matching_nodes = _find_symbol_nodes(G, symbol_name)
    if not matching_nodes:
        return {"status": "not_found", "symbol": symbol_name}

    target = matching_nodes[0]
    call_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "calls"
    ]
    call_graph = G.edge_subgraph(call_edges)

    callers = []
    callees = []

    if target in call_graph:
        for p in call_graph.predecessors(target):
            p_data = G.nodes.get(p, {})
            callers.append(f"{p_data.get('name')} ({p_data.get('file')}:{p_data.get('start_line')})")
        for s in call_graph.successors(target):
            s_data = G.nodes.get(s, {})
            callees.append(f"{s_data.get('name')} ({s_data.get('file')}:{s_data.get('start_line')})")

    return {
        "status": "ok",
        "symbol": G.nodes[target].get("name", symbol_name),
        "file": G.nodes[target].get("file"),
        "callers": callers,
        "callees": callees,
    }


def get_topology_summary(G: nx.DiGraph | None = None) -> dict:
    """Get high-level connectivity stats of the codebase topology."""
    if G is None:
        G = load_graph()

    symbols = [n for n, d in G.nodes(data=True) if d.get("node_type") == "symbol"]
    files = [n for n, d in G.nodes(data=True) if d.get("node_type") == "file"]
    call_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "calls"]
    import_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "imports"]

    # Top called functions (highest in-degree in call graph)
    call_graph = G.edge_subgraph(call_edges)
    in_degrees = dict(call_graph.in_degree()) if call_graph.number_of_nodes() > 0 else {}
    top_called = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    top_called_formatted = [
        {"name": G.nodes[node].get("name", node), "callers": count, "file": G.nodes[node].get("file", "")}
        for node, count in top_called if count > 0
    ]

    return {
        "total_symbols": len(symbols),
        "total_files": len(files),
        "total_calls": len(call_edges),
        "total_import_edges": len(import_edges),
        "top_called_symbols": top_called_formatted,
    }
