"""
LangGraph Agent Graph -- The brain of the system.
Routes user messages through Parser, Search, Explain, and Runner nodes.
Uses keyword-based intent routing (fast, reliable, no LLM overhead).
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState
from agent.tools.parser_tool import parse_repo, load_chunks
from agent.tools.indexer_tool import build_index, load_index
from agent.tools.search_tool import search, get_token_contributions
from agent.tools.explain_tool import explain_code, find_bugs, suggest_improvements, chat, answer_query
from agent.tools.runner_tool import run_file
from config import REPO_DIR, get_repo_dir, set_repo_dir
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# Intent Router (keyword-based, no LLM needed)
# ---------------------------------------------------------------------------

INDEX_PATTERNS   = re.compile(r"\b(index(?:ing|es)?|parse|parsing|load|scan|re-?index|rebuild|clone|github\.com)\b", re.I)
RUN_PATTERNS     = re.compile(r"\b(run(?:s|ning)?|exec(?:ute|uting)?|output|result of|execute\w*)\b", re.I)
BUG_PATTERNS     = re.compile(r"\b(bugs?|errors?|issues?|fix(?:es)?|problems?|wrong|crash(?:es)?|fail(?:s|ures?|ing)?|vulnerab\w*)\b", re.I)
IMPROVE_PATTERNS = re.compile(r"\b(improv(?:e|ements?|ing)|refactor(?:ing)?|optimi[zs](?:e|ation|ing)|clean(?:up)?|better|suggest(?:ions?)?)\b", re.I)
SEARCH_PATTERNS  = re.compile(
    r"\b(find|where|which|search(?:ing)?|locate|show|what|explain(?:ing|ation)?|how|why|describe|does|tell|about|language|languages|stack|frameworks?)\b", re.I
)

OVERVIEW_PATTERNS = re.compile(
    r"\b("
    r"what (does|is) this (repo|project|codebase)|"
    r"about this (repo|project|codebase)|"
    r"repo overview|codebase overview|project overview|"
    r"summary of (this|the) (repo|codebase|project)|"
    r"summarize (this|the) (repo|codebase|project)|"
    r"what files|list files|what are the files|"
    r"tell me about this (repo|project|codebase)|"
    r"(languages?|tech stack|technologies|frameworks|technology)\b.*(repo|project|codebase|written|used)?|"
    r"(what|which) language"
    r")\b",
    re.I,
)

GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|who\s+are\s+you|how\s+are\s+you)\b", re.I
)

IMPACT_PATTERNS = re.compile(
    r"\b("
    r"impact|blast radius|"
    r"who calls|what calls|who invokes|who uses|"
    r"what does .* call|callees?|callers?|"
    r"which functions? break|break if i change|change (this|the) signature|"
    r"call graph|dependency graph|topology|dependencies of|depend on"
    r")\b",
    re.I,
)


def _detect_intent(query: str) -> str:
    cleaned = query.strip()
    if GREETING_PATTERNS.search(cleaned):
        return "chat"
    if IMPACT_PATTERNS.search(cleaned):
        return "graph"
    if OVERVIEW_PATTERNS.search(cleaned):
        return "overview"
    if INDEX_PATTERNS.search(cleaned):
        return "index"
    if RUN_PATTERNS.search(cleaned):
        return "run"
    if BUG_PATTERNS.search(cleaned):
        return "bug"
    if IMPROVE_PATTERNS.search(cleaned):
        return "improve"
    if SEARCH_PATTERNS.search(cleaned):
        return "search"
    return "chat"


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """Detect intent from the latest user message."""
    query = state["query"]
    intent = _detect_intent(query)
    return {"intent": intent}


def index_node(state: AgentState) -> dict:
    """Parse the repo and build the FAISS index."""
    query = state.get("query", "")
    target = None

    # Check for github URL in query
    gh_match = re.search(r"(https?://github\.com/[^\s]+|git@github\.com:[^\s]+|github\.com/[^\s]+)", query)
    if gh_match:
        target = gh_match.group(1).rstrip(".,;")
    else:
        # Check entire remaining string after command keyword (handles Windows paths with spaces)
        remainder = re.sub(r"^(?:index|load|clone|parse)\s+", "", query.strip(), flags=re.I).strip().strip('"\'')
        if remainder:
            candidate = Path(remainder).expanduser().resolve()
            if candidate.exists() and candidate.is_dir():
                target = str(candidate)

        if not target:
            # Check if a single token path was given
            parts = query.split()
            for p in parts[1:]:
                p_clean = p.strip('"\'')
                candidate = Path(p_clean).expanduser().resolve()
                if candidate.exists() and candidate.is_dir():
                    target = str(candidate)
                    break

    repo_path = target if target else str(get_repo_dir())

    parse_result = parse_repo(repo_path)
    if parse_result["status"] != "ok":
        return {"final_response": f"Parsing failed: {parse_result['message']}"}

    chunks = load_chunks()
    index_result = build_index(chunks)
    if index_result["status"] != "ok":
        return {"final_response": f"Indexing failed: {index_result['message']}"}

    return {
        "final_response": (
            f"Done. {parse_result['message']}. "
            f"{index_result['message']}. "
            f"Ask me anything about this codebase."
        )
    }


def search_node(state: AgentState) -> dict:
    """Semantic search over the indexed code."""
    result = search(state["query"])
    if result["status"] != "ok":
        return {"code_chunks": [], "final_response": result["message"]}
    return {"code_chunks": result["results"]}


def overview_node(state: AgentState) -> dict:
    """Provide a synthesized overview of the active repository structure and contents."""
    chunks = load_chunks()
    active_repo = get_repo_dir()
    if not chunks:
        return {"final_response": f"Repository `{active_repo.name}` is not indexed yet. Type `index this repo` to begin."}

    files_map = {}
    lang_counts = {}
    for c in chunks:
        f = c["file"]
        if f not in files_map:
            files_map[f] = []
        name = c.get("name", "")
        if name and name != "<anonymous>":
            files_map[f].append(name)
        lang = c.get("language") or "unknown"
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    lang_summary = ", ".join(f"{k} ({v} chunks)" for k, v in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True))

    overview_lines = [
        f"Repository: {active_repo.name} ({len(files_map)} files, {len(chunks)} chunks)",
        f"Languages: {lang_summary}",
    ]
    for f, symbols in list(files_map.items())[:30]:
        sym_str = f" [declarations: {', '.join(symbols[:5])}]" if symbols else ""
        overview_lines.append(f"- {f}{sym_str}")

    context_str = "\n".join(overview_lines)
    answer = answer_query(state["query"], context=context_str)

    # If user specifically asked about language/stack, return the direct answer with clean language breakdown
    if re.search(r"\b(languages?|tech stack|technologies|frameworks?|stack|written in)\b", state["query"], re.I):
        return {
            "final_response": (
                f"{answer}\n\n"
                f"**Detected Languages in `{active_repo.name}`:**\n"
                + "\n".join(f"- `{k}`: {v} chunks" for k, v in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True))
            )
        }

    parts = [answer, f"\n**Languages:** {lang_summary}", "\n**Indexed Files:**"]
    for f in sorted(files_map.keys())[:25]:
        parts.append(f"- `{f}` ({len(files_map[f])} symbols)")
    if len(files_map) > 25:
        parts.append(f"- ... and {len(files_map) - 25} more files")

    return {"final_response": "\n".join(parts)}


def explain_node(state: AgentState) -> dict:
    """Directly answer user questions using relevant code snippets."""
    chunks = state.get("code_chunks", [])
    query  = state["query"]

    # Filter to genuinely relevant chunks (score >= 0.22)
    relevant = [c for c in chunks if c.get("score", 0) >= 0.22]

    if not relevant:
        # Score too low -- answer conversational or general codebase question
        all_chunks = load_chunks()
        files = sorted(set(c["file"] for c in all_chunks))
        lang_counts = {}
        for c in all_chunks:
            lang = c.get("language") or "unknown"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        lang_summary = ", ".join(f"{k} ({v} chunks)" for k, v in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True))
        repo_info = f"Languages in active repo: {lang_summary}\nFiles in active repo ({len(files)} total): {', '.join(files[:30])}" if files else ""
        response = answer_query(query, context=repo_info)
        return {"explanation": response, "final_response": response}

    # Format top 1-2 chunks for context
    context_blocks = []
    for c in relevant[:2]:
        context_blocks.append(
            f"File: {c['file']} (lines {c['start_line']}-{c['end_line']}):\n"
            f"```{c.get('language', '')}\n{c['code']}\n```"
        )
    combined_ctx = "\n\n".join(context_blocks)
    top = relevant[0]
    file_info = f"{top['file']} (lines {top['start_line']}-{top['end_line']})"

    answer = answer_query(query, context=combined_ctx, file_info=file_info)

    # Token contribution explainability
    contributions = get_token_contributions(query, top["code"])
    contrib_str = ", ".join(
        f"'{c['token']}' ({c['contribution']:.2f})" for c in contributions[:4]
    )

    parts = [answer]
    parts.append("\n**Matched Code:**")
    for c in relevant[:2]:
        parts.append(f"`{c['file']}` (lines {c['start_line']}-{c['end_line']}, score: {c['score']:.2f})")
        parts.append(f"```{c.get('language', '')}\n{c['code']}\n```")

    if contrib_str:
        parts.append(f"\n*Query token attribution: {contrib_str}*")

    return {"explanation": answer, "final_response": "\n".join(parts)}


def bug_node(state: AgentState) -> dict:
    """Search for relevant code, then analyze it for bugs."""
    chunks = state.get("code_chunks", [])
    if not chunks:
        return {"final_response": "No code found to analyze. Try indexing the repo first."}

    top = chunks[0]
    file_info = f"{top['file']} (lines {top['start_line']}-{top['end_line']})"
    analysis = find_bugs(top["code"], file_info)

    parts = [analysis, ""]
    parts.append(f"--- Analyzed Code ---")
    parts.append(f"[{top['file']}  lines {top['start_line']}-{top['end_line']}]")
    parts.append(f"```{top.get('language', '')}")
    parts.append(top["code"])
    parts.append("```")

    return {"final_response": "\n".join(parts)}


def improve_node(state: AgentState) -> dict:
    """Search for relevant code, then suggest improvements."""
    chunks = state.get("code_chunks", [])
    if not chunks:
        return {"final_response": "No code found to improve. Try indexing the repo first."}

    top = chunks[0]
    file_info = f"{top['file']} (lines {top['start_line']}-{top['end_line']})"
    suggestions = suggest_improvements(top["code"], file_info)

    parts = [suggestions, ""]
    parts.append(f"--- Original Code ---")
    parts.append(f"[{top['file']}  lines {top['start_line']}-{top['end_line']}]")
    parts.append(f"```{top.get('language', '')}")
    parts.append(top["code"])
    parts.append("```")

    return {"final_response": "\n".join(parts)}


def run_node(state: AgentState) -> dict:
    """Extract a filename from the query and execute it."""
    query = state["query"]
    # Try to find a .py filename in the query
    match = re.search(r'[\w/\\]+\.py', query)
    if not match:
        # If no filename found, search for relevant code and report
        chunks = state.get("code_chunks", [])
        if chunks:
            return {"final_response": f"Which file should I run? I found: {chunks[0]['file']}"}
        return {"final_response": "Please specify which .py file to run."}

    filename = match.group()
    result = run_file(filename)
    parts = [f"Ran: `{filename}`", f"Exit code: {result['exit_code']}"]
    if result["stdout"]:
        parts.append(f"\n--- stdout ---\n```\n{result['stdout']}\n```")
    if result["stderr"]:
        parts.append(f"\n--- stderr ---\n```\n{result['stderr']}\n```")
    if result.get("timed_out"):
        parts.append(f"\n(Timed out after {result.get('timeout', '?')}s)")

    return {"run_output": result["stdout"], "final_response": "\n".join(parts)}


def chat_node(state: AgentState) -> dict:
    """General chat -- if question refers to the repo or codebase, provide repository context."""
    query = state["query"]
    chunks = load_chunks()
    if chunks and re.search(r"\b(repo|repository|codebase|project|code|files?|languages?|stack)\b", query, re.I):
        active_repo = get_repo_dir()
        lang_counts = {}
        for c in chunks:
            lang = c.get("language") or "unknown"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        lang_summary = ", ".join(f"{k} ({v} chunks)" for k, v in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True))
        files = sorted(set(c["file"] for c in chunks))
        repo_info = (
            f"Active Repository: {active_repo.name}\n"
            f"Languages: {lang_summary}\n"
            f"Files in active repo ({len(files)} total): {', '.join(files[:25])}"
        )
        response = answer_query(query, context=repo_info)
    else:
        response = chat(query)
    return {"final_response": response}


def graph_analysis_node(state: AgentState) -> dict:
    """Analyze AST call graph, caller-callee relationships, and impact blast-radius."""
    from agent.tools.graph_tool import analyze_impact, get_topology_summary
    query = state.get("query", "")

    # Extract target symbol from natural language query
    patterns = [
        r"(?:break\s+if\s+(?:i|we)\s+(?:change|modify))\s+[`'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]?",
        r"(?:change|modify|update|refactor)\s+(?:the\s+)?(?:signature\s+of\s+)?(?:function\s+|method\s+)?[`'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]?",
        r"(?:who calls|what calls|who invokes|who uses)\s+[`'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]?",
        r"(?:callers of|callees of|impact of)\s+[`'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]?",
        r"[`'\"]([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]",
    ]
    target_symbol = None
    stopwords = {"if", "i", "we", "this", "the", "a", "an", "function", "functions", "signature", "method", "it", "my", "code"}
    for p in patterns:
        m = re.search(p, query, re.I)
        if m:
            candidate = m.group(1).strip()
            if candidate.lower() not in stopwords:
                target_symbol = candidate
                break

    # If general topology query (e.g. "show call graph", "codebase topology")
    if not target_symbol or target_symbol.lower() in ("graph", "topology", "callgraph", "summary"):
        summary = get_topology_summary()
        res = [
            "### Codebase AST Call Graph & Dependency Topology\n",
            f"- **Indexed Symbols (Functions/Classes):** {summary['total_symbols']}",
            f"- **Indexed Source Files:** {summary['total_files']}",
            f"- **Detected Call Edges (Caller -> Callee):** {summary['total_calls']}",
            f"- **File Import Dependencies:** {summary['total_import_edges']}\n",
        ]
        if summary.get("top_called_symbols"):
            res.append("**Most Central / Frequently Called Functions:**")
            for item in summary["top_called_symbols"]:
                res.append(f"- `{item['name']}` ({item['file']}) — invoked by **{item['callers']}** callers")
        res.append("\n*Tip: Ask 'Which functions break if I change <function_name>?' for impact analysis.*")
        return {"final_response": "\n".join(res)}

    # Perform impact analysis for target_symbol
    impact = analyze_impact(target_symbol)
    if impact.get("status") == "not_found":
        return {
            "final_response": (
                f"### Impact Analysis: `{target_symbol}`\n\n"
                f"Symbol `{target_symbol}` was not directly found in the indexed AST definition table.\n"
                f"Ensure the repository is indexed and check spelling of the function or method name."
            )
        }

    res = [
        f"### Impact & Blast Radius Analysis: `{impact['symbol']}`\n",
        f"- **Definition Location:** `{impact['file']}` (lines {impact['lines']})",
        f"- **Architectural Risk Level:** **{impact['risk_level']}**",
        f"- **Direct Callers:** {impact['direct_caller_count']}",
        f"- **Transitive Callers (Blast Radius):** {impact['transitive_caller_count']}\n",
    ]

    if impact["direct_callers"]:
        res.append("**Direct Callers (Will break if signature changes):**")
        for c in impact["direct_callers"]:
            res.append(f"- `{c['name']}` in `{c['file']}` (line {c['line']})")
        res.append("")

    if impact["transitive_callers"]:
        res.append("**Transitive Dependency Chain (Secondary impact):**")
        for c in impact["transitive_callers"]:
            res.append(f"- `{c['name']}` in `{c['file']}` (line {c['line']})")
        res.append("")

    if impact["dependent_files"]:
        res.append("**Files Importing This Module:**")
        for f in impact["dependent_files"]:
            res.append(f"- `{f}`")
        res.append("")

    if not impact["direct_callers"] and not impact["transitive_callers"]:
        res.append("*No internal callers detected. This appears to be an entry point, public API endpoint, or top-level handler.*")
    else:
        res.append("**Refactoring Precautions:**")
        res.append("1. Maintain backward-compatibility or provide default parameter values.")
        res.append("2. Update call sites in the direct caller functions listed above.")
        res.append("3. Execute integration/unit tests covering the transitive dependency chain before committing.")

    return {"final_response": "\n".join(res)}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_router(state: AgentState) -> str:
    intent = state.get("intent", "chat")
    if intent == "index":
        return "index_node"
    if intent == "graph":
        return "graph_node"
    if intent == "overview":
        return "overview_node"
    if intent in ("search",):
        return "search_node"
    if intent == "bug":
        return "search_then_bug"
    if intent == "improve":
        return "search_then_improve"
    if intent == "run":
        return "search_then_run"
    return "chat_node"


def route_after_search(state: AgentState) -> str:
    return "explain_node"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    # Add nodes
    g.add_node("router",        router_node)
    g.add_node("index_node",    index_node)
    g.add_node("overview_node", overview_node)
    g.add_node("search_node",   search_node)
    g.add_node("explain_node",  explain_node)
    g.add_node("bug_node",      bug_node)
    g.add_node("improve_node",  improve_node)
    g.add_node("run_node",      run_node)
    g.add_node("graph_node",    graph_analysis_node)
    g.add_node("chat_node",     chat_node)

    # For bug/improve/run: search first, then specialized node
    g.add_node("search_then_bug",     search_node)
    g.add_node("search_then_improve", search_node)
    g.add_node("search_then_run",     search_node)

    # Entry
    g.set_entry_point("router")

    # Router -> intent-based routing
    g.add_conditional_edges("router", route_after_router, {
        "index_node":          "index_node",
        "overview_node":       "overview_node",
        "search_node":         "search_node",
        "search_then_bug":     "search_then_bug",
        "search_then_improve": "search_then_improve",
        "search_then_run":     "search_then_run",
        "graph_node":          "graph_node",
        "chat_node":           "chat_node",
    })

    # overview -> END
    g.add_edge("overview_node", END)

    # graph -> END
    g.add_edge("graph_node", END)

    # search -> explain -> END
    g.add_edge("search_node", "explain_node")
    g.add_edge("explain_node", END)

    # search -> bug -> END
    g.add_edge("search_then_bug", "bug_node")
    g.add_edge("bug_node", END)

    # search -> improve -> END
    g.add_edge("search_then_improve", "improve_node")
    g.add_edge("improve_node", END)

    # search -> run -> END
    g.add_edge("search_then_run", "run_node")
    g.add_edge("run_node", END)

    # index -> END
    g.add_edge("index_node", END)

    # chat -> END
    g.add_edge("chat_node", END)

    # Compile with memory
    memory = MemorySaver()
    return g.compile(checkpointer=memory)


# Singleton
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def invoke(query: str, thread_id: str = "default") -> str:
    """
    Main entry point. Send a query, get a response.
    Maintains conversation memory via thread_id.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {
            "query": query,
            "messages": [{"role": "user", "content": query}],
            "intent": "",
            "need_run": False,
            "code_chunks": [],
            "explanation": "",
            "run_output": "",
            "final_response": "",
        },
        config=config,
    )
    return result.get("final_response", "I could not generate a response.")
