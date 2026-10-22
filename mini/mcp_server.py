"""
Model Context Protocol (MCP) Server for codeAtlas.
Implements the JSON-RPC 2.0 stdio transport protocol.
Exposes repository indexing, semantic vector search, token explainability, and code execution.
"""
import sys
import json
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tools.repo_loader import clone_or_load_repo
from agent.tools.parser_tool import parse_repo, load_chunks
from agent.tools.indexer_tool import build_index
from agent.tools.search_tool import search, get_token_contributions
from agent.tools.explain_tool import explain_code, answer_query
from agent.tools.runner_tool import run_file
from config import get_repo_dir

SERVER_NAME = "codeAtlas-mcp-server"
SERVER_VERSION = "1.0.0"

TOOLS_REGISTRY = [
    {
        "name": "index_repository",
        "description": "Clone and parse a GitHub repository or local directory using Tree-sitter AST, then index embeddings with FAISS.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "GitHub repository URL (e.g. 'https://github.com/user/repo') or local directory path."
                }
            },
            "required": ["source"]
        }
    },
    {
        "name": "search_code",
        "description": "Perform semantic vector similarity search over the indexed codebase using MiniLM transformer embeddings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query about the code (e.g. 'where is authentication handled')."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top matching code chunks to return.",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "explain_code_match",
        "description": "Get detailed explanation and token-attribution scoring (SHAP style) for why a code snippet matches a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The natural language query."},
                "code": {"type": "string", "description": "The code snippet."},
                "file_path": {"type": "string", "description": "Relative file path of the snippet."}
            },
            "required": ["query", "code"]
        }
    },
    {
        "name": "get_repo_overview",
        "description": "Get a manifest of all indexed files, languages, and detected function/class symbols in the active repository.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "run_python_file",
        "description": "Execute a Python script in a sandboxed subprocess and return stdout, stderr, and exit code.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative or absolute path to the Python file."
                }
            },
            "required": ["file_path"]
        }
    }
]


def handle_initialize(req_id):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION
            },
            "capabilities": {
                "tools": {}
            }
        }
    }


def handle_tools_list(req_id):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "tools": TOOLS_REGISTRY
        }
    }


def handle_tools_call(req_id, params):
    tool_name = params.get("name")
    args = params.get("arguments", {})

    try:
        if tool_name == "index_repository":
            src = args.get("source", "")
            clone_res = clone_or_load_repo(src)
            if clone_res["status"] != "ok":
                res = clone_res
            else:
                parse_res = parse_repo(str(clone_res["path"]))
                chunks = load_chunks()
                idx_res = build_index(chunks)
                res = {
                    "status": "ok",
                    "repository": clone_res["name"],
                    "chunks": parse_res.get("chunks", 0),
                    "files": parse_res.get("files_parsed", 0),
                    "message": f"Successfully indexed {parse_res.get('chunks', 0)} chunks."
                }

        elif tool_name == "search_code":
            query = args.get("query", "")
            top_k = int(args.get("top_k", 3))
            res = search(query, top_k=top_k)

        elif tool_name == "explain_code_match":
            query = args.get("query", "")
            code = args.get("code", "")
            file_path = args.get("file_path", "")
            attribution = get_token_contributions(query, code)
            explanation = explain_code(query=query, code=code, file_info=file_path)
            res = {
                "status": "ok",
                "explanation": explanation,
                "token_contributions": attribution[:10]
            }

        elif tool_name == "get_repo_overview":
            chunks = load_chunks()
            active_repo = get_repo_dir()
            files_map = {}
            for c in chunks:
                f = c["file"]
                if f not in files_map:
                    files_map[f] = []
                name = c.get("name", "")
                if name and name != "<anonymous>":
                    files_map[f].append(name)

            res = {
                "status": "ok",
                "repository": active_repo.name,
                "path": str(active_repo),
                "total_chunks": len(chunks),
                "files": [
                    {"file": f, "symbol_count": len(syms), "symbols": syms[:10]}
                    for f, syms in sorted(files_map.items())
                ]
            }

        elif tool_name == "run_python_file":
            fp = args.get("file_path", "")
            res = run_file(fp)

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(res, indent=2)
                    }
                ]
            }
        }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32000,
                "message": f"Error executing {tool_name}: {str(e)}",
                "data": traceback.format_exc()
            }
        }


def run_stdio_server():
    """Stdio loop processing JSON-RPC messages from stdin to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "initialize":
                resp = handle_initialize(req_id)
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                resp = handle_tools_list(req_id)
            elif method == "tools/call":
                resp = handle_tools_call(req_id, params)
            elif method == "ping":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not supported: {method}"}
                }

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
            }
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
