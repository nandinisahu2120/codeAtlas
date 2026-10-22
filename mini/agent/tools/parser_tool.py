"""
Universal Parser Tool
Parses any code repository using Tree-sitter (40+ languages).
Falls back to line-based chunking for unsupported file types.
"""
import json
from pathlib import Path
from config import REPO_DIR, CHUNKS_FILE, DATA_DIR, CHUNK_MAX_LINES, get_repo_dir, set_repo_dir

# ---------------------------------------------------------------------------
# Directories / extensions to always skip
# ---------------------------------------------------------------------------
SKIP_DIRS = {
    '.git', 'venv', 'env', '__pycache__', 'node_modules',
    '.next', 'dist', 'build', '.idea', '.vscode', 'coverage',
    '.mypy_cache', '.pytest_cache',
}
SKIP_EXTS = {
    '.min.js', '.min.css', '.map', '.lock', '.png', '.jpg',
    '.gif', '.svg', '.ico', '.pdf', '.zip', '.whl', '.pyc',
}
SUPPORTED_EXTS = {
    '.py', '.js', '.mjs', '.cjs',
    '.ts', '.tsx',
    '.java',
    '.go',
    '.rs',
    '.cpp', '.cc', '.cxx',
    '.c', '.h', '.hpp',
}

# ---------------------------------------------------------------------------
# Language registry  (lazy-loaded so missing grammars don't crash on import)
# ---------------------------------------------------------------------------
_LANG_MAP: dict | None = None


def _build_lang_map() -> dict:
    """
    Attempt to import each tree-sitter language package.
    Only successfully imported languages are registered.
    """
    from tree_sitter import Language
    reg = {}

    def _try(exts: list[str], import_fn, def_types: set[str]):
        try:
            lang = Language(import_fn())
            for ext in exts:
                reg[ext] = {'language': lang, 'def_types': def_types}
        except Exception:
            pass  # grammar not installed -- silently skip

    _try(['.py'], lambda: __import__('tree_sitter_python').language(),
         {'function_definition', 'async_function_definition', 'class_definition'})

    _try(['.js', '.mjs', '.cjs'], lambda: __import__('tree_sitter_javascript').language(),
         {'function_declaration', 'class_declaration', 'method_definition',
          'generator_function_declaration'})

    _try(['.ts', '.tsx'], lambda: __import__('tree_sitter_typescript').language_typescript(),
         {'function_declaration', 'class_declaration', 'method_definition',
          'interface_declaration', 'type_alias_declaration'})

    _try(['.java'], lambda: __import__('tree_sitter_java').language(),
         {'method_declaration', 'class_declaration', 'interface_declaration',
          'constructor_declaration'})

    _try(['.go'], lambda: __import__('tree_sitter_go').language(),
         {'function_declaration', 'method_declaration', 'type_declaration'})

    _try(['.rs'], lambda: __import__('tree_sitter_rust').language(),
         {'function_item', 'struct_item', 'impl_item', 'trait_item', 'enum_item'})

    _try(['.cpp', '.cc', '.cxx', '.hpp'], lambda: __import__('tree_sitter_cpp').language(),
         {'function_definition', 'class_specifier', 'struct_specifier'})

    _try(['.c', '.h'], lambda: __import__('tree_sitter_c').language(),
         {'function_definition', 'struct_specifier'})

    return reg


def _lang_map() -> dict:
    global _LANG_MAP
    if _LANG_MAP is None:
        _LANG_MAP = _build_lang_map()
    return _LANG_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_name(node) -> str:
    """Best-effort extraction of a definition node's identifier name."""
    name_node = node.child_by_field_name('name')
    if name_node:
        return name_node.text.decode('utf-8', errors='ignore')
    for child in node.children:
        if child.type in ('identifier', 'type_identifier'):
            return child.text.decode('utf-8', errors='ignore')
        # Handle Go/C spec wrappers
        if child.type in ('type_spec', 'declarator', 'function_declarator'):
            sub_name = _get_name(child)
            if sub_name != '<anonymous>':
                return sub_name
    return '<anonymous>'


def _walk(node, def_types: set, source_bytes: bytes, rel_path: str,
          lang_label: str, chunks: list):
    if node.type in def_types:
        start_line = node.start_point[0] + 1
        end_line   = node.end_point[0] + 1
        code = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

        if len(code.splitlines()) <= CHUNK_MAX_LINES:
            name = _get_name(node)
            chunks.append({
                'id':         f'{rel_path}::{name}@{start_line}',
                'file':       rel_path,
                'start_line': start_line,
                'end_line':   end_line,
                'type':       node.type,
                'name':       name,
                'code':       code,
                'docstring':  '',
                'text':       f'{name}\n{code}',   # combined for better embedding
                'language':   lang_label,
            })

    for child in node.children:
        _walk(child, def_types, source_bytes, rel_path, lang_label, chunks)


def _parse_with_treesitter(file_path: Path, source: str, lang_config: dict, root: Path) -> list[dict]:
    from tree_sitter import Parser
    parser = Parser(lang_config['language'])
    source_bytes = source.encode('utf-8')
    tree = parser.parse(source_bytes)
    rel_path  = file_path.relative_to(root).as_posix()
    lang_label = file_path.suffix.lstrip('.')
    chunks: list[dict] = []
    _walk(tree.root_node, lang_config['def_types'], source_bytes, rel_path, lang_label, chunks)
    return chunks


def _parse_fallback(file_path: Path, source: str, root: Path) -> list[dict]:
    """Line-window chunking for file types with no grammar installed."""
    rel_path = file_path.relative_to(root).as_posix()
    lines    = source.splitlines()
    chunks   = []
    step     = 40
    for i in range(0, len(lines), step):
        block = lines[i: i + step]
        code  = '\n'.join(block)
        chunks.append({
            'id':         f'{rel_path}::chunk@{i + 1}',
            'file':       rel_path,
            'start_line': i + 1,
            'end_line':   i + len(block),
            'type':       'chunk',
            'name':       f'chunk_{i + 1}',
            'code':       code,
            'docstring':  '',
            'text':       code,
            'language':   file_path.suffix.lstrip('.') or 'unknown',
        })
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_repo(repo_path: str | None = None) -> dict:
    """
    Walk a repository and extract code chunks using Tree-sitter.
    Saves results to data/chunks.json and returns a status dict.
    Supports local paths as well as remote GitHub repository URLs.
    """
    from agent.tools.repo_loader import clone_or_load_repo

    if repo_path:
        load_res = clone_or_load_repo(repo_path)
        if load_res["status"] != "ok":
            return load_res
        root = load_res["path"]
    else:
        root = get_repo_dir()

    if not root.exists():
        return {'status': 'error', 'message': f'Path not found: {root}'}

    lmap          = _lang_map()
    all_chunks    = []
    files_parsed  = 0
    files_skipped = 0

    for f in sorted(root.rglob('*')):
        if not f.is_file():
            continue
        # Skip noise directories
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        # Skip binary / minified / lock files
        if any(f.name.endswith(e) for e in SKIP_EXTS):
            continue
        # Only process known code extensions
        if f.suffix not in SUPPORTED_EXTS:
            continue

        try:
            source = f.read_text(encoding='utf-8', errors='ignore').strip()
            if not source:
                continue

            if f.suffix in lmap:
                chunks = _parse_with_treesitter(f, source, lmap[f.suffix], root)
            else:
                chunks = _parse_fallback(f, source, root)

            all_chunks.extend(chunks)
            files_parsed += 1
        except Exception:
            files_skipped += 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")

    return {
        'status':              'ok',
        'chunks':              len(all_chunks),
        'files_parsed':        files_parsed,
        'files_skipped':       files_skipped,
        'languages_supported': list(lmap.keys()),
        'message':             f'Parsed {len(all_chunks)} chunks from {files_parsed} files  '
                               f'(skipped {files_skipped})',
    }


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        return []
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
