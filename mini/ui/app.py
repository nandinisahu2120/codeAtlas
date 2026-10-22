"""
Streamlit Chat UI for the Code Agent.
Run with: streamlit run ui/app.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"

import re
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from agent.graph import invoke
from agent.tools.indexer_tool import load_index
from agent.tools.parser_tool import load_chunks
from agent.tools.repo_loader import is_github_url
from agent.tools.explain_tool import preload_model
from config import REPO_DIR, get_repo_dir, set_repo_dir
import threading

if "model_warmed" not in st.session_state:
    st.session_state.model_warmed = True
    threading.Thread(target=preload_model, daemon=True).start()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="codeAtlas",
    page_icon="//",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }

/* Dark background */
.stApp {
    background: #0a0b0f;
}

/* Top bar */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    background: linear-gradient(135deg, #12131a 0%, #1a1b26 100%);
    border-bottom: 1px solid #2a2b3d;
    border-radius: 0 0 16px 16px;
    margin-bottom: 1rem;
}
.top-bar h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 600;
    color: #c8d3f5;
    margin: 0;
    letter-spacing: 0.05em;
}
.top-bar .stats {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #636da6;
    letter-spacing: 0.02em;
}
.top-bar .dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-green { background: #9ece6a; }
.dot-red   { background: #f7768e; }

/* Chat messages */
.stChatMessage {
    background: #14151f !important;
    border: 1px solid #1e1f2e !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stChatMessage p, .stChatMessage li {
    color: #a9b1d6 !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
}
.stChatMessage code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    background: #1a1b2e !important;
    color: #7dcfff !important;
    padding: 2px 5px !important;
    border-radius: 4px !important;
}
.stChatMessage pre {
    background: #1a1b2e !important;
    border: 1px solid #2a2b3d !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
.stChatMessage pre code {
    color: #c8d3f5 !important;
    background: transparent !important;
    padding: 0 !important;
}

/* Chat input */
.stChatInput > div {
    background: #14151f !important;
    border: 1px solid #2a2b3d !important;
    border-radius: 12px !important;
}
.stChatInput textarea {
    font-family: 'DM Sans', sans-serif !important;
    color: #c8d3f5 !important;
    font-size: 0.92rem !important;
}
.stChatInput textarea::placeholder {
    color: #4a4f6e !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1017;
    border-right: 1px solid #1e1f2e;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #636da6;
    font-size: 0.82rem;
}

/* Button */
.stButton > button {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #c8d3f5;
    background: linear-gradient(135deg, #1a1b2e 0%, #24253a 100%);
    border: 1px solid #2a2b3d;
    border-radius: 8px;
    padding: 0.4rem 1rem;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #24253a 0%, #2e2f4a 100%);
    border-color: #7aa2f7;
    color: #7aa2f7;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(122, 162, 247, 0.15);
}

/* Spinner */
.stSpinner > div {
    border-top-color: #7aa2f7 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session-1"


# ---------------------------------------------------------------------------
# Top bar with status
# ---------------------------------------------------------------------------
active_repo = get_repo_dir()
index, meta = load_index()
chunks = load_chunks()
indexed = index is not None and index.ntotal > 0
chunk_count = index.ntotal if indexed else 0
file_count = len(set(c["file"] for c in chunks)) if chunks else 0
repo_name = active_repo.name

dot_class = "dot-green" if indexed else "dot-red"
status_text = f"{chunk_count} chunks / {file_count} files indexed" if indexed else "not indexed"

st.markdown(f"""
<div class="top-bar">
    <h1>// codeAtlas</h1>
    <div class="stats">
        <span class="dot {dot_class}"></span>
        {repo_name} &mdash; {status_text}
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Repository Source")
    repo_input = st.text_input(
        "GitHub URL or Local Path",
        value=str(active_repo),
        placeholder="https://github.com/owner/repo or /path/to/folder",
        help="Paste any public GitHub URL (e.g. https://github.com/psf/requests) or a local directory",
        key="repo_path_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load & Index", use_container_width=True):
            cleaned_input = repo_input.strip()
            if not cleaned_input:
                st.error("Please provide a repository URL or local path.")
            else:
                action_label = "Cloning and indexing" if is_github_url(cleaned_input) else "Indexing"
                with st.spinner(f"{action_label} {cleaned_input}..."):
                    response = invoke(f"index {cleaned_input}", st.session_state.thread_id)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    **Try any repository:**
    - Paste GitHub URLs like: `https://github.com/pallets/flask`
    - Or type in chat: `load https://github.com/psf/requests`
    - Search: "where is authentication handled?"
    - Explain: "explain calculate_average"
    - Bugs: "find bugs in the codebase"
    - Improve: "how can I improve this code?"
    - Run: "run run_tests.py"
    """)


# ---------------------------------------------------------------------------
# Repository Connection Card (Main Screen)
# ---------------------------------------------------------------------------
with st.expander("Switch or Connect Another Repository", expanded=not indexed):
    st.markdown("Enter any public **GitHub repository URL** or a **local folder path**:")
    main_repo_col, btn_col = st.columns([4, 1])
    with main_repo_col:
        main_repo_input = st.text_input(
            "Repository Source",
            value="",
            placeholder="https://github.com/pallets/flask",
            label_visibility="collapsed",
            key="main_repo_input_box",
        )
    with btn_col:
        if st.button("Load Repo", use_container_width=True, key="main_load_btn"):
            if main_repo_input.strip():
                source = main_repo_input.strip()
                action_label = "Cloning and indexing" if is_github_url(source) else "Indexing"
                with st.spinner(f"{action_label} {source}..."):
                    response = invoke(f"index {source}", st.session_state.thread_id)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()


# ---------------------------------------------------------------------------
# Welcome message
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    welcome = (
        f"I am **codeAtlas**, your private code agent.\n\n"
        f"- **Active Codebase**: `{active_repo}`\n"
        f"- **Index Status**: {f'{chunk_count} code chunks across {file_count} files indexed' if indexed else 'Not indexed yet'}\n\n"
        f"You can analyze **any GitHub repository** (e.g. `https://github.com/psf/requests`) or local folder by entering it in the repository box above or typing `load <github-url>` directly in chat."
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome})


# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask about your codebase or paste a GitHub URL..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # If user pasted a GitHub URL directly, normalize to index command
    cleaned_prompt = prompt.strip()
    if is_github_url(cleaned_prompt) and not cleaned_prompt.lower().startswith(("index", "load", "clone")):
        query_to_send = f"index {cleaned_prompt}"
        is_index_action = True
    else:
        query_to_send = prompt
        is_index_action = bool(re.search(r"\b(index|load|clone)\b", prompt, re.I))

    # Get agent response
    with st.chat_message("assistant"):
        spinner_msg = "Cloning and indexing repository..." if is_index_action else "Thinking..."
        with st.spinner(spinner_msg):
            response = invoke(query_to_send, st.session_state.thread_id)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    if is_index_action:
        st.rerun()
