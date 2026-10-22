from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages:       Annotated[list, add_messages]
    query:          str
    intent:         str          # "index" | "search" | "run" | "chat"
    need_run:       bool
    code_chunks:    List[dict]   # retrieved code chunks from FAISS
    explanation:    str          # LLM-generated explanation
    run_output:     str          # subprocess output
    final_response: str
