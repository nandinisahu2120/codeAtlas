"""
Explain Tool -- Uses the local HuggingFace LLM (Qwen2.5-1.5B) to generate
natural-language explanations, bug reports, and improvement suggestions.
Fully offline, no API calls.
"""
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import platform
import threading

try:
    torch.set_num_threads(4)
except Exception:
    pass

from config import LLM_MODEL, MAX_NEW_TOKENS

_pipe = None
_pipe_lock = threading.Lock()
_loading_status = "unloaded"


def _get_pipe():
    """Lazy-load the local LLM pipeline with thread safety and OS-specific optimizations."""
    global _pipe, _loading_status
    if _pipe is not None:
        return _pipe

    with _pipe_lock:
        if _pipe is not None:
            return _pipe

        _loading_status = "loading"
        print(f"[codeAtlas] Initializing local LLM: {LLM_MODEL} ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            # Windows x86 CPU does not natively accelerate bfloat16; use float32 to avoid severe slowdowns/hangs
            is_win = platform.system() == "Windows"
            model_dtype = torch.float32 if is_win else torch.bfloat16

            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                dtype=model_dtype,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )

            pad_id = tokenizer.eos_token_id
            _pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=pad_id,
            )
            _loading_status = "ready"
            print("[codeAtlas] Local LLM loaded successfully.")
        except Exception as e:
            _loading_status = f"error: {e}"
            print(f"[codeAtlas] Warning: Failed to load local LLM: {e}")
            raise e

    return _pipe


def preload_model():
    """Warm up the pipeline so queries do not suffer model load latency."""
    try:
        return _get_pipe()
    except Exception as e:
        print(f"[codeAtlas] Preload warning: {e}")
        return None


# ---------------------------------------------------------------------------
# System prompts for different intents
# ---------------------------------------------------------------------------

QA_SYSTEM = """You are an expert codebase assistant.
Answer the user's question directly, accurately, and concisely based on the codebase context provided.
- Directly answer the specific question asked without robotic preambles or generic templates.
- Reference specific file names, function/class names, and line numbers when applicable.
- Never write meta-commentary like 'What the code does' or 'Why it is relevant to the question'.
- If the provided code does not answer the question, state that clearly and briefly."""

BUG_SYSTEM = """You are a senior code reviewer. Analyze the given code for:
1. Bugs, logic errors, or edge cases
2. Security vulnerabilities
3. Performance issues
If no bugs are found, say so clearly. Be specific about line numbers and fixes."""

IMPROVE_SYSTEM = """You are a code quality advisor. Suggest concrete improvements for:
1. Readability and naming
2. Performance optimization
3. Best practices for the language
Be specific. Show the improved version where possible."""

CHAT_SYSTEM = """You are a helpful coding assistant. You have access to a codebase and can answer
general programming questions. Be concise and accurate."""


def _build_prompt(system: str, query: str, code: str = "", file_info: str = "") -> str:
    """Build a chat-style prompt for Qwen2.5-Instruct."""
    parts = [f"<|im_start|>system\n{system}<|im_end|>"]

    user_msg = query
    if file_info:
        user_msg += f"\n\nFile Info:\n{file_info}"
    if code:
        user_msg += f"\n\nCode Context:\n```\n{code}\n```"

    parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _generate(prompt: str) -> str:
    try:
        pipe = _get_pipe()
        out  = pipe(prompt, return_full_text=False)
        text = out[0]["generated_text"].strip()
        # Clean trailing special tokens
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            text = text.replace(tok, "").strip()
        return text
    except Exception as e:
        return (
            f"The local AI model (Qwen2.5-1.5B) is currently downloading or initializing in the terminal.\n\n"
            f"**Status detail:** `{e}`\n\n"
            f"*Check the Command Prompt/Terminal window running codeAtlas to see the progress.*"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_query(query: str, context: str = "", file_info: str = "") -> str:
    """Answer a user question directly using the provided codebase context."""
    prompt = _build_prompt(QA_SYSTEM, query, code=context, file_info=file_info)
    return _generate(prompt)


def explain_code(query: str, code: str, file_info: str = "") -> str:
    """Answer code questions directly using code context."""
    return answer_query(query, context=code, file_info=file_info)


def find_bugs(code: str, file_info: str = "") -> str:
    """Analyze code for bugs and vulnerabilities."""
    prompt = _build_prompt(BUG_SYSTEM, "Find bugs and issues in this code.", code, file_info)
    return _generate(prompt)


def suggest_improvements(code: str, file_info: str = "") -> str:
    """Suggest code quality improvements."""
    prompt = _build_prompt(IMPROVE_SYSTEM, "Suggest improvements for this code.", code, file_info)
    return _generate(prompt)


def chat(query: str, context: str = "") -> str:
    """General chat response, optionally with code context."""
    prompt = _build_prompt(CHAT_SYSTEM, query, context)
    return _generate(prompt)
