import base64
import logging
import os
import sqlite3
import time
from pathlib import Path
from dotenv import load_dotenv
import certifi
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, message_chunk_to_message
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from tools import tools

logger = logging.getLogger("uvicorn.error")

Path("data").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Free model registry
#
#   groq   -> free tier, needs GROQ_API_KEY   (https://console.groq.com/keys)
#   gemini -> free tier, needs GOOGLE_API_KEY (https://aistudio.google.com/apikey)
#   ollama -> 100% free & local, needs Ollama running (https://ollama.com)
#
# Models must support tool calling, which the agent needs.
# ---------------------------------------------------------------------------
def _env_list(name: str, default: str) -> list[str]:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


# Provider catalogs change often, so each list can be overridden in .env
# (comma separated model ids). Defaults are free-tier models with tool calling.
GROQ_MODELS = _env_list("GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b")
GEMINI_MODELS = _env_list("GEMINI_MODELS", "gemini-flash-latest,gemini-flash-lite-latest")
OLLAMA_MODELS = _env_list("OLLAMA_MODELS", "")  # e.g. llama3.1,qwen2.5 (needs `ollama pull <model>`)

FREE_MODELS = {}
for _m in GROQ_MODELS:
    FREE_MODELS[_m] = {"provider": "groq", "label": f"{_m} (Groq)"}
for _m in GEMINI_MODELS:
    FREE_MODELS[_m] = {"provider": "gemini", "label": f"{_m} (Gemini)"}
for _m in OLLAMA_MODELS:
    FREE_MODELS[f"ollama:{_m}"] = {"provider": "ollama", "label": f"{_m} (Ollama, local)"}

PROVIDER_KEYS = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY", "ollama": None}


def provider_available(provider: str) -> bool:
    key = PROVIDER_KEYS.get(provider)
    return key is None or bool(os.getenv(key))


def available_models() -> list[dict]:
    """Models whose provider is configured (API key present / local)."""
    return [
        {"id": model_id, "label": info["label"], "provider": info["provider"]}
        for model_id, info in FREE_MODELS.items()
        if provider_available(info["provider"])
    ]


def _default_model() -> str:
    env_default = os.getenv("DEFAULT_MODEL", "").strip()
    if env_default in FREE_MODELS and provider_available(FREE_MODELS[env_default]["provider"]):
        return env_default
    models = available_models()
    if not models:
        raise RuntimeError(
            "No free AI provider configured. Set GROQ_API_KEY or GOOGLE_API_KEY "
            "in .env, or set OLLAMA_MODELS and run Ollama locally."
        )
    return models[0]["id"]


SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named Thread Pilot, similar to ChatGPT.

You can:
1. Answer general questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for the latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memories when useful.
7. Use a calculator for mathematical calculations.

Rules:
- If the user asks about the latest news, current events, recent updates, today's information, or current prices, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use the calculator for mathematical questions.
- When using web search, summarize the results clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""


def normalize_model_name(model_name: str | None) -> str:
    """
    Validate selected model from frontend.
    If model is missing, unknown, or its provider isn't configured, fall back to the default.
    """
    if model_name:
        model_name = model_name.strip()
        info = FREE_MODELS.get(model_name)
        if info and provider_available(info["provider"]):
            return model_name
    return _default_model()


# Give up on a model that sends nothing for this many seconds. This limits the
# wait between chunks, so long answers that keep streaming are not cut off.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "12"))


def create_llm(model_name: str):
    """
    Create the chat model for the given (already normalized) model id.

    Provider-level retries are off: retrying an overloaded model is slow, so
    call_with_fallback() moves on to the next model instead.
    """
    provider = FREE_MODELS[model_name]["provider"]

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, temperature=0, max_retries=0, timeout=LLM_TIMEOUT)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # max_retries=1 means a single attempt (0 would use the SDK's own default).
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_retries=1,
            timeout=LLM_TIMEOUT,
            # Otherwise the SDK sends LLM_TIMEOUT as a limit on the whole answer.
            additional_headers={"X-Server-Timeout": "120"},
            # Gemini 3+ thinks at "high" by default, which is slow. Gemini 2.x lacks this option.
            thinking_level=None if "gemini-2" in model_name else "low",
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name.split(":", 1)[1],
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )

    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Failover: when a model is overloaded, rate limited or down, try the next one.
# ---------------------------------------------------------------------------
COOLDOWN_SECONDS = 120      # skip a failing model for this long
MAX_HISTORY_CHARS = 16000   # ~4k tokens; Groq's free tier has low per-minute token limits

# Placeholder Gemini accepts for tool calls that were made by another provider.
_SKIP_SIGNATURE = base64.b64encode(b"skip_thought_signature_validator").decode()
_SIGNATURES_KEY = "__gemini_function_call_thought_signatures__"

_LLM_CACHE = {}
_cooldown_until = {}


class AllModelsBusyError(RuntimeError):
    """Every configured model failed to answer."""


def _get_llm(model_name: str):
    if model_name not in _LLM_CACHE:
        _LLM_CACHE[model_name] = create_llm(model_name).bind_tools(tools)
    return _LLM_CACHE[model_name]


def fallback_chain(selected: str) -> list[str]:
    """Selected model first, then the others; recently failing models go last."""
    others = [
        m["id"] for m in available_models()
        if m["id"] != selected and m["provider"] != "ollama"
    ]
    now = time.monotonic()
    return sorted([selected] + others, key=lambda m: _cooldown_until.get(m, 0) > now)


def trim_history(messages: list) -> list:
    """
    Keep only the most recent turns. A turn starts at a HumanMessage, so tool
    calls are never separated from their results. The latest turn is always kept.
    """
    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not starts:
        return messages

    sizes = [len(m.text) for m in messages]
    keep_from = starts[-1]
    for start in reversed(starts[:-1]):
        if sum(sizes[start:]) > MAX_HISTORY_CHARS:
            break
        keep_from = start
    return messages[keep_from:]


def _for_provider(messages: list, provider: str) -> list:
    """Gemini rejects tool calls that carry no thought signature, so add one if missing."""
    if provider != "gemini":
        return messages

    prepared = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls and not m.additional_kwargs.get(_SIGNATURES_KEY):
            # Only the first tool call of a step needs a signature.
            signatures = {m.tool_calls[0]["id"]: _SKIP_SIGNATURE}
            m = m.model_copy(update={
                "additional_kwargs": {**m.additional_kwargs, _SIGNATURES_KEY: signatures}
            })
        prepared.append(m)
    return prepared


def _status_code(error: Exception):
    return getattr(error, "status_code", None) or getattr(error, "code", None)


def call_with_fallback(selected: str, messages: list, config: RunnableConfig):
    """
    Stream a reply from the selected model, switching models if it fails.
    Once text has reached the user we no longer switch, to avoid a duplicated answer.
    """
    last_error = None
    started = time.monotonic()

    for attempt in range(2):
        if attempt:
            # Everything failed. Try once more, unless the user has waited long already.
            if time.monotonic() - started > 10:
                break
            time.sleep(1)

        for name in fallback_chain(selected):
            reply = None
            text_sent = False
            try:
                prepared = _for_provider(messages, FREE_MODELS[name]["provider"])
                for chunk in _get_llm(name).stream(prepared, config):
                    reply = chunk if reply is None else reply + chunk
                    text_sent = text_sent or bool(chunk.text)
                if reply is None:
                    raise RuntimeError("empty response")
                _cooldown_until.pop(name, None)
                return message_chunk_to_message(reply)
            except Exception as error:
                last_error = error
                logger.warning("Model %s failed: %s", name, str(error)[:200])
                if text_sent:
                    raise
                # 400/413 are about this request, not about the model's health.
                if _status_code(error) not in (400, 413):
                    _cooldown_until[name] = time.monotonic() + COOLDOWN_SECONDS

    raise AllModelsBusyError(
        "All AI models are busy right now. Please try again in a minute."
    ) from last_error


_checkpoint_conn = sqlite3.connect("data/langgraph_checkpoints.sqlite", check_same_thread=False)
_checkpointer = SqliteSaver(_checkpoint_conn)


def build_agent(model_name: str):
    """
    Build one LangGraph agent for the selected free model.
    """
    def chatbot_node(state: MessagesState, config: RunnableConfig):
        message = [SystemMessage(content=SYSTEM_PROMPT)] + trim_history(state["messages"])
        response = call_with_fallback(model_name, message, config)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")
    return workflow.compile(checkpointer=_checkpointer)


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """
    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]
