import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import certifi
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from tools import tools

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


def create_llm(model_name: str):
    """Create the chat model for the given (already normalized) model id."""
    provider = FREE_MODELS[model_name]["provider"]

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, temperature=0, max_retries=3)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=3)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name.split(":", 1)[1],
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )

    raise ValueError(f"Unknown provider: {provider}")


_checkpoint_conn = sqlite3.connect("data/langgraph_checkpoints.sqlite", check_same_thread=False)
_checkpointer = SqliteSaver(_checkpoint_conn)


def build_agent(model_name: str):
    """
    Build one LangGraph agent for the selected free model.
    """
    llm_with_tools = create_llm(model_name).bind_tools(tools)

    def chatbot_node(state: MessagesState):
        message = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(message)
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
