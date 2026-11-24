import ast
import math
import operator
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from database import save_memory, search_memory
from rag import retrieve_from_rag
load_dotenv()

web_search = TavilySearch(max_results=5,topic="general",search_depth="basic")

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS = {"abs": abs, "min": min, "max": max, "sum": sum}
_FUNCTIONS.update({
    name: getattr(math, name)
    for name in (
        "sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan", "asin", "acos",
        "atan", "atan2", "sinh", "cosh", "tanh", "degrees", "radians", "hypot",
        "floor", "ceil", "trunc", "fabs", "gcd",
    )
})
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _power(base, exponent):
    # Keeps expressions such as 9**9**9 from using up the server's CPU and memory.
    if isinstance(base, int) and isinstance(exponent, int):
        if base.bit_length() * abs(exponent) > 10000:
            raise ValueError("number too large")
    return base ** exponent


def _round(value, digits=None):
    # A huge negative digit count makes Python build an equally huge number.
    if digits is not None and abs(digits) > 300:
        raise ValueError("too many digits")
    return round(value, digits)


def _factorial(n):
    if n > 500:
        raise ValueError("number too large")
    return math.factorial(n)


_OPERATORS[ast.Pow] = _power
_FUNCTIONS["pow"] = _power
_FUNCTIONS["factorial"] = _factorial
_FUNCTIONS["round"] = _round


def _evaluate(node):
    """Evaluate arithmetic only. Anything that is not listed here is rejected."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_number(node.left), _number(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_number(node.operand))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate(item) for item in node.elts]
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _CONSTANTS[_name(node)]
    if isinstance(node, ast.Call) and not node.keywords:
        return _FUNCTIONS[_name(node.func)](*[_evaluate(arg) for arg in node.args])
    raise ValueError("unsupported expression")


def _number(node):
    value = _evaluate(node)
    if isinstance(value, list):
        # [1] * 10**9 would fill the server's memory.
        raise ValueError("unsupported expression")
    return value


def _name(node):
    """Accepts sqrt and math.sqrt, nothing else."""
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math":
        name = node.attr
    else:
        raise ValueError("unsupported expression")
    if name not in _FUNCTIONS and name not in _CONSTANTS:
        raise ValueError(f"unknown name: {name}")
    return name


@tool
def calculator(expression:str) ->str:
    """
    Useful for simple math calculations. Input should be a valid math expression.
    Example: 2+2, math.sqrt(16), 10*5
    """
    try:
        if len(expression) > 300:
            raise ValueError("expression too long")
        # A list is only useful as a function argument, e.g. sum([1, 2, 3]).
        result = _evaluate(ast.parse(expression.strip(), mode="eval"))
        if isinstance(result, list):
            raise ValueError("unsupported expression")
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


# The thread id comes from the request's own config, which LangChain injects
# and hides from the model. A global would be shared between users.
def _thread_id(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


@tool
def search_uploaded_documents(query:str, config:RunnableConfig) -> str:
    """
    Search uploaded documents for relevent information.
    Use this when the user asks about uploaded PDFs, DOCX, TXT, notes, files, or documents."""
    return retrieve_from_rag(query=query,thread_id=_thread_id(config))


@tool
def remember_this(memory:str, config:RunnableConfig) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remembber something.
    """
    return save_memory(thread_id=_thread_id(config),memory=memory)

@tool
def recall_memory(query:str, config:RunnableConfig) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """
    return search_memory(thread_id=_thread_id(config),query=query)


tools =[calculator,search_uploaded_documents,remember_this,recall_memory,web_search]