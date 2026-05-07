import math
from datetime import datetime
from src.rag_pipeline import RAGPipeline

_rag_instance: RAGPipeline | None = None


def _rag() -> RAGPipeline:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
    return _rag_instance


def search_knowledge_base(query: str) -> str:
    """Search indexed documents for information relevant to the query."""
    results = _rag().search(query)
    if not results:
        return "No relevant information found in the knowledge base for that query."
    parts = [f"[Source: {r['source']}]\n{r['content']}" for r in results]
    return "\n\n---\n\n".join(parts)


def calculate(expression: str) -> str:
    """Evaluate a safe mathematical expression."""
    try:
        safe_globals = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "log": math.log,
            "log2": math.log2,
            "log10": math.log10,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "pow": pow,
        }
        result = eval(expression.replace("^", "**"), safe_globals, {})
        return f"{result}"
    except Exception as exc:
        return f"Calculation error: {exc}"


def get_date(_: str = "") -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%A, %B %d, %Y — %H:%M")


TOOLS: dict[str, dict] = {
    "search_knowledge_base": {
        "fn": search_knowledge_base,
        "description": (
            "Search the knowledge base for information on AI, ML, and related topics. "
            "Input: a descriptive search query string."
        ),
    },
    "calculate": {
        "fn": calculate,
        "description": (
            "Evaluate a mathematical expression. Supports: +, -, *, /, **, sqrt(), "
            "log(), sin(), cos(), pi, e. Input: the expression as a string."
        ),
    },
    "get_date": {
        "fn": get_date,
        "description": "Get the current date and time. Input: empty string.",
    },
}
