# Project Documentation — AI Research Agent

## Project Summary

| Attribute | Value |
|---|---|
| Project Type | Agentic AI / LLM Application |
| Pattern | ReAct (Reason + Act) |
| Retrieval | RAG with TF-IDF |
| LLM Backend | OpenAI API (gpt-4o-mini) |
| Complexity | Intermediate–Advanced |
| Status | Complete |
| Python Version | 3.10+ |

## Business Context

Agentic AI systems are the most in-demand AI engineering skill in 2025–2026. Every major AI product company — Salesforce, ServiceNow, Microsoft, Google, Anthropic — is building agent-based workflows. This project demonstrates:

1. Understanding of how production AI agents work at the architecture level
2. Ability to build RAG pipelines for knowledge grounding
3. LLM API integration with structured output parsing
4. Tool-use patterns that match LangChain, LangGraph, and AutoGen abstractions

## Technical Design

### ReAct Agent Pattern

The ReAct pattern structures agent reasoning as an alternating sequence:

```
User Query
    │
    ▼
Thought: (internal reasoning — what do I need to do?)
Action: (which tool to use)
Action Input: (what to pass to the tool)
    │
    ▼
Observation: (tool result — fed back to LLM)
    │
    ▼
Thought: (continue reasoning...)
    ...
Final Answer: (return to user when complete)
```

This structure forces the LLM to reason explicitly before acting, which significantly reduces errors compared to direct question-answering.

### RAG Pipeline Design

```
Knowledge Base Files (.txt)
         │
         ▼
Sentence Boundary Chunking        (~500 chars per chunk)
         │
         ▼
TF-IDF Vectorization              (fit at startup)
         │
         ▼
Index Stored in Memory            (numpy matrix)
         │
   [At Query Time]
         │
         ▼
Query → TF-IDF Transform          (using fitted vectorizer)
         │
         ▼
Cosine Similarity vs. All Chunks
         │
         ▼
Top-K Results Returned            (threshold filtered)
```

**Why TF-IDF instead of embeddings?**
- No external API calls required for retrieval
- No GPU or additional model downloads
- Fully deterministic and interpretable
- Sufficient for structured, domain-specific knowledge bases
- In production, replace with sentence-transformers + ChromaDB for better semantic matching

### Tool Safety Design

The calculator uses restricted `eval()`:

```python
safe_globals = {"__builtins__": {}, "sqrt": math.sqrt, ...}
eval(expression, safe_globals, {})
```

By setting `__builtins__` to an empty dict, Python's standard library and all filesystem/network access is blocked. Only the explicitly whitelisted math functions are available. This is a standard pattern for safe expression evaluation.

## Comparison to Production Systems

| This Project | Production Equivalent | Upgrade Path |
|---|---|---|
| TF-IDF retrieval | Embedding-based vector search | sentence-transformers + ChromaDB |
| Regex output parsing | OpenAI function calling / structured outputs | Add `response_format={"type": "json_object"}` |
| In-memory index | Persistent vector database | ChromaDB, Pinecone, Weaviate |
| txt knowledge base | Enterprise document stores | PDF parsing, web scraping, database connectors |
| CLI interface | Web API | Wrap `agent.run()` in FastAPI endpoint |
| Single agent | Multi-agent orchestration | LangGraph, AutoGen |

## Agent Limitations and Mitigations

| Limitation | Mitigation Applied |
|---|---|
| Hallucination | RAG grounds answers in retrieved documents |
| Infinite loops | `MAX_AGENT_ITERATIONS` cap (default: 10) |
| Tool errors crash agent | Try/except in `_run_tool()` returns error string |
| Unknown tools requested | Tool registry lookup with clear error message |
| Context overflow | Max tokens set on each LLM call |

## Key Files Reference

| File | Purpose | Key Class/Function |
|---|---|---|
| `src/config.py` | Settings from .env | Module-level constants |
| `src/rag_pipeline.py` | Document indexing & search | `RAGPipeline.search()` |
| `src/tools.py` | Tool implementations & registry | `TOOLS` dict, `calculate()`, `search_knowledge_base()` |
| `src/agent.py` | ReAct loop | `ResearchAgent.run()` |
| `main.py` | CLI entry point | `main()` |

## Extending the Project

### Add a New Tool

```python
# In src/tools.py

def web_search(query: str) -> str:
    """Search the web for current information."""
    # ... implementation
    return results_string

TOOLS["web_search"] = {
    "fn": web_search,
    "description": "Search the web for recent news and information. Input: search query.",
}
```

No other code changes needed — the agent picks it up automatically.

### Add Persistent Memory

```python
# In src/agent.py — store conversation history between sessions
import json

def save_history(messages, path="chat_history.json"):
    json.dump(messages, open(path, "w"))

def load_history(path="chat_history.json"):
    if Path(path).exists():
        return json.load(open(path))
    return []
```

### Convert to REST API (FastAPI)

```python
from fastapi import FastAPI
from src.agent import ResearchAgent

app = FastAPI()
agent = ResearchAgent()

@app.post("/chat")
def chat(query: str):
    return {"answer": agent.run(query)}
```

## Skills Demonstrated

This project directly demonstrates skills required in AI Engineering roles:

- **Agentic AI Architecture** — ReAct pattern, tool use, multi-step reasoning
- **RAG Systems** — Document loading, chunking, vectorization, similarity search
- **LLM API Integration** — OpenAI Python SDK, prompt design, response parsing
- **Prompt Engineering** — Structured prompts for controlled agent behavior
- **Python Software Design** — Modular architecture, configuration management
- **Safe Code Execution** — Restricted eval() for calculator tool
- **Error Handling** — Graceful degradation on tool failures
- **Extensible Design** — Registry pattern for tools and documents
