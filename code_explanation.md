# Code Explanation — AI Research Agent

## File-by-File Walkthrough

---

### `src/config.py`

**Purpose:** Centralized configuration loaded from a `.env` file.

```python
from dotenv import load_dotenv
load_dotenv()                          # reads .env file into environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
```

All constants flow from environment variables with safe defaults. This pattern makes the project deployable to any environment (local, Docker, cloud) without code changes.

---

### `src/rag_pipeline.py`

**Purpose:** Index knowledge base documents and serve similarity search.

**`_chunk_text(text, source)`** — Splits a document into overlapping sentence-boundary chunks:
- Splits on sentence boundaries (`.`, `!`, `?`)
- Accumulates sentences until chunk exceeds `CHUNK_SIZE` characters
- Each chunk tagged with its source filename

```python
chunks.append((" ".join(current), source))   # (text, filename) tuples
```

**`_load_and_index()`** — On startup, reads all `.txt` files and builds a TF-IDF index:

```python
self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
```

**`search(query, top_k)`** — The retrieval function:

```python
query_vec = self.vectorizer.transform([query])          # query → TF-IDF vector
scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
top_indices = scores.argsort()[-top_k:][::-1]           # top-k by score
```

Returns `top_k` chunks with their source file and similarity score. Chunks with score < 0.01 are filtered out (no meaningful match).

---

### `src/tools.py`

**Purpose:** Implement and register callable tools for the agent.

**`search_knowledge_base(query)`** — Wraps `RAGPipeline.search()` and formats results:
```python
parts = [f"[Source: {r['source']}]\n{r['content']}" for r in results]
return "\n\n---\n\n".join(parts)
```

**`calculate(expression)`** — Uses Python's `eval()` safely by restricting available builtins:

```python
safe_globals = {
    "__builtins__": {},           # remove ALL Python builtins
    "sqrt": math.sqrt,            # only allow safe math functions
    "pi": math.pi,
    ...
}
result = eval(expression, safe_globals, {})
```

This prevents code injection while still supporting complex math expressions.

**`TOOLS` registry** — Dictionary mapping tool name → function + description:

```python
TOOLS = {
    "search_knowledge_base": {"fn": search_knowledge_base, "description": "..."},
    "calculate": {"fn": calculate, "description": "..."},
    "get_date": {"fn": get_date, "description": "..."},
}
```

Adding a new tool is a single dictionary entry — the agent automatically picks it up.

---

### `src/agent.py`

**Purpose:** The core ReAct agent loop.

**System prompt construction:**

```python
tool_desc = "\n".join(
    f"- {name}: {info['description']}" for name, info in TOOLS.items()
)
self.system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
```

The tool descriptions are dynamically injected from the `TOOLS` registry, so they stay in sync automatically.

**`_parse_response(text)`** — Parses LLM output using regex:

```python
final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
action_match = re.search(r"Action:\s*(\w+)", text)
input_match  = re.search(r"Action Input:\s*(.+?)(?:\nObservation|$)", text, re.DOTALL)
```

Checks for "Final Answer" first. If not found, extracts "Action" and "Action Input".

**`run(user_query)`** — The main agent loop:

```python
for iteration in range(MAX_AGENT_ITERATIONS):
    response = self.client.chat.completions.create(...)   # call LLM
    action_type, action_value = self._parse_response(...)  # parse output

    if action_type == "final":
        return action_value                                # done

    observation = self._run_tool(action_type, action_value)  # execute tool
    messages.append({"role": "user", "content": f"Observation: {observation}"})
    # loop continues with observation in context
```

**Key design:** Observations are appended as user messages, not system messages. This matches how leading agent frameworks (LangChain, LangGraph) handle observation injection.

---

### `main.py`

**Purpose:** Clean CLI chat loop.

```python
while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("exit", "quit", "q"):
        break
    answer = agent.run(user_input)
    print(f"\nAgent: {answer}\n")
```

All complexity is inside `agent.run()`. The main loop stays simple and user-friendly.

---

## Message Flow Diagram

```
messages = [system_prompt, user_query]
                │
         ┌──────▼──────┐
         │  LLM Call   │  → "Thought: ...\nAction: search_knowledge_base\nAction Input: RAG"
         └──────┬──────┘
                │ append assistant message
         ┌──────▼──────────────────────────────┐
         │  _parse_response()                  │
         │  → action_type = "search_kb"        │
         │  → action_value = "RAG"             │
         └──────┬──────────────────────────────┘
                │
         ┌──────▼──────────────────────────────┐
         │  _run_tool("search_kb", "RAG")       │
         │  → RAGPipeline.search("RAG")         │
         │  → returns top-3 document chunks     │
         └──────┬──────────────────────────────┘
                │ append "Observation: [chunks]" as user message
         ┌──────▼──────┐
         │  LLM Call   │  → "Thought: ...\nFinal Answer: RAG is..."
         └──────┬──────┘
                │
           return final answer to user
```
