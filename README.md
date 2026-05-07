# AI Research Agent — ReAct + RAG

An intelligent research assistant that uses the **ReAct** (Reason + Act) agent pattern combined with **Retrieval-Augmented Generation (RAG)** to answer questions using a local knowledge base, perform calculations, and retrieve the current date.

## How It Works

The agent follows a Thought → Action → Observation loop until it has enough information to give a Final Answer — the same pattern used in production AI agent systems.

```
User Query
    │
    ▼
[LLM] Thought: I need to look this up.
      Action: search_knowledge_base
      Action Input: "what is RAG?"
    │
    ▼
[Tool] Executes search → returns relevant document chunks
    │
    ▼
[LLM] Observation received → continues reasoning
      Thought: I now have enough information.
      Final Answer: RAG is...
    │
    ▼
User sees the answer
```

## Features

- ReAct agent loop with configurable max iterations
- TF-IDF RAG pipeline (no external embedding API required)
- Three built-in tools: document search, calculator, date retrieval
- Extensible — add new tools in `src/tools.py` with one entry
- Configurable via `.env` file
- Clean CLI chat interface

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI API (gpt-4o-mini by default) |
| Agent Pattern | ReAct (Reason + Act) |
| Retrieval | TF-IDF + Cosine Similarity (scikit-learn) |
| Config | python-dotenv |
| Language | Python 3.10+ |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your-key-here

# 3. Run the agent
python main.py
```

## Example Conversation

```
You: What is RAG and why is it useful?

Agent: Thinking...
  [Agent] Tool: search_knowledge_base('RAG retrieval augmented generation')
  [Agent] Observation received (487 chars)

Agent: Retrieval-Augmented Generation (RAG) combines LLM generation with
a retrieval system to ground responses in factual documents. It solves the
hallucination problem by giving the model actual source material...

---

You: What is 15% of 4250?

Agent: Thinking...
  [Agent] Tool: calculate('4250 * 0.15')

Agent: 15% of 4250 is 637.5.
```

## Project Structure

```
project2_ai_research_agent/
├── src/
│   ├── config.py           # Environment variables & settings
│   ├── rag_pipeline.py     # Document loading, chunking, TF-IDF search
│   ├── tools.py            # Tool registry and implementations
│   └── agent.py            # ReAct agent loop
├── knowledge_base/
│   ├── ai_concepts.txt     # AI/GenAI topic knowledge
│   └── ml_topics.txt       # ML fundamentals knowledge
├── main.py                 # CLI chat interface
├── .env.example            # Environment variable template
├── requirements.txt
├── README.md
├── overview.md
├── code_explanation.md
├── execution_guide.md
└── project_documentation.md
```

## Adding Custom Knowledge

Drop any `.txt` files into the `knowledge_base/` folder. They are automatically indexed at startup. No code changes required.

## Skills Demonstrated

- ReAct agent architecture
- RAG pipeline implementation
- Tool-use and function routing
- LLM API integration (OpenAI)
- Modular Python design
- Configuration management
