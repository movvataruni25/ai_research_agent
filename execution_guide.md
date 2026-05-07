# Execution Guide — AI Research Agent

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key (from https://platform.openai.com)
- pip (Python package manager)

## Step 1 — Navigate to Project Folder

```bash
cd project2_ai_research_agent
```

## Step 2 — Create a Virtual Environment

```bash
# Create
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Installs: `openai`, `scikit-learn`, `numpy`, `python-dotenv`

## Step 4 — Configure API Key

```bash
# Copy the example env file
cp .env.example .env
```

Open `.env` in any text editor and set your API key:

```
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini
```

`gpt-4o-mini` is recommended — it's fast, cheap, and works well for agent tasks.

## Step 5 — Run the Agent

```bash
python main.py
```

**Startup output:**
```
╔══════════════════════════════════════════════════╗
║          AI Research Agent  (ReAct + RAG)        ║
╚══════════════════════════════════════════════════╝
 Commands: type your question, or 'exit' to quit.

[RAG] Indexed 24 chunks from 2 file(s).
```

## Example Queries to Try

### Knowledge Base Questions
```
You: What is Retrieval-Augmented Generation?
You: Explain the ReAct agent pattern
You: What is TF-IDF and how does it work?
You: What is prompt chaining?
You: What are the differences between supervised and unsupervised learning?
You: How does hallucination happen in LLMs?
You: What is MLOps?
```

### Multi-Step Reasoning
```
You: What is RAG and what problems does it solve for LLMs?
You: Compare TF-IDF with embedding-based retrieval approaches
You: Explain overfitting and how regularization addresses it
```

### Calculator Tool
```
You: What is 15% of 4250?
You: Calculate the square root of 144
You: What is 2 to the power of 10?
You: Calculate log(1000)
```

### Date Tool
```
You: What is today's date?
You: What day of the week is it?
```

### Combined Reasoning
```
You: What is F1 score and if my model has precision 0.8 and recall 0.9, what is my F1?
```

## Sample Agent Output

```
You: What is RAG and why is it useful?

Agent: Thinking...

  [Agent] Tool: search_knowledge_base('RAG retrieval augmented generation')
  [Agent] Observation received (523 chars)

Agent: Retrieval-Augmented Generation (RAG) is a technique that combines the
generative capabilities of LLMs with a retrieval system to ground responses
in factual, up-to-date information.

A RAG pipeline has two phases:
1. Retrieval — the query is used to search a vector database for relevant chunks
2. Generation — those chunks are passed as context to the LLM

RAG directly addresses the hallucination problem by providing the model with
actual source material, making responses more accurate and trustworthy. It is
widely used in enterprise document Q&A, customer support bots, and knowledge
management systems.
```

## Troubleshooting

| Error | Solution |
|---|---|
| `OPENAI_API_KEY is not set` | Create `.env` file with your key |
| `AuthenticationError` | Check your API key is correct and active |
| `RateLimitError` | Wait a few seconds, or upgrade your OpenAI plan |
| `No relevant information found` | Rephrase query or add more docs to `knowledge_base/` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |

## Adding Custom Knowledge

1. Create any `.txt` file in the `knowledge_base/` folder
2. Restart the agent — it auto-indexes on startup
3. Ask questions about the new content

## Switching to GPT-4o (Better Reasoning)

In your `.env` file:
```
OPENAI_MODEL=gpt-4o
```

Note: GPT-4o costs more per token but handles complex multi-step reasoning better.

## Switching to Claude (Anthropic)

Replace `src/agent.py` client with:

```python
import anthropic
self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

And update the completion call format accordingly.
