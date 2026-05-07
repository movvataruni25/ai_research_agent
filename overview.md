# Project Overview — AI Research Agent

## What This Project Does

This project implements an **AI Research Agent** that can autonomously reason through multi-step questions using a combination of:
1. A **RAG (Retrieval-Augmented Generation)** pipeline to search a local knowledge base
2. A **calculator tool** for mathematical computations
3. A **date/time tool** for temporal queries
4. An **LLM** (GPT-4o-mini) as the reasoning backbone

The agent uses the **ReAct** pattern — a state-of-the-art technique where the model alternates between reasoning steps (Thought) and actions (Action → Observation) until it reaches a confident final answer.

## Why This Project Matters for Jobs

AI agents are the fastest-growing area in AI engineering in 2025–2026. Companies building LLM applications are moving from simple chatbots to **autonomous agents** that can use tools, plan multi-step tasks, and operate with minimal human supervision. This project directly demonstrates:

- Understanding of the ReAct agent architecture (LangChain, LangGraph, AutoGen all use this)
- Ability to build RAG systems from scratch
- LLM API integration experience
- Tool routing and safe execution patterns
- Prompt engineering for agent behavior control

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    ResearchAgent                        │
│                                                         │
│  User Query                                             │
│      │                                                  │
│      ▼                                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │              ReAct Agent Loop                    │   │
│  │                                                  │   │
│  │  [LLM] Reason → generate Thought + Action        │   │
│  │      │                                           │   │
│  │      ▼                                           │   │
│  │  Parse Action & Action Input from LLM response   │   │
│  │      │                                           │   │
│  │      ▼                                           │   │
│  │  ┌──────────────────────────────────────────┐   │   │
│  │  │              Tool Router                 │   │   │
│  │  │  search_knowledge_base │ calculate │ date│   │   │
│  │  └──────────────────────────────────────────┘   │   │
│  │      │                                           │   │
│  │      ▼                                           │   │
│  │  Observation fed back to LLM context             │   │
│  │      │                                           │   │
│  │      └──── repeat until "Final Answer" ─────────┘   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  RAGPipeline (rag_pipeline.py)                          │
│  ├── Load .txt files from knowledge_base/               │
│  ├── Chunk text into ~500-char segments                 │
│  ├── TF-IDF vectorize all chunks                        │
│  └── search() → cosine similarity → top-k results      │
└────────────────────────────────────────────────────────┘
```

## ReAct Pattern Explained

The ReAct pattern (from the 2022 paper "ReAct: Synergizing Reasoning and Acting in Language Models") teaches agents to interleave reasoning with action-taking:

```
Thought: I need to find information about RAG.
Action: search_knowledge_base
Action Input: RAG retrieval augmented generation
Observation: [retrieved document chunks]

Thought: I have enough context. Now I can explain it.
Final Answer: RAG is a technique that...
```

This mirrors how a human researcher works: form a hypothesis → look it up → synthesize → answer.

## Key Design Principles

| Principle | Implementation |
|---|---|
| Separation of concerns | config, rag, tools, agent are independent modules |
| No magic dependencies | TF-IDF retrieval works without any embedding API |
| Safe tool execution | Calculator uses restricted `eval()` with only safe builtins |
| Extensible tools | New tools added as dict entries in `TOOLS` registry |
| Configurable | All settings in `.env`, read through `config.py` |

## Real-World Counterparts

| This Project | Production Equivalent |
|---|---|
| ReAct loop | LangChain AgentExecutor, LangGraph |
| TF-IDF RAG | Full vector RAG with embeddings + ChromaDB/Pinecone |
| Tool registry | LangChain Tool class, OpenAI function calling |
| Knowledge base .txt files | Enterprise document stores, SharePoint, Confluence |
| CLI chat | Slack bot, web chat UI, API endpoint |
