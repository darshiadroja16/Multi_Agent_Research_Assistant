<div align="center">

# 🔬 Multi-Agent Research Assistant

### *Type a research query. Get a structured summary in 30 seconds.*

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph)
[![Google Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-E8A838?style=for-the-badge&logo=databricks&logoColor=white)](https://trychroma.com)
[![ArXiv](https://img.shields.io/badge/ArXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

<br/>

> A multi-agent AI system that autonomously searches ArXiv, retrieves relevant
> paper chunks using RAG, synthesizes structured research summaries using
> Llama 3.3 70B, and self-improves output through a Critic feedback loop.

<br/>

### 🔗 Live Demo
**[→ Try it here](https://multiagentresearchassistant-jyfbxsqksjvcyhgvar235w.streamlit.app/)**

</div>

---

## ✨ Features

| | Feature | What It Does |
|---|---|---|
| 🔍 | **Auto Paper Discovery** | Searches ArXiv and fetches top 6 relevant papers |
| 🧩 | **Smart Chunking** | Splits papers into overlapping chunks to preserve context |
| 🎯 | **Semantic Search** | Finds most relevant chunks using cosine similarity |
| 🤖 | **LLM Summarization** | Llama 3.3 70B generates structured JSON output |
| 🔁 | **Self-Critique Loop** | Critic Agent re-generates summary if quality score < 7 |
| 📊 | **Structured Output** | Key findings, methodology, research gaps, recommendations |
| 🌐 | **Clean UI** | Streamlit frontend with metrics and source paper links |
| 🔎 | **Agent Monitoring** | LangSmith traces every agent step |

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                        STREAMLIT UI                             ║
║              multiagentresearchassistant.streamlit.app          ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║  HTTP POST /research
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║                      FASTAPI BACKEND                            ║
║           multi-agent-research-assistant.onrender.com           ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║                   LANGGRAPH ORCHESTRATOR                        ║
║                    agents/orchestrator.py                       ║
║                                                                  ║
║  ┌──────────────┐   ┌─────────────┐   ┌───────────────────┐   ║
║  │ SEARCH NODE  │──▶│  RAG NODE   │──▶│  SUMMARIZER NODE  │   ║
║  │              │   │             │   │                   │   ║
║  │  ArXiv API   │   │ Gemini API  │   │  Groq API         │   ║
║  │  6 papers    │   │ embed+store │   │  Llama 3.3 70B    │   ║
║  │  fetched     │   │ ChromaDB    │   │  JSON summary     │   ║
║  └──────────────┘   │ top 5 chunk│   └─────────┬─────────┘   ║
║                      └─────────────┘             │              ║
║                                                  ▼              ║
║                                       ┌───────────────────┐    ║
║                                       │   CRITIC NODE     │    ║
║                                       │                   │    ║
║                                       │  Groq API         │    ║
║                                       │  score 1-10       │    ║
║                                       └─────────┬─────────┘    ║
║                                                 │               ║
║                               ┌─────────────────┴──────────┐   ║
║                               │                            │   ║
║                            score < 7                   score ≥ 7║
║                               │                            │   ║
║                               ▼                            ▼   ║
║                       ┌─────────────┐          ┌──────────────┐║
║                       │  loop back  │          │ FORMAT NODE  │║
║                       │  summarizer │          │ final answer │║
║                       │  (max 2x)   │          │  → response  │║
║                       └─────────────┘          └──────────────┘║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 💡 How It Works

**1. Search Agent** — ArXiv se papers fetch karo
```
query → ArXiv API → top 6 papers (title + abstract + url)
```

**2. RAG Agent** — Chunk, embed, store, retrieve
```
papers → chunk (400 chars, 50 overlap)
       → Gemini text-embedding-004 → 768-dim vectors
       → ChromaDB cosine similarity search
       → top 5 relevant chunks
```

**3. Summarizer Agent** — LLM generates structured output
```
chunks + query → Groq Llama 3.3 70B → JSON
{
  key_findings, methodology_comparison,
  research_gaps, recommendations, confidence_score
}
```

**4. Critic Agent** — Quality gate with self-improvement
```
summary → Groq evaluates → score 1-10
score ≥ 7  →  accepted → format → response
score < 7  →  rejected → back to summarizer (max 2 loops)
```

---

## 🛠️ Built With

<div align="center">

| | | | | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="https://skillicons.dev/icons?i=python" width="48"/><br/>Python | <img src="https://skillicons.dev/icons?i=fastapi" width="48"/><br/>FastAPI | <img src="https://skillicons.dev/icons?i=github" width="48"/><br/>GitHub | <img src="https://skillicons.dev/icons?i=git" width="48"/><br/>Git | <img src="https://skillicons.dev/icons?i=vscode" width="48"/><br/>VS Code | <img src="https://skillicons.dev/icons?i=render" width="48"/><br/>Render | <img src="https://skillicons.dev/icons?i=cloudflare" width="48"/><br/>Streamlit |

### APIs & Services

| Tool | Role |
|:---:|:---|
| 🟣 **Groq** | LLM inference — Llama 3.3 70B (free, fastest) |
| 🔵 **Google Gemini** | Text embeddings — 768 dimensional vectors |
| 🟠 **ArXiv** | Research paper database — 2M+ papers, completely free |
| 🟡 **ChromaDB** | Vector database — stores and searches embeddings |
| 🔴 **LangGraph** | Multi-agent orchestration — graph based agent flow |
| 🟤 **LangSmith** | Agent monitoring — traces every step |
| 🩷 **Streamlit** | Python-only frontend — no JS needed |

</div>

---

## 📁 Project Structure

```
Multi_Agent_Research_Assistant/
│
├── 📂 agents/
│   └── orchestrator.py        ← LangGraph graph — all 5 nodes + conditional edge
│
├── 📂 tools/
│   ├── embedder.py            ← Gemini API — text to 768-dim vector
│   ├── vector_store.py        ← ChromaDB — chunk, store, semantic search
│   └── llm_tool.py            ← Groq LLM — summarizer + critic prompts
│
├── 📂 models/
│   └── schemas.py             ← ResearchState TypedDict + Pydantic API models
│
├── 📂 frontend/
│   └── app.py                 ← Streamlit UI — search bar + results display
│
├── 📂 learning/               ← practice files used while building this
│   ├── rag_pipeline.py        ← RAG from scratch experiment
│   ├── prompt_test.py         ← prompt engineering tests
│   └── integration_test.py    ← all tools integration test
│
├── main.py                    ← FastAPI server — /research /health
├── requirements.txt           ← dependencies for deployment
├── .env.example               ← copy this to create your .env
└── pyproject.toml             ← dependencies managed with uv
```

---

## ⚙️ Setup & Installation

### Requirements
- Python 3.11+
- Git

### 1 — Clone

```bash
git clone https://github.com/darshiadroja16/Multi_Agent_Research_Assistant.git
cd Multi_Agent_Research_Assistant
```

### 2 — Virtual Environment

```bash
pip install uv
uv venv

# activate
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate          # Windows
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 4 — Set Up API Keys

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
GEMINI_API_KEY=           # aistudio.google.com → free 1500 req/day
GROQ_API_KEY=             # console.groq.com → free 14400 req/day
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=        # smith.langchain.com → free unlimited
LANGCHAIN_PROJECT=research-assistant
BACKEND_URL=http://localhost:8000
```

> ⚠️ Never commit `.env` — it is already in `.gitignore`

### 5 — Run Backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6 — Run Frontend

Open a new terminal:

```bash
source .venv/bin/activate
streamlit run frontend/app.py --server.port 8501
```

### 7 — Open Browser

```
http://localhost:8501
```

---

## 🚀 Deployment

| Service | Platform | URL |
|---|---|---|
| Backend (FastAPI) | Render | [multi-agent-research-assistant-52n9.onrender.com](https://multi-agent-research-assistant-52n9.onrender.com) |
| Frontend (Streamlit) | Streamlit Cloud | [multiagentresearchassistant.streamlit.app](https://multiagentresearchassistant-jyfbxsqksjvcyhgvar235w.streamlit.app/) |

> ℹ️ Free tier on Render sleeps after 15 min of inactivity. First request may take 30-50 seconds to wake up.

---

## 🔭 Future Scope

- [ ] PDF upload — summarize any paper directly
- [ ] Side-by-side paper comparison
- [ ] Save and export research history
- [ ] Add Semantic Scholar as second paper source
- [ ] Streaming responses so results appear progressively
- [ ] Citation export as BibTeX or PDF

---

<div align="center">

---

got tired of reading papers manually so i built something to do it for me 🤷‍♀️

⭐ drop a star if this was useful

</div>