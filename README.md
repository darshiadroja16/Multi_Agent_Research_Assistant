<div align="center">

# 🔬 Multi-Agent Research Assistant

### *Type a research query. Get a structured summary in 30 seconds.*

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph)
[![Google Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-E8A838?style=for-the-badge&logo=databricks&logoColor=white)](https://trychroma.com)
[![ArXiv](https://img.shields.io/badge/ArXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org)

<br/>

> A multi-agent AI system that autonomously searches ArXiv, retrieves relevant
> paper chunks using RAG, synthesizes structured research summaries using
> Llama 3.3 70B, and self-improves output through a Critic feedback loop.

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
║                      frontend/app.py                            ║
║                        port 8501                                ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║  HTTP POST /research
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║                      FASTAPI BACKEND                            ║
║                         main.py                                 ║
║                         port 8000                               ║
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

### Core AI Stack
| | | | | |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://skillicons.dev/icons?i=python" width="48"/><br/>Python | <img src="https://skillicons.dev/icons?i=fastapi" width="48"/><br/>FastAPI | <img src="https://skillicons.dev/icons?i=git" width="48"/><br/>Git | <img src="https://skillicons.dev/icons?i=github" width="48"/><br/>GitHub | <img src="https://skillicons.dev/icons?i=vscode" width="48"/><br/>VS Code |

### APIs & Services
| Tool | Role |
|:---:|:---|
| 🟣 **Groq** | LLM inference — Llama 3.3 70B (free, fastest) |
| 🔵 **Google Gemini** | Text embeddings — 768 dimensional vectors |
| 🟠 **ArXiv** | Research paper database — 2M+ papers, free |
| 🟡 **ChromaDB** | Vector database — stores and searches embeddings |
| 🔴 **LangGraph** | Multi-agent orchestration — graph based flow |
| 🟤 **LangSmith** | Agent monitoring — traces every step |
| 🩷 **Streamlit** | Python-only frontend UI |

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
├── main.py                    ← FastAPI server — /research /health
├── .env                       ← API keys (not committed — see .env.example)
├── .env.example               ← Template — copy this to create .env
└── pyproject.toml             ← Dependencies managed with uv
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
uv add google-generativeai chromadb groq langgraph langchain \
       fastapi uvicorn streamlit arxiv python-dotenv requests
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

> ⚠️ Never commit `.env` — it's already in `.gitignore`

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

## 🔭 Future Scope

- [ ] **PDF Upload** — summarize any paper directly without ArXiv search
- [ ] **Paper Comparison** — compare two specific papers side by side
- [ ] **Research History** — save and revisit past queries
- [ ] **Cloud Deployment** — Railway (backend) + Streamlit Cloud (frontend)
- [ ] **More Sources** — Semantic Scholar, PubMed, IEEE alongside ArXiv
- [ ] **Streaming Output** — results appear progressively instead of all at once
- [ ] **Citation Export** — download findings as PDF or BibTeX

---

<div align="center">

---

Made with ☕ by **Darshi Adroja**

*Final year B.Tech CSE · Parul University, Vadodara*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/darshi-adroja-1a3a9b33b/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/darshiadroja16)

⭐ **Star this repo if it helped you understand RAG or multi-agent systems**

</div>