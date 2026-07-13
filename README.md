Now let's update the README to reflect the GraphRAG feature. Open `README.md` and replace the entire content with this:

```markdown
# 🎫 RAG Support Ticket Search & Assistant

A production-grade **GraphRAG system** that helps support engineers quickly find relevant tickets and get AI-generated answers. Combines **vector similarity search** (ChromaDB) with **knowledge graph traversal** (Neo4j) for richer, more accurate results than basic RAG.

Built as a technical demonstration directly matching real-world AI engineering requirements.

---

## 🎯 What It Does

Instead of keyword search, this system:
1. Converts support tickets into vector embeddings (ChromaDB)
2. Builds a knowledge graph connecting tickets by tags and relationships (Neo4j)
3. When a user asks a question:
   - Finds semantically similar tickets via vector search
   - Expands results through graph traversal to find related tickets
   - Feeds combined context to GPT-4o
   - Returns accurate, grounded AI response

---

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────────────────────┐
│  Vector Search (ChromaDB)       │  ← semantic similarity
│  + Graph Traversal (Neo4j)      │  ← relationship expansion  
└─────────────────────────────────┘
    ↓
Combined Context (richer than basic RAG)
    ↓
GPT-4o generates answer
    ↓
Display answer + relevant tickets
```

## 🆚 GraphRAG vs Basic RAG

| | Basic RAG | GraphRAG (this project) |
|---|---|---|
| Search method | Vector similarity only | Vector + graph traversal |
| Finds related tickets | ❌ | ✅ through shared tags |
| Context richness | Limited | Rich — multiple hops |
| Accuracy | Good | Better |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Database | ChromaDB |
| Knowledge Graph | Neo4j |
| RAG Framework | LangChain |
| Web UI | Streamlit |
| Data Formats | JSON + XML |
| Containerization | Docker |

---

## 📁 Project Structure

```
rag-support-tickets/
├── src/
│   ├── document_loader.py   # Loads JSON and XML tickets
│   ├── vector_store.py      # ChromaDB vector store manager
│   ├── graph_store.py       # Neo4j knowledge graph manager
│   └── rag_chain.py         # GraphRAG pipeline with GPT-4o
├── data/
│   ├── technical/           # Technical support tickets
│   ├── product/             # Product support tickets
│   └── customer/            # Customer support tickets
├── app.py                   # Streamlit web interface
├── Dockerfile               # Container configuration
└── requirements.txt         # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Neo4j (local or Docker)
- OpenAI API key

### Local Setup

1. Clone the repo:
```bash
git clone https://github.com/hamedkarimi3/rag-support-tickets.git
cd rag-support-tickets
```

2. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Neo4j with Docker:
```bash
docker run -d \
  --name neo4j-rag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5.15
```

5. Add your API keys:
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
echo "NEO4J_URI=bolt://localhost:7687" >> .env
echo "NEO4J_USERNAME=neo4j" >> .env
echo "NEO4J_PASSWORD=password123" >> .env
```

6. Run the app:
```bash
streamlit run app.py
```

---

## 💡 Key Features

- **GraphRAG** — combines vector search + graph traversal for richer context
- **Knowledge Graph** — tickets connected by shared tags and support type
- **Semantic Search** — finds tickets by meaning, not keywords
- **Multi-format Support** — loads both JSON and XML ticket files
- **Persistent Storage** — vector store saved to disk, no re-embedding on restart
- **Filter by Type** — search within technical, product, or customer tickets
- **Graph Stats Dashboard** — shows tickets, tags, and relations in real time
- **Production Ready** — proper error handling, logging, and Docker support

---

## 🔧 How GraphRAG Works Here

**Step 1 — Vector Search:**
> "Chrome login error" → finds ticket T002 (Chrome 403 error)

**Step 2 — Graph Traversal:**
> T002 shares tag "login" with C001 (account login issue)
> C001 shares tag "browser" with T003 (Outlook sync)
> System retrieves all related tickets automatically

**Step 3 — Richer Context:**
> GPT-4o receives T002 + C001 + related tickets → better, more complete answer

---

## 👤 Author

Hami Karimi — Senior Data & AI Engineer
- LinkedIn: linkedin.com/in/hami-k-4266a01b9
- GitHub: github.com/hamedkarimi3
- Email: hamikarimi.ai@gmail.com
```


