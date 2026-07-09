
```
# 🎫 RAG Support Ticket Search & Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that helps support engineers quickly find relevant tickets and get AI-generated answers to customer issues.

Built as a technical demonstration of RAG architecture using Python, LangChain, ChromaDB, and OpenAI GPT-4o.

---

## 🎯 What It Does

Instead of keyword search, this system:
1. Converts support tickets into vector embeddings
2. When a user asks a question, finds the most semantically similar tickets
3. Feeds those tickets as context to GPT-4o
4. Returns an accurate, grounded AI response

---

## 🏗️ Architecture

```
User Query
    ↓
Embed query (OpenAI text-embedding-ada-002)
    ↓
Similarity search (ChromaDB)
    ↓
Retrieved tickets as context
    ↓
GPT-4o generates answer
    ↓
Display answer + relevant tickets
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Database | ChromaDB |
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
│   └── rag_chain.py         # RAG pipeline with GPT-4o
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

4. Add your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

5. Run the app:
```bash
streamlit run app.py
```

### Docker Setup

```bash
docker build -t rag-support-tickets .
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key-here rag-support-tickets
```

---

## 💡 Key Features

- **Semantic Search** — finds relevant tickets by meaning, not just keywords
- **Multi-format Support** — loads both JSON and XML ticket files
- **Persistent Storage** — vector store saved to disk, no re-embedding on restart
- **Filter by Type** — search within technical, product, or customer tickets
- **Similarity Scores** — shows how relevant each retrieved ticket is
- **Production Ready** — proper error handling, logging, and Docker support

---

## 🔧 How RAG Works Here

**Without RAG:**
> User asks "Chrome login error" → GPT-4o answers from general knowledge

**With RAG:**
> User asks "Chrome login error" → System finds ticket T002 about Chrome 403 error → GPT-4o answers using that specific ticket as context → More accurate, grounded answer

---

## 👤 Author

Hami Karimi — Senior Data & AI Engineer
- LinkedIn: linkedin.com/in/hami-k-4266a01b9
- Email: hamikarimi.ai@gmail.com
```

