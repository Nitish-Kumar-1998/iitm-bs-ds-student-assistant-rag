# 🎓 IITM BS RAG Assistant

A RAG-powered AI assistant that helps IITM BS Data Science students get instant answers from official programme documents.

🌐 **Live App:** https://iitm-bs-ds-student-assistant-rag.vercel.app/
⚙️ **Backend API:** https://iitm-bs-ds-student-assistant-rag.onrender.com

---

## 🧠 How It Works

```
Your Question
     ↓
Gemini Embeddings (convert question to vector)
     ↓
Qdrant Vector DB (hybrid search: 70% vector + 30% BM25)
     ↓
Top 20 relevant chunks retrieved
     ↓
Groq LLM — llama-3.3-70b-versatile (generates answer)
     ↓
Streaming response with source citations
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 16, React 19, TypeScript |
| Backend | FastAPI, Python 3.11, Uvicorn |
| Vector DB | Qdrant v1.17.0 |
| LLM | Groq — llama-3.3-70b-versatile |
| Embeddings | Gemini — models/gemini-embedding-001 (3072 dim) |
| Search | Hybrid: vector (0.7) + BM25 (0.3) |
| Containers | Docker, Docker Compose |
| CI/CD | GitHub Actions → Render (auto-deploy on push) |

---

## ⚡ Quick Start (Local)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Node.js 22+](https://nodejs.org/) (LTS)
- [Git](https://git-scm.com/)

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nitish-Kumar-1998/iitm-bs-ds-student-assistant-rag.git
cd iitm-bs-ds-student-assistant-rag
```

---

### Step 2: Get Your Free API Keys

**Groq API Key** (LLM):
1. Go to https://console.groq.com/keys
2. Sign up → Create API Key

**Gemini API Key** (Embeddings):
1. Go to https://ai.google.dev/
2. Click "Get API Key" → Create

---

### Step 3: Create `.env` File

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
```

---

### Step 4: Start Backend + Database

```bash
docker compose up -d
```

Verify:
```bash
docker compose ps
```

Expected output:
```
NAME      STATUS
qdrant    Up (healthy)
backend   Up (healthy)
```

---

### Step 5: Load Programme Data

**Mac/Linux:**
```bash
bash load_data.sh
```

**Windows:**
```bash
load_data.bat
```

This runs once and takes 5–10 minutes. It:
- Scrapes official IITM BS Google Docs
- Chunks documents into 591 searchable pieces
- Generates Gemini embeddings
- Uploads to Qdrant

> ⚠️ Gemini free tier allows 1000 embed requests/day. If it fails, wait 24 hours and retry.

---

### Step 6: Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Create `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Open **http://localhost:3000** and start asking questions!

---

## 🛑 Stop Everything

```bash
docker compose down        # stop containers (data preserved)
docker compose down -v     # stop + delete all data (need to re-run load_data.sh)
```

---

## 🔄 Start Again Later

```bash
# Terminal 1
docker compose up -d

# Terminal 2
cd frontend && npm run dev
```

No need to run `load_data.sh` again — data persists in the Docker volume.

---

## 📁 Project Structure

```
iitm-bs-ds-student-assistant-rag/
├── app/                        # Backend (Python/FastAPI)
│   ├── main.py                # API server + RAG pipeline
│   ├── scraper.py             # Scrape IITM BS Google Docs
│   ├── chunker.py             # Split documents into chunks
│   ├── embedder.py            # Generate Gemini embeddings
│   ├── uploader.py            # Upload to Qdrant
│   └── requirements.txt       # Python dependencies
│
├── frontend/                  # Frontend (Next.js/React)
│   ├── app/
│   │   └── page.tsx           # Chat interface
│   └── components/            # UI components
│
├── .github/workflows/         # CI/CD
│   └── deploy.yml             # Auto-deploy to Render on push
│
├── docker-compose.yml         # Docker config
├── Dockerfile                 # Backend container build
├── .env.example               # API key template
├── load_data.sh               # Data pipeline (Mac/Linux)
├── load_data.bat              # Data pipeline (Windows)
└── README.md
```

---

## 🎯 Common Commands

```bash
# Start containers
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs backend
docker compose logs qdrant

# Restart backend
docker compose restart backend

# Stop containers
docker compose down

# Start frontend
cd frontend && npm run dev

# Load data (Mac/Linux)
bash load_data.sh

# Load data (Windows)
load_data.bat
```

---

## 🆘 Troubleshooting

### Docker pull fails with EOF
Pinned image versions are already set in `docker-compose.yml`. If you see EOF errors:
- Add DNS to Docker Desktop → Settings → Docker Engine: `"dns": ["8.8.8.8", "8.8.4.4"]`
- Avoid using `:latest` tags

### Qdrant shows unhealthy
The healthcheck uses a TCP check (no curl inside container). If still failing:
```bash
docker logs qdrant
```

### Backend shows unhealthy but app works
```bash
curl http://localhost:8000/health
```
If it returns JSON, the backend is fine — the Docker healthcheck is just a TCP port check.

### Embedding fails after retries
Gemini free tier quota (1000/day) exceeded. Wait 24 hours or use a new Google account's API key.

### Frontend can't reach backend
Make sure `frontend/.env.local` exists with:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```
Then restart: `Ctrl+C` → `npm run dev`

### New `.env` values not loading
`docker compose restart` doesn't reload env. Use:
```bash
docker compose down && docker compose up -d
```

---

## ✅ Success Checklist

- [ ] Docker Desktop installed and running
- [ ] Node.js 22+ installed
- [ ] Repository cloned
- [ ] `.env` created with API keys
- [ ] `docker compose up -d` → both containers healthy
- [ ] `load_data.sh` completed successfully
- [ ] `frontend/.env.local` created
- [ ] `npm run dev` running at localhost:3000
- [ ] Questions answered correctly at localhost:3000

---

## 📦 Version

**v1.0.0** — First stable local release

---

**Happy learning! 🎉**