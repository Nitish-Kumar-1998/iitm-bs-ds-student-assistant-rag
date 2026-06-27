# 🚀 Local Setup Guide — IITM BS RAG Assistant

This guide walks you through setting up the entire RAG system on your machine for development.

---

## ⚙️ Prerequisites

Before starting, ensure you have:

1. **Python 3.11+** — [Download](https://www.python.org/downloads/)
2. **Node.js 18+** — [Download](https://nodejs.org/)
3. **Git** — [Download](https://git-scm.com/)
4. **Docker** (optional, for Qdrant) — [Download](https://www.docker.com/)

### Check Installations
```bash
python --version      # Should be 3.11 or higher
node --version        # Should be 18 or higher
git --version         # Any recent version
```

---

## 📋 Step 1: Clone the Repository

```bash
git clone https://github.com/Nitish-Kumar-1998/iitm-bs-ds-student-assistant-rag.git
cd iitm-bs-ds-student-assistant-rag
```

---

## 🔑 Step 2: Get API Keys

You need **two free API keys** to run this locally:

### A. Groq API Key (LLM)
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up with your email
3. Go to **API Keys** section
4. Create a new API key
5. Copy it (you'll use it soon)

### B. Google Gemini API Key (Embeddings)
1. Visit [ai.google.dev](https://ai.google.dev)
2. Click **Get API Key**
3. Create a new project or use existing
4. Generate an API key
5. Copy it (you'll use it soon)

---

## 🗄️ Step 3: Set Up Qdrant (Vector Database)

### Option A: Using Docker (Recommended)
```bash
# Pull and run Qdrant container
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

**Verify it's running:**
```bash
curl http://localhost:6333/health
# Should return: {"ok":true}
```

### Option B: Without Docker (Local Install)
If you can't use Docker:
1. Download from [github.com/qdrant/qdrant/releases](https://github.com/qdrant/qdrant/releases)
2. Extract the binary
3. Run it:
   ```bash
   ./qdrant
   # Listens on http://localhost:6333
   ```

---

## 🐍 Step 4: Set Up Backend

### 4.1 Create `.env` File
In the **root directory** of the project, create a `.env` file:

```bash
# In the root folder, create .env
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
EOF
```

Replace `your_groq_api_key_here` and `your_gemini_api_key_here` with your actual keys.

### 4.2 Create Python Virtual Environment
```bash
cd app
python -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
venv\Scripts\activate
```

### 4.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.4 Run Backend Server
```bash
python main.py
# Or:
uvicorn main:app --reload --port 8000
```

You should see:
```
✅ Gemini ready: models/gemini-embedding-001
✅ Qdrant connected
✅ LLM client ready: llama-3.3-70b-versatile
✅ Backend ready
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Backend is now running at:** `http://localhost:8000`

---

## ⚛️ Step 5: Set Up Frontend

### 5.1 Install Dependencies
Open a **new terminal** (keep backend running in the first one):
```bash
cd frontend
npm install
```

### 5.2 Start Dev Server
```bash
npm run dev
```

You should see:
```
  ▲ Next.js 16.1.6
  - Local:        http://localhost:3000
```

**Frontend is now running at:** `http://localhost:3000`

---

## 📚 Step 6: Load Data (One-Time Setup)

Once backend and frontend are running, you need to populate the vector database with programme documents.

### 6.1 Scrape Google Docs
```bash
cd app
python scraper.py
```

**What it does:**
- Fetches official IITM BS Google Docs
- Converts them to markdown
- Saves to `app/output/docs/`
- Takes 2-5 minutes

**Output:**
```
✅ SCRAPING COMPLETE
  Documents scraped:    2
  Images saved:         15
  Reference links:      45 (enriched)
```

### 6.2 Chunk Documents
```bash
python chunker.py
```

**What it does:**
- Splits documents into 512-char chunks with 50-char overlap
- Extracts tables, images, reference links
- Saves to `app/output/chunks/all_chunks.json`

### 6.3 Generate Embeddings
```bash
python embedder.py
```

**What it does:**
- Creates vector embeddings for each chunk using Gemini
- Takes ~2-3 minutes

### 6.4 Upload to Qdrant
```bash
python uploader.py
```

**What it does:**
- Pushes all chunks to Qdrant with both dense (vector) and sparse (BM25) indices
- Ready for hybrid search

**Verify:** Check Qdrant health
```bash
curl http://localhost:6333/collections/iitm_bs
# Should show collection with points count
```

---

## 🧪 Step 7: Test Everything

### Test Backend Health
```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "ok",
  "llm_provider": "groq",
  "llm_model": "llama-3.3-70b-versatile",
  "embed_model": "models/gemini-embedding-001",
  "qdrant": "ok — 450 points"
}
```

### Test in Browser
1. Open **http://localhost:3000**
2. Type a question like: *"What are the fees for the BS degree?"*
3. You should see:
   - Streaming response from the LLM
   - Source citations with links
   - Loading indicators

---

## 🛠️ Troubleshooting

### Backend won't start
**Error:** `GROQ_API_KEY not set`
```bash
# Make sure .env is in the root folder (not app/)
# Verify keys are correct:
cat .env
```

**Error:** `Connection refused — localhost:6333`
```bash
# Start Qdrant:
docker start qdrant
# Or verify it's running:
docker ps | grep qdrant
```

### Frontend won't load
**Error:** `Failed to fetch from /ask`
```bash
# Ensure backend is running:
curl http://localhost:8000/health

# Check frontend is pointing to correct backend:
# In frontend/hooks/useChat.ts, verify API_URL = 'http://localhost:8000'
```

**Error:** `npm: command not found`
```bash
# Install Node.js from https://nodejs.org/
node --version  # Verify
```

### Qdrant connection fails
**Error:** `Connection refused`
```bash
# Check if Qdrant is running:
curl http://localhost:6333/health

# If not, start it:
docker run -d -p 6333:6333 qdrant/qdrant:latest

# Wait 5 seconds, then test again
sleep 5
curl http://localhost:6333/health
```

---

## 📁 Project Structure

```
.
├── app/                    # Backend (Python/FastAPI)
│   ├── main.py            # API server
│   ├── config.py          # Settings
│   ├── scraper.py         # Download docs
│   ├── chunker.py         # Split documents
│   ├── embedder.py        # Create embeddings
│   ├── uploader.py        # Push to Qdrant
│   ├── requirements.txt    # Python dependencies
│   └── output/            # Generated files (chunks, images, etc.)
│
├── frontend/              # Frontend (Next.js/React)
│   ├── app/
│   │   └── page.tsx       # Main chat interface
│   ├── components/        # React components
│   ├── hooks/             # useChat hook for API calls
│   ├── package.json       # Node dependencies
│   └── next.config.ts     # Next.js config
│
├── .env                   # API keys (create this!)
├── Dockerfile             # Docker image definition
└── README.md              # This file
```

---

## 🚀 Common Commands

### Start Everything (after first-time setup)

**Terminal 1 - Backend:**
```bash
cd app
source venv/bin/activate  # or: .\venv\Scripts\activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal 3 - Monitor (optional):**
```bash
# Check if services are running
curl http://localhost:8000/health    # Backend
curl http://localhost:3000            # Frontend (in browser)
```

### Clear and Restart
```bash
# Clear all generated data
rm -rf app/output
rm -rf app/output/chunks

# Restart services
# (make sure Qdrant is still running)

# Re-run pipeline
cd app
python scraper.py && python chunker.py && python embedder.py && python uploader.py
```

### Run Tests/Evaluation
```bash
cd app
python evaluator.py
# Generates: eval_report.json with retrieval + answer quality metrics
```

---

## 📊 Environment Variables Explained

| Variable | Purpose | Example |
|----------|---------|---------|
| `GROQ_API_KEY` | LLM API for generating answers | `gsk_xxxxx...` |
| `GEMINI_API_KEY` | Google Gemini for embeddings | `AIzaSy...` |
| `QDRANT_URL` | Vector database connection | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant auth (empty for local) | `` |

---

## ✅ Success Checklist

After setup, verify:

- [ ] Backend starts without errors (`python main.py`)
- [ ] Frontend loads (`http://localhost:3000`)
- [ ] Backend health check passes (`curl http://localhost:8000/health`)
- [ ] Qdrant is running (`curl http://localhost:6333/health`)
- [ ] Data pipeline completed (all scripts ran)
- [ ] You can ask a question and get a response
- [ ] Responses include source citations

---

## 📞 Need Help?

If something doesn't work:

1. **Check logs** — Look at terminal output for error messages
2. **Verify ports** — Ensure 3000 (frontend), 8000 (backend), 6333 (Qdrant) are free
3. **Check API keys** — Make sure `.env` has valid Groq + Gemini keys
4. **Restart services** — Kill and restart backend/frontend
5. **Docker issues** — Try `docker restart qdrant`

---

## 🎯 Next Steps

Once everything is running locally:

- **Modify system prompt** → Edit `SYSTEM_PROMPT` in `app/main.py`
- **Customize UI** → Edit components in `frontend/components/`
- **Add new features** → Add endpoints to `app/main.py`
- **Deploy** → Use Docker compose or cloud platforms (Vercel for frontend, Railway/Render for backend)

---

**Happy coding! 🎉**
