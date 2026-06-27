# 🚀 IITM BS RAG Assistant

A RAG-powered AI assistant that helps IITM BS Data Science students answer academic queries using official programme documents.

---

## ⚡ Quick Start (5 Minutes)

### Prerequisites
You need **3 things installed**:

1. **Docker Desktop** — [Download](https://www.docker.com/products/docker-desktop)
   - This runs the backend + database in containers
   - No need to install Python separately!

2. **Node.js** — [Download](https://nodejs.org/) (LTS version)
   - For the frontend website

3. **Git** — [Download](https://git-scm.com/)
   - To download the code

---

## 📥 Setup Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nitish-Kumar-1998/iitm-bs-ds-student-assistant-rag.git
cd iitm-bs-ds-student-assistant-rag
```

---

### Step 2: Get Your Free API Keys

**Groq API Key** (for AI answers):
1. Go to https://console.groq.com/keys
2. Sign up → Create API Key
3. Copy the key

**Gemini API Key** (for understanding questions):
1. Go to https://ai.google.dev/
2. Click "Get API Key" → Create
3. Copy the key

---

### Step 3: Create `.env` File

Copy the example file and add your keys:

```bash
# Copy template
cp .env.example .env
```

Then edit `.env` and replace:
```
GROQ_API_KEY=paste_your_groq_key_here
GEMINI_API_KEY=paste_your_gemini_key_here
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
```

---

### Step 4: Start Backend + Database

```bash
docker-compose up -d
```

**What happens:**
- Docker downloads and starts **2 containers**:
  - `qdrant` = Vector database (stores programme data)
  - `backend` = FastAPI server (answers questions)
- Takes 30-60 seconds

**Verify it worked:**
```bash
docker-compose ps
```

You should see:
```
NAME      COMMAND      STATUS
qdrant    "..."        Up (healthy)
backend   "..."        Up
```

✅ If both say "Up", you're good!

---

### Step 5: Start Frontend

Open a **new terminal** in the same folder:

```bash
cd frontend
npm install
npm run dev
```

Wait 1-2 minutes for it to start. You'll see:
```
  ▲ Next.js 16.1.6
  - Local:        http://localhost:3000
```

✅ Frontend is running!

---

### Step 6: Load Programme Data

Open **another new terminal** in the root folder:

**On Mac/Linux:**
```bash
bash load_data.sh
```

**On Windows:**
```bash
load_data.bat
```

**What happens:**
- Scrapes official IITM BS Google Docs
- Splits documents into searchable chunks
- Creates embeddings (AI converts text to vectors)
- Uploads to database

**Expected output:**
```
================================
Step 1/4: Scraping Google Docs...
✅ Scraping complete

Step 2/4: Chunking documents...
✅ Chunking complete

Step 3/4: Generating embeddings...
✅ Embeddings complete

Step 4/4: Uploading to Qdrant...
✅ Upload complete

🎉 Data pipeline complete!
================================
```

This takes **5-10 minutes** (only runs once).

---

## 🌐 Try It Out!

1. Open your browser
2. Go to: **http://localhost:3000**
3. Ask a question like:
   - *"What are the fees for BS degree?"*
   - *"What courses are available?"*
   - *"How long does the programme take?"*

You'll see:
- ✅ Streaming AI response
- ✅ Source citations with links
- ✅ Loading indicators

---

## 🛑 Stop Everything

When you're done:

```bash
docker-compose down
```

To start again later:
```bash
docker-compose up -d
npm run dev    # in frontend folder
```

---

## 🐳 How Docker Works (Explanation)

**Without Docker:**
- Install Python 3.11
- Install FastAPI, Qdrant client, Gemini library, etc.
- Run Qdrant locally
- Set up virtual environment
- Complex setup! 😵

**With Docker:**
- Docker containers = boxes with everything inside
- Your `docker-compose.yml` says: "Create 2 boxes"
- Box 1: Python + FastAPI + all libraries (backend)
- Box 2: Qdrant database
- Docker downloads everything automatically
- You just run: `docker-compose up -d`
- Simple! ✅

**What you have:**
```
Your Computer
├── Code (git clone)
├── .env file (your API keys)
├── docker-compose.yml (tells Docker what to run)
└── Docker Desktop App (manages containers)
```

When you run `docker-compose up -d`:
```
Docker reads docker-compose.yml
  ↓
Downloads backend image (contains Python + FastAPI + all libraries)
  ↓
Downloads Qdrant image
  ↓
Creates 2 containers from those images
  ↓
Starts them on your computer
  ↓
Now you can use http://localhost:8000 (backend)
and http://localhost:6333 (database)
```

---

## 🆘 Troubleshooting

### "Docker is not installed" or "docker: command not found"
```bash
# Download and install Docker Desktop
# https://www.docker.com/products/docker-desktop
# Then restart your terminal
```

### "Backend container won't start"
```bash
# Check Docker is running (open Docker Desktop)
# Then check the error:
docker-compose logs backend

# Restart:
docker-compose restart backend
```

### "npm: command not found"
```bash
# Install Node.js from https://nodejs.org/
# Then restart terminal
```

### "Failed to fetch from backend"
```bash
# Make sure backend is running:
docker-compose ps

# If not, start it:
docker-compose up -d

# Wait 10 seconds then refresh browser
```

### "API keys not working"
```bash
# Make sure .env file exists and has your real keys:
cat .env

# Should show your actual keys, not placeholders
# If not, edit .env with correct keys

# Restart backend:
docker-compose restart backend
```

---

## 📁 Project Structure

```
iitm-bs-ds-student-assistant-rag/
├── app/                      # Backend (Python/FastAPI)
│   ├── main.py              # API server
│   ├── scraper.py           # Download docs
│   ├── chunker.py           # Split documents
│   ├── embedder.py          # Create embeddings
│   ├── uploader.py          # Upload to Qdrant
│   └── requirements.txt      # Python dependencies
│
├── frontend/                # Frontend (Next.js/React)
│   ├── app/
│   │   └── page.tsx         # Chat interface
│   ├── components/          # UI components
│   └── package.json         # Node dependencies
│
├── docker-compose.yml       # Docker config (runs containers)
├── Dockerfile               # How to build backend container
├── .env.example            # Template for API keys
├── load_data.sh            # Automation for Mac/Linux
├── load_data.bat           # Automation for Windows
└── README.md               # This file
```

---

## 🎯 Common Commands

```bash
# Start everything
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs backend
docker-compose logs qdrant

# Stop everything
docker-compose down

# Restart a service
docker-compose restart backend

# Load data (Mac/Linux)
bash load_data.sh

# Load data (Windows)
load_data.bat

# Start frontend
cd frontend && npm run dev

# Delete all data and restart
docker-compose down -v
docker-compose up -d
```

---

## ✅ Success Checklist

- [ ] Docker Desktop installed and running
- [ ] Node.js installed (`node --version` works)
- [ ] Repository cloned
- [ ] `.env` file created with your API keys
- [ ] `docker-compose up -d` shows 2 containers up
- [ ] `npm run dev` starts frontend at localhost:3000
- [ ] `load_data.sh` (or `.bat`) runs without errors
- [ ] Can ask questions and get answers at localhost:3000

---

## 📞 Need Help?

1. Check the logs: `docker-compose logs backend`
2. Make sure Docker Desktop is running
3. Verify `.env` has correct API keys
4. Try restarting: `docker-compose restart backend
---

**Happy coding! 🎉**
