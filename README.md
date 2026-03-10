# 📚 N8n RAG Chatbot

AI-powered Q&A over N8n documentation PDFs, built with Claude + PyMuPDF + HF Embeddings.

## Stack
- **Backend:** Flask (Python)
- **AI:** Claude claude-sonnet-4 (Anthropic)
- **Embeddings:** HF Inference API (`all-MiniLM-L6-v2`)
- **PDF Parsing:** PyMuPDF (fitz)
- **Hosting:** Render.com

---

## 🚀 Deploy to Render

### 1. Fork / push this repo to GitHub

### 2. Create a Render account
Go to [render.com](https://render.com) → sign up free

### 3. New Web Service → Connect GitHub repo
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- **Runtime:** Python 3

### 4. Set environment variables in Render dashboard
| Key | Value |
|-----|-------|
| `ANTHROPIC_API_KEY` | your Claude API key |
| `GDRIVE_FOLDER_ID` | `1vfPGyeDHCWz8wMQumz_XQMoQVzFSf8Cw` |

### 5. Deploy
Render builds and deploys automatically. Your URL will be:
```
https://n8n-rag-chatbot.onrender.com
```

---

## 📁 Project Structure
```
├── app.py              # Flask app — API + serves frontend
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── static/
│   └── index.html      # Frontend UI (served by Flask)
└── .gitignore
```

## ⚠️ Notes
- On Render free tier, the server **sleeps after 15 minutes** of inactivity
- First request after sleep takes ~30 seconds to wake up
- PDFs are re-downloaded and re-indexed on every cold start
- To avoid re-indexing every time, upgrade to Render paid tier with a persistent disk
