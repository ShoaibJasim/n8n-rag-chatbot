"""
app.py — RAG Chatbot for Render.com
Pre-indexed: loads rag_store.json from Drive or local file.
Zero PDF processing on server — fast cold start, low memory.
"""

import os, re, json, time, threading, hashlib, requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GDRIVE_FOLDER_ID  = os.environ.get("GDRIVE_FOLDER_ID", "1vfPGyeDHCWz8wMQumz_XQMoQVzFSf8Cw")
CLAUDE_MODEL      = "claude-sonnet-4-20250514"
PDF_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_pdfs")
STORE_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_store.json")
STATIC_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(PDF_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

sessions   = {}
ready      = False
status_log = []

def log(msg):
    print(msg, flush=True)
    status_log.append(msg)

# ── Embeddings via HF free API (no local model = no RAM spike) ────────────────
def embed(texts):
    if isinstance(texts, str): texts = [texts]
    resp = requests.post(HF_EMBED_URL,
        headers={"Content-Type": "application/json"},
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=60)
    resp.raise_for_status()
    result = resp.json()
    if result and isinstance(result[0][0], list):
        return [[sum(t[i] for t in v)/len(v) for i in range(len(v[0]))] for v in result]
    return result

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    return dot / ((sum(x*x for x in a)**0.5) * (sum(x*x for x in b)**0.5) + 1e-9)

# ── JSON vector store ─────────────────────────────────────────────────────────
def load_store():
    if os.path.exists(STORE_PATH):
        with open(STORE_PATH) as f: return json.load(f)
    return []

def save_store(docs):
    with open(STORE_PATH, "w") as f: json.dump(docs, f)

def query_store(question, k=5):
    docs = load_store()
    if not docs: return []
    q_vec  = embed([question])[0]
    scored = sorted([(cosine_sim(q_vec, d["vector"]), d) for d in docs], key=lambda x: -x[0])
    return [{"text": d["text"], "source": d["source"],
             "page": d["page"], "score": round(s,3)} for s,d in scored[:k]]

# ── PDF download ──────────────────────────────────────────────────────────────
def download_all_pdfs():
    log(f"  Folder: {GDRIVE_FOLDER_ID}")
    try:
        import gdown
        gdown.download_folder(
            url=f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
            output=PDF_DIR, quiet=False, use_cookies=False)
    except Exception as e:
        log(f"  ❌ Download error: {e}")

# ── PDF parsing (PyMuPDF) ─────────────────────────────────────────────────────
def extract_pages(pdf_path):
    """
    Uses PyMuPDF (fitz) — fastest parser, best text accuracy.
    Extracts text with layout preservation. Falls back to OCR for image-only pages.
    """
    import fitz  # pymupdf
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            # get_text("text") gives clean plain text with layout order preserved
            text = page.get_text("text")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            # If page has almost no text, try OCR via fitz's built-in pixmap
            if len(text) < 50:
                try:
                    mat  = fitz.Matrix(2.0, 2.0)   # 2x zoom = ~150 DPI
                    pix  = page.get_pixmap(matrix=mat, alpha=False)
                    # Convert pixmap to PIL image then run tesseract
                    from PIL import Image
                    import pytesseract, io
                    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = re.sub(r"\n{3,}", "\n\n",
                                  pytesseract.image_to_string(img, config="--psm 3")).strip()
                except Exception:
                    pass

            if text:
                pages.append((i, text))
        doc.close()
    except Exception as e:
        log(f"  ❌ {os.path.basename(pdf_path)}: {e}")
    return pages

def chunk_text(text, size=800, overlap=150):
    if len(text) <= size: return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in [".\n", ". ", "\n\n", "\n", " "]:
                pos = text.rfind(sep, start + overlap, end)
                if pos != -1: end = pos + len(sep); break
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        start = end - overlap
    return chunks

# ── Index builder ─────────────────────────────────────────────────────────────
def build_index():
    pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    if not pdfs:
        log("  ❌ No PDFs found"); return
    existing      = load_store()
    existing_ids  = {d["id"] for d in existing}
    new_docs      = []
    for fname in pdfs:
        path   = os.path.join(PDF_DIR, fname)
        pages  = extract_pages(path)
        chunks = []
        for page_num, text in pages:
            for i, chunk in enumerate(chunk_text(text)):
                cid = hashlib.md5(f"{fname}::{page_num}::{i}".encode()).hexdigest()
                if cid not in existing_ids:
                    chunks.append({"id": cid, "text": chunk, "source": fname, "page": page_num})
        log(f"  📄 {fname} → {len(pages)} pages, {len(chunks)} new chunks")
        for b in range(0, len(chunks), 32):
            batch = chunks[b:b+32]
            try:
                vecs = embed([c["text"] for c in batch])
                for c, v in zip(batch, vecs):
                    c["vector"] = v
                    new_docs.append(c)
            except Exception as e:
                log(f"    ❌ Embed error: {e}")
    if new_docs:
        save_store(existing + new_docs)
        log(f"  ✅ Added {len(new_docs)} chunks (total: {len(existing)+len(new_docs)})")
    else:
        log(f"  ℹ️  Already indexed ({len(existing)} chunks)")

# ── Startup — loads pre-built index, no PDF processing ───────────────────────
GDRIVE_STORE_ID = os.environ.get("GDRIVE_STORE_FILE_ID", "")  # optional: Drive file ID of rag_store.json

def startup():
    global ready
    log("🚀 Loading pre-built RAG index...")

    # Option A: rag_store.json already committed in repo (fastest)
    if os.path.exists(STORE_PATH) and os.path.getsize(STORE_PATH) > 100:
        docs  = load_store()
        files = sorted({d["source"] for d in docs})
        log(f"✅ Loaded {len(docs)} chunks from {len(files)} doc(s):")
        for f in files: log(f"   • {f}")
        ready = True
        return

    # Option B: download rag_store.json from Google Drive
    if GDRIVE_STORE_ID:
        log("  Downloading rag_store.json from Google Drive...")
        try:
            import gdown
            gdown.download(
                url=f"https://drive.google.com/uc?id={GDRIVE_STORE_ID}",
                output=STORE_PATH, quiet=False)
            docs  = load_store()
            files = sorted({d["source"] for d in docs})
            log(f"✅ Loaded {len(docs)} chunks from {len(files)} doc(s):")
            for f in files: log(f"   • {f}")
            ready = True
            return
        except Exception as e:
            log(f"  ❌ Failed to download store: {e}")

    log("❌ No rag_store.json found!")
    log("   Run index_local.py on your computer first,")
    log("   then commit rag_store.json to your GitHub repo.")
    ready = False

threading.Thread(target=startup, daemon=True).start()

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable assistant specialising in N8n workflow automation.
Answer ONLY from the context passages below.
- Be specific and cite sources: [Source: filename.pdf, page X]
- If context is insufficient, say so honestly.
- Never fabricate node names, steps, or config values.

Context:
{context}"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    html = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html):
        return send_from_directory(STATIC_DIR, "index.html")
    return f"<pre>static/index.html not found.\nSTATIC_DIR={STATIC_DIR}\nFiles: {os.listdir(os.path.dirname(STATIC_DIR))}</pre>", 404

@app.route("/health")
def health():
    docs  = load_store()
    files = sorted({d["source"] for d in docs})
    if not ready:
        return jsonify({"status":"loading","message":"Indexing...","log":status_log[-8:]}), 202
    return jsonify({"status":"ok","chunks":len(docs),"documents":files})

@app.route("/chat", methods=["POST"])
def chat():
    if not ready:
        return jsonify({"error":"Still loading — try again in a moment."}), 503
    data       = request.json or {}
    question   = data.get("question","").strip()
    session_id = data.get("session_id","default")
    if not question: return jsonify({"error":"No question"}), 400

    chunks = query_store(question)
    if not chunks: return jsonify({"answer":"No relevant content found.","sources":[]})

    context = "\n\n---\n\n".join(f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in chunks)
    if session_id not in sessions: sessions[session_id] = []
    history = sessions[session_id]
    history.append({"role":"user","content":question})

    import anthropic
    resp   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
        model=CLAUDE_MODEL, max_tokens=1024,
        system=SYSTEM_PROMPT.format(context=context),
        messages=history[-20:])
    answer = resp.content[0].text
    history.append({"role":"assistant","content":answer})
    return jsonify({"answer":answer,
                    "sources":[{"file":c["source"],"page":c["page"],"score":c["score"]} for c in chunks]})

@app.route("/clear", methods=["POST"])
def clear():
    sessions.pop((request.json or {}).get("session_id","default"), None)
    return jsonify({"status":"cleared"})

@app.route("/documents")
def documents():
    docs  = load_store()
    files = sorted({d["source"] for d in docs})
    return jsonify({"documents":files,"total_chunks":len(docs),"status":"ok" if ready else "loading"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
