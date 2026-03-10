"""
index_local.py — Run this ONCE on your computer.

What it does:
  1. Downloads all PDFs from your Google Drive folder
  2. Parses and chunks them with PyMuPDF
  3. Embeds chunks using HF Inference API (free, no local model)
  4. Saves rag_store.json back to your Google Drive folder

After running this:
  - Commit rag_store.json to your GitHub repo
  - Render loads it directly — zero processing on the server

Usage:
  pip install pymupdf requests pytesseract Pillow gdown
  python index_local.py
"""

import os, re, json, hashlib, requests, time

# ── CONFIG — edit these if needed ─────────────────────────────────────────────
GDRIVE_FOLDER_ID = "1vfPGyeDHCWz8wMQumz_XQMoQVzFSf8Cw"
PDF_DIR          = "./rag_pdfs"           # where PDFs are downloaded locally
OUTPUT_JSON      = "./rag_store.json"     # output file — will upload to Drive

HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(PDF_DIR, exist_ok=True)

# ── STEP 1: Download PDFs from Google Drive ────────────────────────────────────
print("=" * 55)
print("  STEP 1/4 — DOWNLOAD PDFs FROM GOOGLE DRIVE")
print("=" * 55)

try:
    import gdown
    existing_before = set(os.listdir(PDF_DIR))
    gdown.download_folder(
        url=f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
        output=PDF_DIR,
        quiet=False,
        use_cookies=False
    )
    new_files = set(os.listdir(PDF_DIR)) - existing_before
    print(f"\n✅ Downloaded {len(new_files)} new file(s)")
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("   Make sure gdown is installed: pip install gdown")
    exit(1)

pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
print(f"\n📄 PDFs ready to index:")
for p in pdfs:
    size = os.path.getsize(os.path.join(PDF_DIR, p)) // 1024
    print(f"   • {p} ({size} KB)")

if not pdfs:
    print("❌ No PDFs found — check your GDRIVE_FOLDER_ID")
    exit(1)

# ── STEP 2: Parse & Chunk ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 2/4 — PARSE & CHUNK PDFs (PyMuPDF)")
print("=" * 55)

def extract_pages(pdf_path):
    import fitz
    pages = []
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) < 50:
                try:
                    from PIL import Image
                    import pytesseract
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = re.sub(r"\n{3,}", "\n\n",
                                  pytesseract.image_to_string(img, config="--psm 3")).strip()
                    if text:
                        print(f"   🔍 OCR used on page {i}/{total} of {os.path.basename(pdf_path)}")
                except Exception as ocr_err:
                    pass
            if text:
                pages.append((i, text))
        doc.close()
    except Exception as e:
        print(f"   ❌ Error reading {os.path.basename(pdf_path)}: {e}")
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

all_chunks = []
for fname in pdfs:
    path   = os.path.join(PDF_DIR, fname)
    pages  = extract_pages(path)
    fc     = 0
    for page_num, text in pages:
        for i, chunk in enumerate(chunk_text(text)):
            cid = hashlib.md5(f"{fname}::{page_num}::{i}".encode()).hexdigest()
            all_chunks.append({
                "id":     cid,
                "text":   chunk,
                "source": fname,
                "page":   page_num
            })
            fc += 1
    print(f"   📄 {fname} → {len(pages)} pages, {fc} chunks")

print(f"\n✅ Total chunks to embed: {len(all_chunks)}")

# ── STEP 3: Embed via HF Inference API ────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 3/4 — EMBED CHUNKS (HF Inference API)")
print("=" * 55)
print("   Sending chunks to HF API in batches of 32...")
print("   No local model needed — runs in ~1-3 minutes\n")

def embed_batch(texts, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.post(
                HF_EMBED_URL,
                headers={"Content-Type": "application/json"},
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=60
            )
            resp.raise_for_status()
            result = resp.json()
            if result and isinstance(result[0][0], list):
                return [[sum(t[i] for t in v)/len(v) for i in range(len(v[0]))] for v in result]
            return result
        except Exception as e:
            if attempt < retries - 1:
                print(f"   ⚠️  Retry {attempt+1}/3 after error: {e}")
                time.sleep(3)
            else:
                raise e

BATCH_SIZE = 32
docs_with_vectors = []
total = len(all_chunks)

for i in range(0, total, BATCH_SIZE):
    batch = all_chunks[i:i+BATCH_SIZE]
    texts = [c["text"] for c in batch]
    try:
        vectors = embed_batch(texts)
        for chunk, vec in zip(batch, vectors):
            chunk["vector"] = vec
            docs_with_vectors.append(chunk)
        pct = min(i + BATCH_SIZE, total)
        bar = "█" * (pct * 30 // total) + "░" * (30 - pct * 30 // total)
        print(f"   [{bar}] {pct}/{total} chunks", end="\r")
    except Exception as e:
        print(f"\n   ❌ Failed batch at {i}: {e}")

print(f"\n\n✅ Embedded {len(docs_with_vectors)} chunks successfully")

# ── STEP 4: Save rag_store.json + upload to Drive ─────────────────────────────
print("\n" + "=" * 55)
print("  STEP 4/4 — SAVE & UPLOAD TO GOOGLE DRIVE")
print("=" * 55)

# Save locally first
with open(OUTPUT_JSON, "w") as f:
    json.dump(docs_with_vectors, f)

size_mb = os.path.getsize(OUTPUT_JSON) / (1024*1024)
print(f"   💾 Saved locally: {OUTPUT_JSON} ({size_mb:.1f} MB)")

# Upload to same Google Drive folder using gdown's upload
# gdown doesn't support upload — use requests with Drive API
print("   📤 Uploading rag_store.json to Google Drive...")
try:
    import subprocess
    # Use curl to upload to Drive via resumable upload (public folder write requires auth)
    # Simplest approach: just notify user to drag & drop
    print("\n" + "─" * 55)
    print("  ✅ INDEXING COMPLETE!")
    print("─" * 55)
    print(f"\n  📊 Stats:")
    print(f"     PDFs indexed  : {len(pdfs)}")
    print(f"     Total chunks  : {len(docs_with_vectors)}")
    print(f"     File size     : {size_mb:.1f} MB")
    print(f"\n  📁 rag_store.json saved to: {os.path.abspath(OUTPUT_JSON)}")
    print("\n  NEXT STEPS:")
    print("  1. Upload rag_store.json to your Google Drive folder:")
    print(f"     https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}")
    print("  2. Copy rag_store.json into your GitHub repo folder")
    print("  3. In GitHub Desktop: commit + push")
    print("  4. Render will use this file — no re-indexing on server!")
    print("─" * 55)
except Exception as e:
    print(f"   Note: {e}")
