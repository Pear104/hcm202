# rag/load_to_qdrant.py
import os, re, json
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ====== PDF extractor tốt cho VN ======
# pip install pymupdf
import fitz  # PyMuPDF

# ====== Cấu hình ======
QDRANT_URL = (os.getenv("QDRANT_URL") or "http://localhost:6333").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "HCM_TuTuong")
DATA_FOLDER = os.getenv("DATA_FOLDER", "./doc")

# GPU nếu có
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model embedding
EMBED_MODEL_NAME = "BAAI/bge-m3"  # 1024-dim
VECTOR_SIZE = 1024

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

# ====== Helpers ======
def extract_pdf_text(path: str) -> str:
    text = []
    try:
        with fitz.open(path) as doc:
            for p in doc:
                text.append(p.get_text())
        return "\n".join(text)
    except Exception as e:
        print(f"[WARN] Lỗi đọc PDF {path}: {e}")
        return ""

def read_text_with_fallback(path: str) -> str:
    encs = ["utf-8", "utf-8-sig", "cp1258", "cp1252", "latin-1"]
    for enc in encs:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def iter_source_files(folder: str):
    for fn in os.listdir(folder):
        ext = os.path.splitext(fn)[1].lower()
        if ext in (".pdf", ".txt", ".md", ".json"):
            yield fn, ext

def split_into_chunks(text: str, max_words: int = 180, overlap_words: int = 40) -> List[str]:
    # Fallback length-based (ổn định với PDF)
    words = text.strip().split()
    chunks, i = [], 0
    while i < len(words):
        j = min(i + max_words, len(words))
        chunk = " ".join(words[i:j])
        chunks.append(chunk)
        i = j - overlap_words if (j < len(words)) else j
    return [c.strip() for c in chunks if c.strip()]

def split_into_semantic_chunks(text: str, buffer_size: int = 1, threshold: float = 0.7) -> List[str]:
    # Heuristic semantic split theo câu đơn giản (không phụ thuộc NLTK)
    sents = [s.strip() for s in re.split(r'(?<=[\.\!\?…;])\s+|\n+', text.strip()) if s.strip()]
    if not sents:
        return []
    grouped = []
    for i in range(len(sents)):
        start = max(0, i - buffer_size)
        end = min(len(sents), i + buffer_size + 1)
        grouped.append(" ".join(sents[start:end]))

    embs = embed_model.encode(grouped, normalize_embeddings=True)
    sims = cosine_similarity(embs)
    distances = 1 - np.diag(sims[1:], k=-1)

    chunks, cur = [], [sents[0]]
    cur_words = len(sents[0].split())

    for i, d in enumerate(distances):
        # Tách khi similarity thấp hoặc chunk quá dài
        need_split = (1 - d) < threshold or cur_words > 180
        if need_split:
            chunks.append(" ".join(cur))
            # overlap nhẹ
            tail = cur[-2:] if len(cur) >= 2 else cur
            cur = tail + [sents[i + 1]]
            cur_words = sum(len(x.split()) for x in cur)
        else:
            cur.append(sents[i + 1])
            cur_words += len(sents[i + 1].split())

    if cur:
        chunks.append(" ".join(cur))
    # chặn trần độ dài lần cuối
    final = []
    for ch in chunks:
        split_len = split_into_chunks(ch, max_words=200, overlap_words=40)
        final.extend(split_len if split_len else [ch])
    return [c for c in final if c.strip()]

def load_file_content(path: str, ext: str) -> str:
    if ext == ".pdf":
        return extract_pdf_text(path)
    if ext == ".json":
        raw = read_text_with_fallback(path)
        try:
            data = json.loads(raw)
            return data.get("text") or data.get("content") or raw
        except Exception:
            return raw
    return read_text_with_fallback(path)  # .txt/.md

# ====== Load & Embed ======
def load_and_embed_documents(folder_path: str) -> List[Tuple[str, str, List[float]]]:
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Thư mục không tồn tại: {folder_path}")

    docs: List[Tuple[str, str, List[float]]] = []
    for fn, ext in iter_source_files(folder_path):
        path = os.path.join(folder_path, fn)
        content = load_file_content(path, ext)
        if not content.strip():
            print(f"[SKIP] Rỗng/không đọc được: {fn}")
            continue

        # Dùng semantic trước, fallback length
        chunks = split_into_semantic_chunks(content)
        if not chunks:
            chunks = split_into_chunks(content, max_words=200, overlap_words=40)

        for i, ch in enumerate(chunks, start=1):
            # bge-m3: prefix "passage: " hữu ích cho RAG
            emb = embed_model.encode("passage: " + ch)
            chunk_id = f"{fn}#c{i}"
            docs.append((chunk_id, ch, emb))
            print(f"✅ {chunk_id} done.")
    return docs

# ====== Upload Qdrant ======
def upload_to_qdrant(documents: List[Tuple[str, str, List[float]]], batch_size: int = 256):
    if not documents:
        print("[WARN] Không có document để upload.")
        return

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0, check_compatibility=False)

    # recreate collection
    try:
        if qdrant.collection_exists(COLLECTION_NAME):
            qdrant.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"[WARN] delete_collection: {e}")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

    total = len(documents)
    print(f"🔄 Uploading {total} vectors (batch={batch_size})")
    for i in range(0, total, batch_size):
        batch = documents[i:i+batch_size]
        points = [
            PointStruct(
                id=i + j,
                vector=emb,
                payload={"filename": name, "content": content}
            )
            for j, (name, content, emb) in enumerate(batch)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Uploaded {i+len(points)}/{total}")

if __name__ == "__main__":
    print(f"[cfg] QDRANT_URL={QDRANT_URL}  COLLECTION={COLLECTION_NAME}  MODEL={EMBED_MODEL_NAME}  DEVICE={DEVICE}")
    docs = load_and_embed_documents(DATA_FOLDER)
    upload_to_qdrant(docs)
