import os
import json
import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from dotenv import load_dotenv
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from httpx import Timeout
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False
    
# --- Setup Environment ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# === PATHS / COLLECTION ===
COLLECTION_NAME = "MLN131_2"
DATA_FOLDER = "./doc"  

# --- Load Models ---
model = SentenceTransformer("intfloat/multilingual-e5-base")

# --- Download NLTK models ---
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize


# --- Helpers ------------------------------------------------------------
def read_text_with_fallback(full_path: str) -> str:
    """Đọc file text với nhiều mã hóa thường gặp ở Windows/Vietnamese."""
    encodings = ["utf-8", "utf-8-sig", "cp1258", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            with open(full_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # Cuối cùng: đọc bỏ lỗi ký tự
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
        
def extract_pdf_text(full_path: str) -> str:
    """Trích văn bản từ PDF. Cần PyPDF2."""
    if not PYPDF2_AVAILABLE:
        print(f"[WARN] PyPDF2 chưa cài, bỏ qua PDF: {full_path}")
        return ""
    try:
        reader = PdfReader(full_path)
        pages = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
        return "\n".join(pages)
    except Exception as e:
        print(f"[WARN] Lỗi đọc PDF {full_path}: {e}")
        return ""

# --- Smart Splitter: Sentence-based chunking with overlap ---
def split_into_chunks(text: str, max_words: int = 120, overlap_words: int = 30) -> List[str]:
    """Chunk theo câu + overlap đơn giản (dự phòng)."""
    sentences = sent_tokenize(text.strip())
    chunks, cur, words = [], [], 0
    for sent in sentences:
        w = len(sent.split())
        if words + w <= max_words:
            cur.append(sent); words += w
        else:
            if cur:
                chunks.append(" ".join(cur))
            # overlap
            back, bw = [], 0
            for j in range(len(cur)-1, -1, -1):
                bw += len(cur[j].split())
                if bw >= overlap_words: break
                back.insert(0, cur[j])
            cur = back + [sent]
            words = sum(len(s.split()) for s in cur)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# --- senmatic chunking with cosine similarity
def split_into_semantic_chunks(text: str,
                               sim_threshold: float = 0.85,
                               max_words: int = 180,
                               overlap_words: int = 40) -> List[str]:
    """
    Tách theo ngữ nghĩa: nếu similarity giữa câu kề nhau < sim_threshold thì ngắt.
    Đồng thời cưỡng bức ngắt khi chunk vượt quá max_words (có overlap).
    """
    # 1) Cắt câu chắc chắn bằng NLTK
    sentences = [s.strip() for s in sent_tokenize(text.strip()) if s.strip()]
    if not sentences:
        return []

    # 2) Embed từng câu (nhẹ hơn embed 'grouped')
    sent_embs = model.encode(sentences, normalize_embeddings=True)

    chunks = []
    cur_sents = [sentences[0]]
    cur_words = len(sentences[0].split())

    for i in range(1, len(sentences)):
        # similarity của câu i với câu i-1
        sim = float(np.dot(sent_embs[i], sent_embs[i-1]))  # vì đã normalize
        need_split = (sim < sim_threshold) or (cur_words + len(sentences[i].split()) > max_words)

        if need_split:
            # kết thúc chunk hiện tại
            chunks.append(" ".join(cur_sents))

            # overlap: giữ lại phần đuôi
            if overlap_words > 0:
                tail = []
                wsum = 0
                for s in reversed(cur_sents):
                    wsum += len(s.split())
                    tail.insert(0, s)
                    if wsum >= overlap_words:
                        break
                cur_sents = tail + [sentences[i]]
            else:
                cur_sents = [sentences[i]]

            cur_words = sum(len(s.split()) for s in cur_sents)
        else:
            cur_sents.append(sentences[i])
            cur_words += len(sentences[i].split())

    if cur_sents:
        chunks.append(" ".join(cur_sents))

    return [c for c in (c.strip() for c in chunks) if c]


def iter_source_files(folder_path: str):
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".json", ".txt", ".md", ".pdf"):
            yield filename, ext
       
def load_file_content(full_path: str, ext: str) -> str:
    if ext == ".pdf":
        return extract_pdf_text(full_path)
    if ext == ".json":
        txt = read_text_with_fallback(full_path)
        try:
            data = json.loads(txt)
            return data.get("text") or data.get("content") or txt
        except Exception:
            return txt
    # .txt / .md
    return read_text_with_fallback(full_path)     
            
# --- Load and Embed Documents ---
def load_and_embed_documents_v2(folder_path: str) -> List[Tuple[str, str, List[float]]]:
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Thư mục không tồn tại: {folder_path}")

    documents: List[Tuple[str, str, List[float]]] = []

    for filename, ext in iter_source_files(folder_path):
        full_path = os.path.join(folder_path, filename)
        content = load_file_content(full_path, ext)
        if not content or not content.strip():
            print(f"[SKIP] Rỗng hoặc không đọc được: {filename}")
            continue

        for sec_idx, section in enumerate([content]):
            chunks = split_into_semantic_chunks(section, sim_threshold=0.85, max_words=180, overlap_words=40)
            if not chunks:
                chunks = split_into_chunks(section, max_words=180, overlap_words=40)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}#s{sec_idx+1}_c{chunk_idx+1}"
                emb = model.encode("passage: " + chunk)
                documents.append((chunk_id, chunk, emb))
                print(f"✅ {chunk_id} done.")

    return documents

# --- Upload Chunks to Qdrant ---
def upload_to_qdrant(documents: List[Tuple[str, str, List[float]]], batch_size: int = 256):
    if not documents:
        print("[WARN] Không có document nào để upload.")
        return

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)

    # Xóa & tạo lại collection
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # bge-m3 = 1024
    )

    total = len(documents)
    print(f"🔄 Uploading {total} vectors in batches of {batch_size}")

    for i in range(0, total, batch_size):
        batch = documents[i:i+batch_size]
        points = [
            PointStruct(
                id=i + j,
                vector=embedding,
                payload={"filename": name, "content": content}
            )
            for j, (name, content, embedding) in enumerate(batch)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Uploaded {i+len(points)}/{total}")

# --- Main execution ---
if __name__ == "__main__":
    docs = load_and_embed_documents_v2(DATA_FOLDER)
    upload_to_qdrant(docs)
