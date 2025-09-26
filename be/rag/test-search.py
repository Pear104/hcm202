# tools/test_search.py
import os
import sys
import textwrap
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
    HAVE_CE = True
except Exception:
    HAVE_CE = False

# -------- Config --------
load_dotenv()
QDRANT_URL = (os.getenv("QDRANT_URL") or "http://localhost:6333").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = os.getenv("COLLECTION_NAME") or "MLN131_2"   # đổi nếu bạn dùng tên khác
TOP_K = int(os.getenv("TOP_K") or 5)
USE_RERANK = (os.getenv("USE_RERANK") or "false").lower() == "true"

print(f"[cfg] QDRANT_URL={QDRANT_URL}  COLLECTION={COLLECTION}  TOP_K={TOP_K}  RERANK={USE_RERANK}")

# -------- Clients & models --------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0, check_compatibility=False)

# model E5-base (768-dim) — khớp với bộ đã index
embed_model = SentenceTransformer("intfloat/multilingual-e5-base")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if (USE_RERANK and HAVE_CE) else None

def truncate(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ").strip()
    return (s[: n-1] + "…") if len(s) > n else s

def search_once(query: str, top_k: int = TOP_K):
    # 1) encode query (E5 dùng prefix "query: ")
    qvec = embed_model.encode("query: " + query, normalize_embeddings=True)

    # 2) Qdrant search
    try:
        hits = qdrant.search(
            collection_name=COLLECTION,
            query_vector=qvec,
            limit=max(top_k, 20 if cross_encoder else top_k),
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        print(f"[ERR] Qdrant search failed: {e}")
        print("     → Kiểm tra QDRANT_URL, service Qdrant (port 6333), và collection tồn tại.")
        sys.exit(1)

    if not hits:
        print("[info] Không có kết quả (collection rỗng hoặc query không khớp).")
        return

    # 3) Optional rerank bằng cross-encoder
    if cross_encoder:
        pairs = [(query, h.payload.get("content","")) for h in hits]
        scores = cross_encoder.predict(pairs)
        scored = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:top_k]
        print(f"\n[Result] Top {top_k} (đã rerank bằng cross-encoder):")
        for rank, (h, s) in enumerate(scored, 1):
            fname = h.payload.get("filename", "unknown")
            snip  = truncate(h.payload.get("content", ""))
            print(f"{rank:>2}. [{s:.3f}] {fname}  |  {snip}")
    else:
        print(f"\n[Result] Top {top_k} (theo điểm Qdrant):")
        for i, h in enumerate(hits[:top_k], 1):
            fname = h.payload.get("filename", "unknown")
            snip  = truncate(h.payload.get("content", ""))
            # Với cosine, score càng CAO càng tốt
            print(f"{i:>2}. [{h.score:.3f}] {fname}  |  {snip}")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        q = " ".join(sys.argv[1:])
    else:
        # Query mẫu — bạn thay bằng câu của bạn
        q = "Dân chủ xã hội chủ nghĩa khác gì dân chủ tư sản?"

    print("\nQuery:", q)
    # ping nhẹ để sớm phát hiện sai URL/key
    try:
        cols = qdrant.get_collections()
        print("[ping] get_collections OK. Số collection:", len(cols.collections))
    except UnexpectedResponse as e:
        print(f"[WARN] get_collections failed: {e}")

    search_once(q)
