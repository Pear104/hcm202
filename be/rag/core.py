# rag/core.py
import os, re, json
from typing import List
from dotenv import load_dotenv

# nạp .env TRƯỚC khi đọc env
load_dotenv()

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
import google.generativeai as genai

from .generator import generate_with_groq, generate_with_gemini

# ================== CẤU HÌNH MÔ HÌNH ==================
# GPU nếu có
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Khuyến nghị cho VN + đa ngữ, 1024-dim
EMBED_MODEL_NAME = "BAAI/bge-m3"
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

# Cross-encoder rerank (CPU/GPU tuỳ cài đặt torch; đa số chạy CPU vẫn ổn)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Gemini (dùng cho sinh câu trả lời)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_gen = genai.GenerativeModel("gemini-2.0-flash")

# Qdrant
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "HCM_TuTuong")  # bạn có thể đổi tuỳ ý

_qdrant = None
def get_qdrant():
    """Khởi tạo Qdrant client khi cần (tránh lỗi thứ tự import)."""
    global _qdrant
    if _qdrant is None:
        url = (os.getenv("QDRANT_URL") or "").rstrip("/")
        api_key = os.getenv("QDRANT_API_KEY") or None
        print(f"[rag.core] QDRANT_URL={url or '<empty>'}  COLLECTION={COLLECTION_NAME}")
        _qdrant = QdrantClient(url=url, api_key=api_key, timeout=30.0, check_compatibility=False)
    return _qdrant

# ================== TIỆN ÍCH PROMPT/ROUTER ==================
def rewrite_query(original_query: str) -> str:
    prompt = f"""
Bạn là trợ lý tối ưu truy vấn cho chatbot về Tư tưởng Hồ Chí Minh (TTHCM).

Yêu cầu:
- Viết lại câu hỏi ngắn gọn, học thuật, đúng trọng tâm TTHCM
- Ưu tiên các chủ đề: độc lập dân tộc gắn với CNXH; dân tộc–giai cấp; đại đoàn kết
  dân tộc; nhà nước của dân – do dân – vì dân; dân chủ; đạo đức cách mạng;
  giáo dục – con người; văn hoá; đối ngoại; xây dựng Đảng.
- Thêm từ đồng nghĩa/thuật ngữ tương đương có ích cho truy hồi.

Gốc:
{original_query}

Bản viết lại (một dòng, không giải thích):
"""
    try:
        rewritten = model_gen.generate_content(prompt)
        return rewritten.text.strip()
    except Exception:
        return original_query

def generate_subqueries(user_query: str, max_subqueries: int = 4) -> List[str]:
    prompt = f"""
Bạn là trợ lý tách truy vấn cho chatbot Tư tưởng Hồ Chí Minh.

Hãy tách câu hỏi dưới đây thành tối đa {max_subqueries} tiểu câu, mỗi câu chỉ 1 ý:
(vd: độc lập dân tộc gắn CNXH; dân chủ; đạo đức cách mạng; đại đoàn kết;
nhà nước của dân – do dân – vì dân; văn hoá; giáo dục con người; xây dựng Đảng).

Không trùng lặp, gọn, dễ truy hồi.
Câu gốc:
{user_query}

Danh sách (đánh số):
"""
    try:
        resp = model_gen.generate_content(prompt)
        lines = (resp.text or "").strip().split("\n")
        return [re.sub(r"^\d+\.\s*", "", L).strip() for L in lines if L.strip()][:max_subqueries]
    except Exception:
        return [user_query]

def query_router(query: str) -> str:
    # Có thể mở rộng nếu bạn định multi-tool; hiện tại luôn dùng "document"
    terms = [
        "tư tưởng hồ chí minh", "độc lập dân tộc", "chủ nghĩa xã hội", "dân chủ",
        "đại đoàn kết", "đạo đức cách mạng", "nhà nước của dân do dân vì dân",
        "giáo dục", "văn hoá", "xây dựng đảng", "hồ chí minh"
    ]
    q = query.lower()
    return "document" if any(k in q for k in terms) else "document"

# ================== TRUY HỒI + RERANK ==================
def retrieve_documents(query: str, top_k: int = 10, rerank_k: int = 30, rerank_threshold: float = 0.4):
    qdrant = get_qdrant()
    try:
        # bge-m3: vẫn dùng prefix "query:"/"passage:" là tốt cho RAG
        qvec = embed_model.encode("query: " + query, normalize_embeddings=True)
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=qvec,
            limit=rerank_k,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        print(f"[rag.core] Qdrant search error: {e}")
        return [], 0.0

    if not hits:
        return [], 0.0

    pairs = [(query, h.payload.get("content", "")) for h in hits]
    scores = cross_encoder.predict(pairs)

    reranked = [(h, s) for h, s in zip(hits, scores) if s >= rerank_threshold] \
               or [(h, s) for h, s in zip(hits, scores) if s >= 0.2]
    reranked_sorted = sorted(reranked, key=lambda x: x[1], reverse=True)

    def clean_filename(fn):  # gọn nguồn
        import os
        return os.path.splitext(fn)[0]

    docs = [{
        "filename": clean_filename(h.payload.get("filename", "unknown")),
        "content": h.payload.get("content", ""),
        "score": s
    } for h, s in reranked_sorted[:top_k]]

    return docs, (max([d["score"] for d in docs]) if docs else 0.0)

# ================== ENTRYPOINT CHÍNH ==================
def generate_response(user_query: str, model_name: str = "gemini") -> str:
    # Viết lại + tách truy vấn
    q_rew = rewrite_query(user_query)
    subqueries = generate_subqueries(q_rew)

    # Truy hồi
    all_docs = []
    for sq in subqueries:
        docs, _ = retrieve_documents(sq, top_k=6, rerank_k=20)
        all_docs.extend(docs)

    # Gộp & lọc
    uniq = {d["content"]: d for d in all_docs if d.get("content")}
    unique_docs = list(uniq.values())

    # Nếu không có doc (Qdrant down/collection rỗng) → fallback LLM-only
    if not unique_docs:
        fallback = f"""
Bạn là gia sư về Tư tưởng Hồ Chí Minh. Trả lời rõ ràng, súc tích (~150–200 từ),
ưu tiên các trục: độc lập dân tộc gắn CNXH; dân chủ; đạo đức cách mạng; đại đoàn kết;
nhà nước của dân – do dân – vì dân; văn hoá; giáo dục; xây dựng Đảng.

Câu hỏi: {user_query}
"""
        if model_name.lower() == "gemini":
            return model_gen.generate_content(fallback).text.strip()
        elif model_name.lower() in ["llama3", "gemma"]:
            return generate_with_groq(fallback, model_name)
        return "Unsupported model."

    # Rerank lần 2 theo truy vấn gốc
    pairs = [(user_query, d["content"]) for d in unique_docs]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(
        [{"content": d["content"], "filename": d.get("filename", "unknown"), "score": s}
         for d, s in zip(unique_docs, scores)],
        key=lambda x: x["score"],
        reverse=True
    )
    final_docs = [d for d in reranked if d["score"] >= 0.15] or reranked[:3]

    docs_context = "\n\n".join(d["content"] for d in final_docs[:5])
    used_files = list(dict.fromkeys([d.get("filename") for d in final_docs[:3] if d.get("filename")]))

    # Prompt trả lời theo TTHCM
    answer_prompt = f"""
Bạn là một gia sư am hiểu Tư tưởng Hồ Chí Minh (tiếng Việt).

Yêu cầu trình bày:
- Nêu luận điểm cốt lõi liên quan đến câu hỏi (độc lập dân tộc – CNXH; dân chủ; đạo đức cách mạng;
  đại đoàn kết; nhà nước của dân – do dân – vì dân; giáo dục – văn hoá; xây dựng Đảng), có dẫn giải ngắn gọn.
- Nếu câu hỏi so sánh/ứng dụng thực tiễn, ưu tiên khung TTHCM, có thể gợi mở liên hệ Việt Nam.
- Giới hạn ~150–200 từ.

CÂU HỎI:
{user_query}

Tài liệu nền (chỉ để tham khảo, không cần trích nguyên văn):
{docs_context}

TRẢ LỜI:
"""
    if model_name.lower() == "gemini":
        ans = model_gen.generate_content(answer_prompt).text.strip()
    elif model_name.lower() in ["llama3", "gemma"]:
        ans = generate_with_groq(answer_prompt, model_name)
    else:
        ans = "Unsupported model."

    if used_files:
        ans += "\n\nSources: " + "; ".join(used_files[:5])
    return ans
