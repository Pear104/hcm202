import os
import re
import json
from typing import List
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from qdrant_client import QdrantClient
import google.generativeai as genai
from .generator import generate_with_groq
from dotenv import load_dotenv

# ----------- Load models & config -----------

# Embedding model
load_dotenv()

# embed_model = SentenceTransformer('BAAI/bge-m3')
embed_model = SentenceTransformer('intfloat/multilingual-e5-base')

# Cross-encoder for reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # Reranking model

# Gemini generative model
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_gen = genai.GenerativeModel("gemini-2.0-flash")

# Qdrant client
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
collection_name = "MLN131_2"

_qdrant = None
def get_qdrant():
    global _qdrant
    if _qdrant is None:
        url = (os.getenv("QDRANT_URL") or "").rstrip("/")
        api_key = os.getenv("QDRANT_API_KEY") or None
        # in log nhẹ để debug khi cần:
        print(f"[rag.core] QDRANT_URL={url or '<empty>'}")
        _qdrant = QdrantClient(url=url, api_key=api_key, timeout=30.0, check_compatibility=False)
    return _qdrant
# ----------- Utilities -----------

def rewrite_query(original_query: str) -> str:
    prompt = f"""
Bạn là trợ lý tối ưu truy vấn cho chatbot về Chủ nghĩa Xã hội Khoa học.

Yêu cầu:
- Viết lại câu hỏi ngắn gọn, học thuật, rõ trọng tâm CNXHKH
- Ưu tiên các khái niệm: lực lượng/quan hệ sản xuất, sở hữu (công hữu/tư hữu),
  giai cấp & nhà nước, chuyên chính vô sản, dân chủ XHCN, thời kỳ quá độ,
  tiêu vong của nhà nước, đối chiếu bối cảnh Việt Nam (nếu liên quan)
- Thêm từ đồng nghĩa/thuật ngữ tương đương có ích cho truy hồi.

Gốc:
{original_query}

Bản viết lại (một dòng, không giải thích):
"""
    rewritten = model_gen.generate_content(prompt)
    return rewritten.text.strip()

def query_router(query: str) -> str:
    socialism_terms = [
        "chủ nghĩa xã hội", "chủ nghĩa cộng sản", "mác", "engels", "ăngghen", "lênin",
        "công hữu", "tư hữu", "quan hệ sản xuất", "lực lượng sản xuất",
        "dân chủ xã hội chủ nghĩa", "chuyên chính vô sản",
        "thời kỳ quá độ", "nhà nước và cách mạng", "việt nam", "cnxh khoa học"
    ]
    q = query.lower()
    if any(k in q for k in socialism_terms):
        return "document"
    return "document"

def generate_subqueries(user_query: str, max_subqueries: int = 4) -> List[str]:
    prompt = f"""
Bạn là trợ lý tách truy vấn cho chatbot CNXHKH.

Hãy tách câu hỏi dưới đây thành tối đa {max_subqueries} tiểu câu, mỗi câu CHỈ 1 ý:
(ví dụ: sở hữu/công hữu, lực lượng vs quan hệ sản xuất, giai cấp & nhà nước,
chuyên chính vô sản, dân chủ XHCN, thời kỳ quá độ lên CNCS, tiêu vong nhà nước,
vận dụng tại Việt Nam).

Không trùng lặp, gọn, dễ truy hồi.
Câu gốc:
{user_query}

Danh sách (đánh số):
"""
    try:
        resp = model_gen.generate_content(prompt)
        lines = resp.text.strip().split("\n")
        return [re.sub(r"^\d+\.\s*", "", L).strip() for L in lines if L.strip()][:max_subqueries]
    except Exception:
        return [user_query]

def retrieve_documents(query: str, top_k: int = 10, rerank_k: int = 30, rerank_threshold: float = 0.4) -> tuple[List[dict], float]:
    qdrant = get_qdrant()
    query_vec = embed_model.encode("query: " + query, normalize_embeddings=True)
    initial_results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=rerank_k,
        with_payload=True,
        with_vectors=False,
    )
    if not initial_results:
        return [], 0.0

    pairs = [(query, hit.payload["content"]) for hit in initial_results]
    scores = cross_encoder.predict(pairs)

    reranked = [
        (hit, s) for hit, s in zip(initial_results, scores) if s >= rerank_threshold
    ] or [
        (hit, s) for hit, s in zip(initial_results, scores) if s >= 0.2
    ]

    reranked_sorted = sorted(reranked, key=lambda x: x[1], reverse=True)

    def clean_filename(filename):
        import os
        return os.path.splitext(filename)[0]

    documents = [
        {
            "filename": clean_filename(hit.payload.get("filename", "unknown")),
            "content": hit.payload["content"],
            "score": s
        }
        for hit, s in reranked_sorted[:top_k]
    ]
    max_score = max([d["score"] for d in documents], default=0.0)
    return documents, max_score


# ----------- Main RAG entrypoint -----------
def generate_response(user_query: str, model_name: str = "gemini") -> str:
    # 2) Subqueries
    subqueries = generate_subqueries(user_query)
    all_docs = []
    for sq in subqueries:
        docs, _ = retrieve_documents(sq, top_k=6, rerank_k=15)
        all_docs.extend(docs)

    # 3) Dedup theo content
    uniq_by_content = {d["content"]: d for d in all_docs}
    unique_docs = list(uniq_by_content.values())
    if not unique_docs:
        return "Xin lỗi, mình chưa tìm thấy thông tin phù hợp trong kho tài liệu CNXHKH."

    # 4) Rerank lại theo truy vấn gốc, NHỚ giữ filename
    pairs = [(user_query, d["content"]) for d in unique_docs]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(
        [{"content": d["content"], "filename": d.get("filename","unknown"), "score": s}
         for d, s in zip(unique_docs, scores)],
        key=lambda x: x["score"],
        reverse=True
    )
    final_docs = [d for d in reranked if d["score"] >= 0.15]
    if not final_docs:
        return "Xin lỗi, mình chưa có câu trả lời đủ tin cậy từ tài liệu."

    docs_context = "\n\n".join(d["content"] for d in final_docs[:5])
    used_files = []
    for d in final_docs[:3]:
        if d.get("filename"): used_files.append(d["filename"])
    used_files = list(dict.fromkeys(used_files))  # dedup

    answer_prompt = f"""
Bạn là một gia sư am hiểu Chủ nghĩa Xã hội Khoa học (tiếng Việt).

Hãy trả lời rõ ràng, khái quát hoá khái niệm cốt lõi (VD: sở hữu XHCN vs tư bản,
quan hệ–lực lượng sản xuất, nhà nước, chuyên chính vô sản, dân chủ XHCN, thời kỳ quá độ).
Nếu câu hỏi mang tính so sánh/normative, ưu tiên khung lý luận kinh điển trước
(Mác–Ăngghen–Lênin), sau đó mới gợi mở vận dụng (Việt Nam) nếu phù hợp.
Giới hạn ~150–200 từ khi có thể. Trả lời bằng tiếng Việt.

CÂU HỎI:
{user_query}

Tư liệu nền (chỉ tham khảo, không cần nhắc tới):
{docs_context}

TRẢ LỜI:
"""

    if model_name.lower() == "gemini":
        resp = model_gen.generate_content(answer_prompt)
        ans = resp.text.strip()
    elif model_name.lower() in ["gpt", "gemma"]:
        ans = generate_with_groq(answer_prompt, model_name)
    else:
        ans = f"Unsupported model: {model_name}"

    if used_files:
        ans += "\n\nSources: " + "; ".join(used_files[:5])
    return ans