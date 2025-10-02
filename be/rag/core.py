# rag/core.py
import os, re, json
import logging
from typing import List
from dotenv import load_dotenv

# nạp .env TRƯỚC khi đọc env
load_dotenv()

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
import google.generativeai as genai

from .generator import generate_with_groq, generate_with_gemini

# ================== CAU HINH LOGGING =================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "HCM_TuTuong3")  # bạn có thể đổi tuỳ ý

_qdrant = None


def get_qdrant():
    """Khởi tạo Qdrant client khi cần (tránh lỗi thứ tự import)."""
    global _qdrant
    if _qdrant is None:
        url = (os.getenv("QDRANT_URL") or "").rstrip("/")
        api_key = os.getenv("QDRANT_API_KEY") or None
        print(f"[rag.core] QDRANT_URL={url or '<empty>'}  COLLECTION={COLLECTION_NAME}")
        _qdrant = QdrantClient(
            url=url, api_key=api_key, timeout=30.0, check_compatibility=False
        )
    return _qdrant


# ================== TIỆN ÍCH PROMPT/ROUTER ==================
def rewrite_query(original_query: str) -> str:
    prompt = f"""
Bạn là trợ lý tối ưu truy vấn cho chatbot về Tư tưởng Hồ Chí Minh (TTHCM).

Yêu cầu:
- Viết lại câu hỏi ngắn gọn, học thuật, đúng trọng tâm TTHCM
- Ưu tiên các chủ đề: độc lập dân tộc gắn với CNXH; dân tộc-giai cấp; đại đoàn kết
  dân tộc; nhà nước của dân - do dân - vì dân; dân chủ; đạo đức cách mạng;
  giáo dục - con người; văn hoá; đối ngoại; xây dựng Đảng.
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


def generate_subqueries(user_query: str, max_subqueries: int = 5) -> List[str]:
    prompt = f"""
Bạn là trợ lý tách truy vấn cho chatbot Tư tưởng Hồ Chí Minh.

Hãy tách câu hỏi dưới đây thành tối đa {max_subqueries} tiểu câu, mỗi câu chỉ 1 ý:
(vd: độc lập dân tộc gắn CNXH; dân chủ; đạo đức cách mạng; đại đoàn kết;
nhà nước của dân - do dân - vì dân; văn hoá; giáo dục con người; xây dựng Đảng).

Không trùng lặp, gọn, dễ truy hồi.
Câu gốc:
{user_query}

Danh sách (đánh số):
"""
    try:
        resp = model_gen.generate_content(prompt)
        lines = (resp.text or "").strip().split("\n")
        return [re.sub(r"^\d+\.\s*", "", L).strip() for L in lines if L.strip()][
            :max_subqueries
        ]
    except Exception:
        return [user_query]


def query_router(query: str) -> str:
    # Có thể mở rộng nếu bạn định multi-tool; hiện tại luôn dùng "document"
    terms = [
        "tư tưởng hồ chí minh",
        "độc lập dân tộc",
        "chủ nghĩa xã hội",
        "dân chủ",
        "đại đoàn kết",
        "đạo đức cách mạng",
        "nhà nước của dân do dân vì dân",
        "giáo dục",
        "văn hoá",
        "xây dựng đảng",
        "hồ chí minh",
    ]
    q = query.lower()
    return "document" if any(k in q for k in terms) else "document"


# ================== TRUY HỒI + RERANK ==================
def retrieve_documents(
    query: str, top_k: int = 20, rerank_k: int = 30, rerank_threshold: float = 0.3
):
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
        print(f"[rag.core] Finish qdrant query")
    except Exception as e:
        print(f"[rag.core] Qdrant search error: {e}")
        return [], 0.0

    if not hits:
        return [], 0.0

    pairs = [(query, h.payload.get("content", "")) for h in hits]
    scores = cross_encoder.predict(pairs)

    reranked = [(h, s) for h, s in zip(hits, scores) if s >= rerank_threshold] or [
        (h, s) for h, s in zip(hits, scores) if s >= 0.2
    ]
    reranked_sorted = sorted(reranked, key=lambda x: x[1], reverse=True)

    def clean_filename(fn):  # gọn nguồn
        import os

        return os.path.splitext(fn)[0]

    docs = [
        {
            "filename": clean_filename(h.payload.get("filename", "unknown")),
            "content": h.payload.get("content", ""),
            "score": s,
        }
        for h, s in reranked_sorted[:top_k]
    ]

    return docs, (max([d["score"] for d in docs]) if docs else 0.0)

# ================== Phân loại câu hỏi ==================
def _classify_intent(q: str) -> str:
    ql = q.lower()
    academic_kw = [
        "khái niệm", "luận điểm", "quan điểm", "nguyên tắc", "nội dung", "chuyên đề",
        "bài học", "vận dụng", "môn học", "tư tưởng hồ chí minh", "đề cương", "ôn tập",
        "độc lập gắn chủ nghĩa xã hội", "dân chủ", "đại đoàn kết", "đạo đức cách mạng",
        "nhà nước của dân do dân vì dân", "xây dựng đảng", "giáo dục", "văn hóa"
    ]
    bio_kw = [
        "tiểu sử", "quê", "năm sinh", "gia đình", "tuổi thơ", "câu chuyện", "giai thoại",
        "tên gọi", "hành trình", "bút danh", "hoạt động", "búp sen xanh", "sơn tùng",
        "trường", "làm việc", "bị bắt", "nhà tù", "bài báo", "bài viết", "người thân"
    ]
    if any(k in ql for k in academic_kw) and not any(k in ql for k in bio_kw):
        return "academic"
    if any(k in ql for k in bio_kw):
        return "bio"
    # mặc định: nếu hỏi khái quát/tổng quan → ưu tiên học thuật
    return "academic"

def _mk_sources_note(used_files):
    if not used_files: 
        return ""
    # Gợi ý nguồn tự nhiên (nếu tên file gợi rõ, model có thể nhắc nhẹ)
    # VD: Ho_Chi_Minh_Toan_Tap_Tap6.pdf, GiaoTrinh_TTHCM.pdf, Bup_Sen_Xanh.pdf
    hints = []
    for f in used_files:
        fl = f.lower()
        if "toan_tap" in fl or "toàn tập" in fl or "tap" in fl:
            hints.append("Hồ Chí Minh Toàn tập")
        elif "giao trinh" in fl or "giaotrinh" in fl or "tthcm" in fl:
            hints.append("Giáo trình Tư tưởng Hồ Chí Minh")
        elif "bup sen xanh" in fl or "búp sen xanh" in fl or "son tung" in fl or "sơn tùng" in fl:
            hints.append("Búp sen xanh (Sơn Tùng)")
    # khử trùng lặp
    hints = list(dict.fromkeys(hints))
    return "; ".join(hints) if hints else ""




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
Bạn là gia sư về Hồ Chí Minh và Tư tưởng Hồ Chí Minh. Trả lời rõ ràng, súc tích (~170-250 từ),
ưu tiên các trục: độc lập dân tộc gắn CNXH; dân chủ; đạo đức cách mạng; đại đoàn kết;
nhà nước của dân - do dân - vì dân; văn hoá; giáo dục; xây dựng Đảng, nội dung đề cập trong tài liệu truy vấn được.

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
        [
            {
                "content": d["content"],
                "filename": d.get("filename", "unknown"),
                "score": s,
            }
            for d, s in zip(unique_docs, scores)
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    final_docs = [d for d in reranked if d["score"] >= 0.15] or reranked[:3]

    docs_context = "\n\n".join(d["content"] for d in final_docs[:5])
    used_files = list(
        dict.fromkeys([d.get("filename") for d in final_docs[:3] if d.get("filename")])
    )
    
    # === Answer ===   
    intent = _classify_intent(user_query)
    sources_hint = _mk_sources_note(used_files)

    if intent == "academic":
        # PHONG CÁCH MÔN HỌC TTHCM
        answer_prompt = f"""
    Bạn là gia sư môn **Tư tưởng Hồ Chí Minh** (tiếng Việt). Trả lời NGẮN GỌN nhưng MẠCH LẠC (≈170–250 từ),
    dựa **duy nhất** vào trích đoạn trong phần "Tài liệu nền" (RAG). Không bịa nguồn; nếu tài liệu không nêu, nói "tài liệu chưa nêu rõ".

    Yêu cầu trình bày:
    - Nêu **luận điểm cốt lõi** liên quan câu hỏi (độc lập dân tộc gắn CNXH; dân chủ; đạo đức cách mạng;
    đại đoàn kết; **nhà nước của dân–do dân–vì dân**; giáo dục–văn hoá; xây dựng Đảng).
    - Giải thích **ngắn gọn, có hệ thống** (khái niệm → ý nghĩa → liên hệ thực tiễn nếu tài liệu có).
    - Nếu RAG có đoạn trùng/không chắc, hãy **làm rõ giới hạn** (“tài liệu chỉ cho biết…”, “chưa thấy trích dẫn…”).

    CÂU HỎI:
    {user_query}

    TÀI LIỆU NỀN (trích RAG, dùng làm căn cứ; **không trích nguyên văn dài**):
    {docs_context}

    Nếu thích hợp, bạn có thể nhắc **nguồn tổng quát** như: {sources_hint if sources_hint else "—"}.

    TRẢ LỜI (giữ giọng điệu học thuật, dễ hiểu):
    """
    else:
        # PHONG CÁCH TIỂU SỬ/CÂU CHUYỆN TỰ NHIÊN, GẦN GŨI
        answer_prompt = f"""
    Bạn là người kể chuyện am hiểu **Chủ tịch Hồ Chí Minh** (tiếng Việt). Trả lời **tự nhiên, gần gũi, tôn trọng**,
    nhưng **dựa duy nhất** vào "Tài liệu nền" (RAG). Không bịa chi tiết; nếu không chắc, nói rõ “tài liệu chưa nêu”.

    Yêu cầu:
    - Tập trung **thông tin tiểu sử, bối cảnh, câu chuyện đời** (tuổi thơ, hành trình, bút danh, giai thoại…),
    diễn đạt mạch lạc, **tránh suy đoán**.
    - Có thể chèn **chi tiết sinh động** nếu RAG có (địa danh, mốc thời gian, nhân vật liên quan).
    - Cuối câu trả lời, nhắc **nguồn tổng quát** nếu phù hợp (vd. “Búp sen xanh”, “Hồ Chí Minh Toàn tập”…).

    CÂU HỎI:
    {user_query}

    TÀI LIỆU NỀN (trích RAG, dùng làm căn cứ; không trích nguyên văn dài):
    {docs_context}

    Gợi ý nguồn tổng quát (nếu đúng với trích đoạn): {sources_hint if sources_hint else "—"}.

    TRẢ LỜI (kể chuyện tự nhiên, 170–250 từ):
    """

    # Gọi LLM như cũ
    if model_name.lower() == "gemini":
        ans = model_gen.generate_content(answer_prompt).text.strip()
    elif model_name.lower() in ["gpt", "gemma", "llama3"]:
        ans = generate_with_groq(answer_prompt, model_name)
    else:
        ans = "Unsupported model."

    # + phần Sources như bạn đang làm
    if used_files:
        ans += "\n\nSources: " + "; ".join(used_files[:5])

    return ans