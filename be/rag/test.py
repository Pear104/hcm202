import os, requests
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
    HAVE_CE = True
except Exception:
    HAVE_CE = False

load_dotenv()
QDRANT_URL = (os.getenv("QDRANT_URL")).rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = os.getenv("COLLECTION_NAME") or "MLN131_2"   # đổi nếu bạn dùng tên khác
TOP_K = int(os.getenv("TOP_K") or 5)
USE_RERANK = (os.getenv("USE_RERANK") or "false").lower() == "true"

print(f"[cfg] QDRANT_URL={QDRANT_URL}  COLLECTION={COLLECTION}  TOP_K={TOP_K}  RERANK={USE_RERANK}")
url = (os.getenv("QDRANT_URL") or "").rstrip("/")
key = os.getenv("QDRANT_API_KEY") or ""
headers = {"api-key": key} if key else {}
qdrant_client = QdrantClient(
    url=url, 
    api_key=key,
)
print("QDRANT_URL =", url)
print("API key present?:", bool(key), "len:", len(key))

try:
    r = requests.get(f"{url}/collections", headers=headers, timeout=10)
    print("HTTP", r.status_code)
    print("Body head:", r.text[:200])
    print(qdrant_client.get_collections())
except Exception as e:
    print("ERR:", e)