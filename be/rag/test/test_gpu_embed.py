# tools/test_gpu_embed.py
import os
import time
import platform
import subprocess
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ========= Cấu hình =========
MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")  # đổi nếu bạn dùng model khác
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
N_RUNS = int(os.getenv("N_RUNS", "3"))
TEXTS: List[str] = [
    "Tư tưởng Hồ Chí Minh về độc lập dân tộc gắn liền với chủ nghĩa xã hội.",
    "Đại đoàn kết dân tộc là chiến lược cách mạng, là nguồn sức mạnh to lớn.",
    "Nhà nước của dân, do dân, vì dân; dân chủ là bản chất của chế độ ta.",
    "Đạo đức cách mạng: cần, kiệm, liêm, chính, chí công vô tư.",
    "Giáo dục – văn hóa nhằm xây dựng con người mới xã hội chủ nghĩa."
] * 200  # nhân lên cho bài test có ý nghĩa

def print_header():
    print("=" * 70)
    print(f"Python      : {platform.python_version()}")
    print(f"Torch       : {torch.__version__}")
    print(f"CUDA avail? : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device : {torch.cuda.get_device_name(0)}")
        print(f"CUDA count  : {torch.cuda.device_count()}")
        try:
            # In nhẹ nvidia-smi nếu có (Windows có thể không có trong PATH)
            out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, text=True)
            print("nvidia-smi  :\n" + out.strip())
        except Exception:
            pass
    print(f"MODEL       : {MODEL_NAME}")
    print(f"BATCH_SIZE  : {BATCH_SIZE}, N_RUNS={N_RUNS}")
    print("=" * 70)

def bench(model: SentenceTransformer, device_desc: str) -> float:
    # warmup
    _ = model.encode(TEXTS[:BATCH_SIZE], batch_size=BATCH_SIZE, normalize_embeddings=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.time()
    for _ in range(N_RUNS):
        _ = model.encode(TEXTS, batch_size=BATCH_SIZE, normalize_embeddings=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()

    elapsed = t1 - t0
    total = len(TEXTS) * N_RUNS
    qps = total / elapsed
    print(f"[{device_desc}] {total} texts in {elapsed:.2f}s  →  {qps:.1f} texts/sec")
    return qps

def main():
    print_header()

    # ===== Test CUDA (nếu có) =====
    if torch.cuda.is_available():
        model_cuda = SentenceTransformer(MODEL_NAME, device="cuda")
        # kiểm tra nhanh layer params đang nằm trên GPU
        param_device = next(model_cuda.auto_model.parameters()).device
        print(f"[CUDA] first param device: {param_device}")
        # encode trả numpy, nhưng nội suy nằm trên GPU nếu device=cuda
        qps_cuda = bench(model_cuda, "CUDA")
    else:
        qps_cuda = 0.0
        print("[CUDA] Không khả dụng → bỏ qua test CUDA.")

    # ===== Test CPU =====
    model_cpu = SentenceTransformer(MODEL_NAME, device="cpu")
    param_device_cpu = next(model_cpu.auto_model.parameters()).device
    print(f"[CPU] first param device: {param_device_cpu}")
    qps_cpu = bench(model_cpu, "CPU")

    # ===== Kết luận =====
    print("=" * 70)
    if qps_cuda > 0:
        speedup = (qps_cuda / qps_cpu) if qps_cpu > 0 else float("inf")
        print(f"✅ GPU đang hoạt động. Tốc độ ~ x{speedup:.2f} so với CPU.")
    else:
        print("⚠️  GPU chưa hoạt động. Hãy kiểm tra cài đặt PyTorch CUDA / driver / biến môi trường.")
    print("=" * 70)

if __name__ == "__main__":
    main()
