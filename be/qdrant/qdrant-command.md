From this command, how do I set docker volumes to qdrant_data?
```bash
# `--gpus=all` flag says to Docker that we want to use GPUs.
# `-e QDRANT__GPU__INDEXING=1` flag says to Qdrant that we want to use GPUs for indexing.
sudo docker run -d \
  --gpus=all \
  -p 6333:6333 \
  -p 6334:6334 \
  -e QDRANT__GPU__INDEXING=1 \
  -v qdrant_qdrant_data:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant:gpu-nvidia-latest

sudo docker run -d \
  -p 7979:8000 \
  --name hcm202 \
  jangkuz/hcm202

```

`uvicorn app.main:app --host 0.0.0.0 --port 7979 --reload` 