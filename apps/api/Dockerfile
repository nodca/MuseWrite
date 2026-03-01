FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONTEXT_COMPRESSION_MODE=rerank \
    CONTEXT_COMPRESSION_RERANKER_ENABLED=true \
    CONTEXT_COMPRESSION_RERANKER_MODEL=BAAI/bge-reranker-v2-minicpm-layerwise \
    CONTEXT_COMPRESSION_RERANKER_RUNTIME=onnx \
    CONTEXT_COMPRESSION_RERANKER_TRUST_REMOTE_CODE=true \
    CONTEXT_COMPRESSION_RERANKER_ONNX_PATH=/app/models/bge-reranker-v2-minicpm-layerwise-int8-onnx/model.onnx \
    CONTEXT_COMPRESSION_RERANKER_ONNX_PROVIDER=CPUExecutionProvider

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
