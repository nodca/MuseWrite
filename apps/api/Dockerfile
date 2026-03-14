FROM python:3.12-slim AS base

ARG PRELOAD_RERANKER_ONNX=false

# Install uv from official image (single static binary, ~15MB).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    CONTEXT_COMPRESSION_MODE=rerank \
    CONTEXT_COMPRESSION_RERANKER_ENABLED=true \
    CONTEXT_COMPRESSION_RERANKER_MODEL=BAAI/bge-reranker-v2-minicpm-layerwise \
    CONTEXT_COMPRESSION_RERANKER_RUNTIME=onnx \
    CONTEXT_COMPRESSION_RERANKER_TRUST_REMOTE_CODE=true \
    CONTEXT_COMPRESSION_RERANKER_ONNX_PATH=/app/models/bge-reranker-v2-minicpm-layerwise-int8-onnx/model.onnx \
    CONTEXT_COMPRESSION_RERANKER_ONNX_PROVIDER=CPUExecutionProvider

WORKDIR /app

COPY requirements-docker.txt ./requirements.txt
COPY requirements-export.txt ./requirements-export.txt
RUN uv pip install --no-cache -r requirements.txt

COPY app ./app
COPY scripts ./scripts
RUN if [ "$PRELOAD_RERANKER_ONNX" = "true" ]; then \
      uv pip install --no-cache -r requirements-export.txt && \
      python scripts/export_reranker_onnx.py \
        --model-id "$CONTEXT_COMPRESSION_RERANKER_MODEL" \
        --output "$CONTEXT_COMPRESSION_RERANKER_ONNX_PATH" \
        --max-length 384 \
        --opset 17 \
        --cutoff-layer 28 \
        --trust-remote-code \
        --onnx-provider "$CONTEXT_COMPRESSION_RERANKER_ONNX_PROVIDER"; \
    fi \
    && rm -rf /root/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
