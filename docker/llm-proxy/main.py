"""LLM Proxy — OpenAI chat/completions → Responses API adapter.

Receives standard ``/v1/chat/completions`` requests and translates them
into the ``/v1/responses`` format expected by foxnio-style API providers.

Also proxies ``/v1/responses`` directly (pass-through).

Environment variables::

    LLM_UPSTREAM_URL    — e.g. https://www.foxnio.com/v1
    LLM_UPSTREAM_KEY    — API key for the upstream provider
    DEFAULT_MODEL       — fallback model name (default: gpt-5.2)
    PROXY_TIMEOUT       — request timeout in seconds (default: 120)
"""

import json
import logging
import os
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-proxy")

app = FastAPI(title="llm-proxy")

UPSTREAM_URL = os.environ.get("LLM_UPSTREAM_URL", "https://www.foxnio.com/v1").rstrip("/")
UPSTREAM_KEY = os.environ.get("LLM_UPSTREAM_KEY", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-5.2")
TIMEOUT = float(os.environ.get("PROXY_TIMEOUT", "120"))


def _upstream_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {UPSTREAM_KEY}",
        "Content-Type": "application/json",
    }


# ── /v1/chat/completions → /v1/responses translation ───────────────


def _chat_to_responses_body(chat_body: dict) -> dict:
    """Translate chat/completions request body to responses format."""
    messages = chat_body.get("messages", [])
    model = chat_body.get("model") or DEFAULT_MODEL

    # Build input: concatenate system + user messages.
    system_parts: list[str] = []
    input_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        else:
            input_parts.append(content)

    responses_body: dict = {
        "model": model,
        "input": "\n\n".join(input_parts) if input_parts else "",
    }

    if system_parts:
        responses_body["instructions"] = "\n\n".join(system_parts)

    # Map parameters.
    if chat_body.get("max_tokens"):
        responses_body["max_output_tokens"] = chat_body["max_tokens"]
    if chat_body.get("max_completion_tokens"):
        responses_body["max_output_tokens"] = chat_body["max_completion_tokens"]
    if chat_body.get("temperature") is not None:
        responses_body["temperature"] = chat_body["temperature"]
    if chat_body.get("top_p") is not None:
        responses_body["top_p"] = chat_body["top_p"]

    # Structured output: response_format → text.format.
    response_format = chat_body.get("response_format")
    if isinstance(response_format, dict):
        fmt_type = response_format.get("type", "")
        if fmt_type == "json_schema":
            responses_body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_format.get("json_schema", {}).get("name", "output"),
                    "strict": response_format.get("json_schema", {}).get("strict", True),
                    "schema": response_format.get("json_schema", {}).get("schema", {}),
                }
            }
        elif fmt_type == "json_object":
            responses_body["text"] = {"format": {"type": "json_object"}}
            # Upstream requires the word "json" in input for json_object mode.
            current_input = responses_body.get("input", "")
            if "json" not in current_input.lower():
                responses_body["input"] = current_input + "\n\nRespond in valid JSON."

    return responses_body


def _responses_to_chat_result(responses_data: dict, model: str) -> dict:
    """Translate responses API result back to chat/completions format."""
    # Extract text from output.
    text = ""
    output_list = responses_data.get("output", [])
    for item in output_list:
        if item.get("type") == "message":
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    text += content_item.get("text", "")

    usage_raw = responses_data.get("usage", {})

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": responses_data.get("model", model),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage_raw.get("input_tokens", 0),
            "completion_tokens": usage_raw.get("output_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Translate chat/completions → responses and back."""
    chat_body = await request.json()
    model = chat_body.get("model") or DEFAULT_MODEL
    stream = chat_body.get("stream", False)

    if stream:
        # Streaming: for now, call non-streaming and wrap as SSE.
        return await _chat_completions_fake_stream(chat_body, model)

    responses_body = _chat_to_responses_body(chat_body)

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{UPSTREAM_URL}/responses",
                json=responses_body,
                headers=_upstream_headers(),
            )
            resp.raise_for_status()
            responses_data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("upstream error: %s %s", exc.response.status_code, exc.response.text[:200])
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text[:500])
    except Exception as exc:
        logger.error("upstream request failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    chat_result = _responses_to_chat_result(responses_data, model)
    return JSONResponse(content=chat_result)


async def _chat_completions_fake_stream(chat_body: dict, model: str):
    """Non-streaming call wrapped as SSE for clients that request stream=true."""
    responses_body = _chat_to_responses_body(chat_body)

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{UPSTREAM_URL}/responses",
                json=responses_body,
                headers=_upstream_headers(),
            )
            resp.raise_for_status()
            responses_data = resp.json()
    except Exception as exc:
        logger.error("upstream stream failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    chat_result = _responses_to_chat_result(responses_data, model)
    text = chat_result["choices"][0]["message"]["content"]

    async def _generate():
        # Send content as a single chunk.
        delta = {
            "id": chat_result["id"],
            "object": "chat.completion.chunk",
            "created": chat_result["created"],
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(delta)}\n\n"
        # Send stop.
        stop = {
            "id": chat_result["id"],
            "object": "chat.completion.chunk",
            "created": chat_result["created"],
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(stop)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── /v1/responses pass-through ──────────────────────────────────────


@app.post("/v1/responses")
async def responses_passthrough(request: Request):
    """Pass-through for clients that use the responses API directly."""
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{UPSTREAM_URL}/responses",
                json=body,
                headers=_upstream_headers(),
            )
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text[:500])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── /v1/models pass-through ─────────────────────────────────────────


@app.get("/v1/models")
async def models_passthrough():
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{UPSTREAM_URL}/models",
                headers=_upstream_headers(),
            )
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/health")
def health():
    return {"ok": True, "upstream": UPSTREAM_URL}
