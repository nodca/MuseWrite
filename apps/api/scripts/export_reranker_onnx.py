from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_LAYERWISE_PROMPT = (
    "Given a query A and a passage B, determine whether the passage contains "
    "an answer to the query by providing a prediction of either 'Yes' or 'No'."
)
_LAYERWISE_SEPARATOR = "\n"


class _LayerWiseRerankerWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, cutoff_layer: int) -> None:
        super().__init__()
        self.model = model
        self.cutoff_layer = max(int(cutoff_layer), 1)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            cutoff_layers=[self.cutoff_layer],
        )
        logits = outputs.logits
        if isinstance(logits, (list, tuple)):
            if not logits:
                raise RuntimeError("unexpected empty reranker logits tuple")
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("unexpected reranker logits type")
        if int(logits.dim()) == 2:
            return logits[:, -1]
        if int(logits.dim()) == 3:
            return logits[:, -1, 0]
        if int(logits.dim()) == 1:
            return logits
        raise RuntimeError(f"unsupported reranker logits rank: {int(logits.dim())}")


def _resolve_output_path(output: str) -> Path:
    raw = Path(str(output).strip())
    if raw.suffix.lower() == ".onnx":
        return raw
    return raw / "model.onnx"


def _build_layerwise_inputs(
    tokenizer: AutoTokenizer,
    pairs: list[tuple[str, str]],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    prompt_tokens = tokenizer(
        _LAYERWISE_PROMPT,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]
    sep_tokens = tokenizer(
        _LAYERWISE_SEPARATOR,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]
    items: list[dict[str, list[int]]] = []
    for query, passage in pairs:
        query_item = tokenizer(
            f"A: {query}",
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length * 3 // 4,
            truncation=True,
        )
        passage_item = tokenizer(
            f"B: {passage}",
            return_tensors=None,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        encoded = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + list(query_item["input_ids"]),
            list(sep_tokens) + list(passage_item["input_ids"]),
            truncation="only_second",
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        input_ids = list(encoded["input_ids"]) + list(sep_tokens) + list(prompt_tokens)
        items.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
        )
    padded = tokenizer.pad(
        items,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    return {
        "input_ids": padded["input_ids"].to(dtype=torch.long),
        "attention_mask": padded["attention_mask"].to(dtype=torch.long),
    }


def _onnx_smoke_test(
    *,
    onnx_path: Path,
    tokenizer: AutoTokenizer,
    max_length: int,
    cutoff_layer: int,
    provider: str,
) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=[provider])
    encoded = _build_layerwise_inputs(
        tokenizer,
        pairs=[("查询词", "候选句子")],
        max_length=max_length,
    )
    ort_inputs: dict[str, object] = {}
    for item in session.get_inputs():
        value = encoded.get(item.name)
        if value is None:
            raise RuntimeError(f"missing onnx input: {item.name}")
        ort_inputs[item.name] = value.detach().cpu().numpy().astype("int64")
    outputs = session.run(None, ort_inputs)
    if not outputs:
        raise RuntimeError("onnx smoke test returned empty outputs")
    scores = outputs[0]
    if getattr(scores, "shape", None) is None:
        raise RuntimeError("onnx smoke test returned invalid score tensor")
    shape = tuple(int(x) for x in scores.shape)
    if len(shape) != 1 or shape[0] != 1:
        raise RuntimeError(
            f"onnx smoke test returned unexpected score shape: {shape} (cutoff_layer={int(cutoff_layer)})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export reranker model to ONNX.")
    parser.add_argument(
        "--model-id",
        default="BAAI/bge-reranker-v2-minicpm-layerwise",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--output",
        default="./models/bge-reranker-v2-minicpm-layerwise-int8-onnx/model.onnx",
        help="ONNX output path (file) or output directory.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=384,
        help="Tokenizer max_length used for export sample.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--onnx-provider",
        default="CPUExecutionProvider",
        help="ONNXRuntime provider used for smoke test.",
    )
    parser.add_argument(
        "--cutoff-layer",
        type=int,
        default=28,
        help="Layer index used by the layerwise reranker.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading Hugging Face model/tokenizer.",
    )
    args = parser.parse_args()

    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] model={args.model_id}")
    print(f"[export] output={output_path}")
    print(f"[export] max_length={args.max_length} opset={args.opset} cutoff_layer={int(args.cutoff_layer)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=bool(args.trust_remote_code),
    )
    tokenizer.save_pretrained(output_path.parent)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.eval()

    sample_inputs = _build_layerwise_inputs(
        tokenizer,
        pairs=[("你是重排器", "请保留与写作任务最相关的设定与关系。")],
        max_length=max(int(args.max_length), 64),
    )
    input_names = list(sample_inputs.keys())
    wrapper = _LayerWiseRerankerWrapper(model, cutoff_layer=max(int(args.cutoff_layer), 1))

    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["score"] = {0: "batch"}

    torch.onnx.export(
        wrapper,
        args=tuple(sample_inputs[name] for name in input_names),
        f=str(output_path),
        input_names=input_names,
        output_names=["score"],
        dynamic_axes=dynamic_axes,
        opset_version=max(int(args.opset), 11),
        do_constant_folding=True,
        external_data=True,
        dynamo=False,
    )
    print("[export] onnx export done")

    _onnx_smoke_test(
        onnx_path=output_path,
        tokenizer=tokenizer,
        max_length=max(int(args.max_length), 64),
        cutoff_layer=max(int(args.cutoff_layer), 1),
        provider=str(args.onnx_provider or "CPUExecutionProvider"),
    )
    print("[export] onnx smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
