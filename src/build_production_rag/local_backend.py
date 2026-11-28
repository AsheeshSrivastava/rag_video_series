"""The local GPU backend. torch and transformers are imported only when it is built."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .embeddings import EmbeddingUnavailable, ModelSpec

_MEAN_POOLING = {"jinaai/jina-embeddings-v2-base-en"}  # every other local model uses CLS
_BATCH_SIZE = 8
# Top-level files only: a pattern without a leading "*" never matches onnx/ or coreml/.
_SNAPSHOT_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "model.safetensors",
    "pytorch_model.bin",
]


class LocalBackend:
    def __init__(self, spec: ModelSpec) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._mean = spec.id in _MEAN_POOLING
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.revision = spec.revision
        # M5: load from the pinned snapshot folder, not the repo ID. Given a repo ID,
        # transformers and jina's remote code make their own Hub calls with no revision.
        path = snapshot_download(spec.id, revision=spec.revision, allow_patterns=_SNAPSHOT_FILES)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        kwargs = {}
        if spec.remote_code_revision is not None:
            kwargs = {
                "trust_remote_code": True,
                "code_revision": spec.remote_code_revision,
            }
        self._model = (
            AutoModel.from_pretrained(path, **kwargs)
            .to(self._device)
            .eval()
        )

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer(text, truncation=False)["input_ids"])

    def embed_raw(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        rows = []
        try:
            with torch.inference_mode():
                for start in range(0, len(texts), _BATCH_SIZE):
                    enc = self._tokenizer(
                        list(texts[start : start + _BATCH_SIZE]),
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    ).to(self._device)
                    hidden = self._model(**enc)[0]
                    if self._mean:
                        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                        pooled = (hidden * mask).sum(1) / mask.sum(1)
                    else:
                        pooled = hidden[:, 0]
                    rows.append(pooled.float().cpu().numpy())
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise EmbeddingUnavailable("GPU ran out of memory") from None
        return np.concatenate(rows) if rows else np.zeros((0, 0), dtype=np.float32)
