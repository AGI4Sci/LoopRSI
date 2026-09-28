"""Pluggable semantic embedding backends for memory retrieval (P3).

Two backends implement the same duck interface (``embed(text) -> vector``):

- ``tfidf``: pure-stdlib character n-gram TF-IDF sparse vectors.  Zero
  dependencies, deterministic, always available (PoC default backend).
- ``dense``: an ONNX sentence-embedding model (e.g. ``bge-small-zh-v1.5``)
  served through ``onnxruntime`` with a stdlib WordPiece tokenizer read from
  ``tokenizer.json``.  Available only when onnxruntime is installed and the
  model files are present; otherwise the caller degrades to no-memory.

Both backends return unit-L2-normalised vectors so that cosine similarity is
simply the dot product.  Pure stdlib + onnxruntime; Python 3.10+.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

Vector = Union[Dict[str, float], List[float]]


# --------------------------------------------------------------------------
# cosine + helpers
# --------------------------------------------------------------------------

def _l2(v: Sequence[float]) -> float:
    return math.sqrt(float(sum(x * x for x in v)))


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity between two embedding vectors.

    Dense (list) and sparse (dict) vectors are both supported.  ``a`` and
    ``b`` must be of the same representation kind (as produced by one backend).
    """
    if isinstance(a, dict):
        a_items = list(a.items())
        b_items = list(b.items())
        small, large = (a_items, b_items) if len(a_items) <= len(b_items) else (b_items, a_items)
        large_map = dict(large)
        dot = 0.0
        for k, v in small:
            w = large_map.get(k)
            if w is not None:
                dot += v * w
        denom = (_l2(list(a.values())) * _l2(list(b.values()))) or 1e-12
        return dot / denom
    denom = (_l2(a) * _l2(b)) or 1e-12
    return math.fsum(x * y for x, y in zip(a, b)) / denom


def cosine_available() -> bool:
    return True


# --------------------------------------------------------------------------
# TF-IDF (pure stdlib)
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+")


def _norm_text(text: str) -> str:
    s = (text or "").lower()
    return " ".join(_TOKEN_RE.findall(s))


def _char_ngrams(text: str, n: int) -> List[str]:
    chars = [c for c in text if c != " "]
    if len(chars) < n:
        return [text] if text else []
    return ["".join(chars[i:i + n]) for i in range(len(chars) - n + 1)]


class TfidfEmbedder:
    """Character n-gram TF-IDF sparse embedder (deterministic, no deps).

    Call :meth:`fit` on the corpus of index texts first (this fixes the idf
    used for every later ``embed``/query), then embed texts with :meth:`embed`.
    """

    name = "tfidf"

    def __init__(self, ngram_range: Tuple[int, int] = (2, 3)) -> None:
        self.ngram_range = tuple(int(x) for x in ngram_range)
        self._corpus: List[Dict[str, int]] = []
        self._idf: Dict[str, float] = {}
        self._n_docs: int = 0

    def _terms(self, text: str) -> Dict[str, int]:
        norm = _norm_text(text)
        counts: Dict[str, int] = {}
        for n in self.ngram_range:
            for gram in _char_ngrams(norm, n):
                counts[gram] = counts.get(gram, 0) + 1
        return counts

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        self._corpus = [self._terms(t) for t in texts]
        self._n_docs = max(1, len(self._corpus))
        df: Dict[str, int] = {}
        for terms in self._corpus:
            for term in terms:
                df[term] = df.get(term, 0) + 1
        self._idf = {t: math.log((1 + self._n_docs) / (1 + df[t])) + 1.0 for t in df}
        return self

    def embed(self, text: str) -> Dict[str, float]:
        terms = self._terms(text)
        if not self._idf:
            # No corpus fitted yet; treat as identity so cosine stays meaningful.
            return {t: float(c) for t, c in terms.items()}
        return {t: self._idf.get(t, 0.0) * float(c) for t, c in terms.items()}

    def available(self) -> bool:
        return True


# --------------------------------------------------------------------------
# Dense ONNX (bge-style) via onnxruntime
# --------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RE = re.compile(r"\s+")


def _bert_clean_text(text: str) -> str:
    text = _CTRL_RE.sub("", text)
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return _SPACE_RE.sub(" ", text).strip()


def _handle_chinese_chars(text: str) -> str:
    out = []
    for ch in text:
        if _CJK_RE.match(ch):
            out.append(" " + ch + " ")
        else:
            out.append(ch)
    return "".join(out)


class _WordPieceTokenizer:
    """Minimal WordPiece tokenizer backed by a HF ``tokenizer.json``."""

    def __init__(self, tokenizer_json: str, lowercase: bool = False,
                 max_input_chars_per_word: int = 100) -> None:
        with open(tokenizer_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model") or {}
        vocab = model.get("vocab") or {}
        self.vocab: Dict[str, int] = {str(t): int(i) for t, i in vocab.items()}
        self.unk_token = model.get("unk_token", "[UNK]")
        self.prefix = model.get("continuing_subword_prefix", "##")
        self.max_input_chars = int(model.get("max_input_chars_per_word", max_input_chars_per_word))
        self.lowercase = bool(lowercase)
        self.unk_id = self.vocab.get(self.unk_token, 100)
        self.cls_id = self.vocab.get("[CLS]", 101)
        self.sep_id = self.vocab.get("[SEP]", 102)
        self.max_length = 512
        # Special tokens from the post_processor template, if present.
        spec = (data.get("post_processor") or {}).get("special_tokens") or {}
        for tok, info in spec.items():
            ids = (info or {}).get("ids") or []
            if tok == "[CLS]" and ids:
                self.cls_id = int(ids[0])
            if tok == "[SEP]" and ids:
                self.sep_id = int(ids[0])

    def _pre_tokenize(self, text: str) -> List[str]:
        t = _bert_clean_text(text)
        if self.lowercase:
            t = t.lower()
        t = _handle_chinese_chars(t)
        # Split on anything that is not an ascii alnum or CJK char so that
        # JSON / punctuation-heavy contexts become word-like tokens instead of
        # one glued token (matching BERT's pre-tokenizer behaviour).
        return [w for w in re.split(r"[^A-Za-z0-9\u4e00-\u9fff]+", t) if w]

    def _wordpiece(self, word: str) -> List[int]:
        if len(word) > self.max_input_chars:
            return [self.unk_id]
        start = 0
        pieces: List[int] = []
        while start < len(word):
            end = len(word)
            cur = None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = self.prefix + sub
                if sub in self.vocab:
                    cur = self.vocab[sub]
                    break
                end -= 1
            if cur is None:
                pieces.append(self.unk_id)
                start += 1
                continue
            pieces.append(cur)
            start = end
        return pieces

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        ids = [self.cls_id]
        tok_ids = [self.cls_id]
        for word in self._pre_tokenize(text):
            for piece in self._wordpiece(word):
                if len(tok_ids) >= self.max_length - 1:
                    break
                tok_ids.append(piece)
            if len(tok_ids) >= self.max_length - 1:
                break
        tok_ids.append(self.sep_id)
        tok_ids = tok_ids[:self.max_length]
        mask = [1] * len(tok_ids)
        return tok_ids, mask


class OnnxEmbedder:
    """Dense ONNX sentence embedder (bge-small style) via onnxruntime."""

    name = "dense"

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.hidden = int(cfg.get("hidden_size", 384))
        self.lowercase = bool(cfg.get("do_lower_case", False))
        self.tokenizer = _WordPieceTokenizer(
            os.path.join(model_dir, "tokenizer.json"), lowercase=self.lowercase
        )
        sess_path = os.path.join(model_dir, "onnx", "model.onnx")
        if not os.path.isfile(sess_path):
            raise FileNotFoundError(f"ONNX model not found: {sess_path}")
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"onnxruntime is not installed: {exc}") from exc
        self._ort = ort
        self.session = ort.InferenceSession(sess_path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        try:
            import numpy as np  # type: ignore
            self._np = np
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"numpy is required for the dense backend: {exc}") from exc

    def available(self) -> bool:
        return True

    def embed(self, text: str) -> List[float]:
        np = self._np
        ids, mask = self.tokenizer.encode(text)
        feeds = {}
        if "input_ids" in self.input_names:
            feeds["input_ids"] = np.array([ids], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = np.array([mask], dtype=np.int64)
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros((1, len(ids)), dtype=np.int64)
        if not feeds:
            raise RuntimeError("no recognized ONNX inputs in the model session")
        output = self.session.run(None, feeds)[0]
        if output.ndim == 2 and output.shape[1] == self.hidden:
            emb = output[0]
        elif output.ndim == 2:
            emb = output[0]
        else:
            # 3D [B, T, H]: bge-style CLS pooling.
            emb = output[0, 0, :]
        norm = float(np.linalg.norm(emb)) or 1e-12
        return [float(v) / norm for v in emb]


def available_backends() -> List[str]:
    """Names of backends that can be instantiated in this environment."""
    out = ["tfidf"]
    try:
        import onnxruntime  # noqa: F401
        out.append("dense")
    except Exception:
        pass
    return out
