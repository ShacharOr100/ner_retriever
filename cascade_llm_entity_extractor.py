"""Utilities for predicting entity spans with the CascadeNER extractor.

This module wraps the `CascadeNER/models_for_CascadeNER` language model and
provides a simple API for generating entity predictions. The model annotates
entities by surrounding them with double hash marks (``##like this##``). The
functions below convert that annotated text back into character offsets
relative to the original sentence.

The helpers are intentionally standalone so that demo scripts can import this
file without relying on any project-internal modules.
"""

from __future__ import annotations

import re
from typing import List, Dict, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

__all__ = ["load_cascadener", "predict_spans_batch", "align_to_original", "extract_entities"]


# ---------------------------------------------------------------------------
# Model loading and prompting
# ---------------------------------------------------------------------------

def load_cascadener():
    """Load the CascadeNER extraction model and tokenizer.

    Returns
    -------
    tokenizer, model: The HuggingFace tokenizer and causal language model
        configured for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "CascadeNER/models_for_CascadeNER",
        subfolder="extractor",
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "CascadeNER/models_for_CascadeNER",
        subfolder="extractor",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).cuda().eval()
    return tokenizer, model


def build_prompt(sentence: str, tokenizer) -> str:
    """Create a chat-style prompt for the CascadeNER model."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": sentence},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Entity span extraction
# ---------------------------------------------------------------------------

def _is_open(text: str, i: int, inside: bool) -> bool:
    if not inside:
        return True
    nxt = i + 2
    return nxt < len(text) and text[nxt].isalnum()


def extract_entities(text: str, longest_only: bool = False) -> List[Tuple[str, int, int]]:
    """Extract entity markers from ``text``.

    Parameters
    ----------
    text: str
        The model output containing ``##entity##`` markers.
    longest_only: bool, optional
        When ``True``, nested entities are removed by keeping only the
        longest span at each location.

    Returns
    -------
    list of tuples ``(entity_text, start, end)`` where ``start`` and ``end``
    are character offsets within the unmarked text.
    """
    clean: List[str] = []
    clean_pos = 0
    i = 0
    stack: List[int] = []
    entities: List[Tuple[str, int, int]] = []

    while i < len(text):
        if text.startswith("##", i):
            if _is_open(text, i, inside=bool(stack)):
                stack.append(clean_pos)
            elif stack:
                start = stack.pop()
                entities.append(("".join(clean[start:clean_pos]), start, clean_pos))
            i += 2
            continue

        clean.append(text[i])
        clean_pos += 1
        i += 1

    if longest_only:
        outer: List[Tuple[str, int, int]] = []
        for t, s, e in sorted(entities, key=lambda x: x[2] - x[1], reverse=True):
            if not any(s >= os and e <= oe for _, os, oe in outer):
                outer.append((t, s, e))
        return outer

    return entities


def _find_next(hay: str, needle: str, start: int) -> Tuple[int, int]:
    k = hay.find(needle, start)
    return (k, k + len(needle)) if k != -1 else (-1, -1)


def _remove_nested(spans: List[Dict]) -> List[Dict]:
    keep: List[Dict] = []
    for sp in sorted(spans, key=lambda d: d["end"] - d["start"], reverse=True):
        if not any(sp["start"] >= k["start"] and sp["end"] <= k["end"] for k in keep):
            keep.append(sp)
    return sorted(keep, key=lambda d: d["start"])


def align_to_original(
    original: str,
    marked: str,
    longest_only: bool = True,
    all_occurrences: bool = True,
) -> List[Dict]:
    """Align ``marked`` text with ``original`` and return entity spans."""
    ents = extract_entities(marked, longest_only)
    out: List[Dict] = []
    cur = 0

    for text, _, _ in ents:
        if all_occurrences:
            for m in re.finditer(re.escape(text), original):
                out.append({"text": text, "start": m.start(), "end": m.end()})
        else:
            b, e = _find_next(original, text, cur)
            if b != -1:
                out.append({"text": text, "start": b, "end": e})
                cur = e

    return _remove_nested(out)



def predict_spans_batch(
    sentences: List[str],
    tokenizer,
    model,
    *,
    max_new_tokens: int = 2096,
) -> List[List[Dict]]:
    """Batch version of :func:`predict_spans` for multiple sentences."""
    prompts = [build_prompt(s, tokenizer) for s in sentences]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_lens = (enc["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    out = model.generate(**enc, max_new_tokens=max_new_tokens)
    preds: List[List[Dict]] = []
    for i, sent in enumerate(sentences):
        gen_tokens = out[i][input_lens[i]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        preds.append(align_to_original(sent, gen_text))
    return preds
