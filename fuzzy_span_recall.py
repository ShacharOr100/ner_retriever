"""Fuzzy span recall calculation for entity extraction.

Fuzzy Recall is used to evaluate the entities detection phase.
"""

from __future__ import annotations

import re
from typing import Sequence, Tuple
import unicodedata
 

STOP_DETS = {"the", "a", "an"}


def normalise(text: str) -> str:
    """Lower-case, strip accents and leading determiners."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    tokens = re.split(r"\W+", text.lower().strip())
    if tokens and tokens[0] in STOP_DETS:
        tokens = tokens[1:]
    return " ".join(tokens)


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def is_match(gold: str, pred: str, min_jacc: float = 0.8) -> bool:
    gold_t, pred_t = set(gold.split()), set(pred.split())
    return (
        gold_t <= pred_t
        or pred_t <= gold_t
        or jaccard(gold_t, pred_t) >= min_jacc
    )


def count_fuzzy_matches(
    gold_texts: Sequence[str],
    pred_texts: Sequence[str],
    min_jacc: float = 0.8,
) -> Tuple[int, int]:
    """Return number of matched gold entities and the total gold count."""
    gold_norm = list({normalise(t) for t in gold_texts if t})
    pred_norm = list({normalise(t) for t in pred_texts if t})

    matched_gold = set()
    matched_pred = set()

    for gi, g in enumerate(gold_norm):
        for pj, p in enumerate(pred_norm):
            if pj in matched_pred:
                continue
            if is_match(g, p, min_jacc):
                matched_gold.add(gi)
                matched_pred.add(pj)
                break

    return len(matched_gold), len(gold_norm)
