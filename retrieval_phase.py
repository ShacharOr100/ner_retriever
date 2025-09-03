"""Compute R-Precision for FewNERD fine types using FAISS.

This script evaluates how well fine-type query embeddings retrieve the
corresponding sentences (text IDs) that contain gold entities of that
fine type. It follows the multi-vector R-precision logic: the index is
built over all entity embeddings (multiple per sentence). Retrieval is
performed with FAISS, then de-duplicated to unique text IDs.

Key points:
- Loads sentence-level gold labels from ``fewnerd_entities.json``.
- Loads entity embeddings from ``fewnerd_embeddings.pth`` (see
  ``embed_dataset.py``).
- Generates a query embedding per fine type (using ``embedding_utils``),
  or loads types from a provided JSON list.
- For each fine type ``t``, retrieves top ``k`` unique text IDs where
  ``k`` equals the number of relevant text IDs for ``t``. Because the
  index contains multiple vectors per text, we search up to ``4*k`` and
  then de-duplicate by text ID.

Example:
    python retrieval_evaluation.py \
        --entities fewnerd_entities.json \
        --embeddings fewnerd_embeddings.pth \
        --fine_types fewnerd_fine_types.json

Outputs a compact table with per-type R-precision and overall macro
average.
"""

from __future__ import annotations

import argparse
import os
import json
from typing import Iterable, Set, Dict, Sequence, List, Tuple, Any

import numpy as np
import torch

import faiss

from embedding_utils import load_llm_and_mlp, embed_entities_dataset
from tqdm.auto import tqdm as _tqdm


def _to_float32(x: Iterable[float] | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(list(x))
    return arr.astype("float32", copy=False)


def _l2_normalize(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return mat / norms


def _iter_allowed_labels(
    entities: List[dict], allowed_fine_types: Set[str] | None
) -> Iterable[Tuple[str, str]]:
    for rec in _tqdm(entities, desc="Building fine_type->ids: records", dynamic_ncols=True):
        tid = str(rec["id"])
        for g in rec.get("gold", []):
            ft = g.get("label")
            if isinstance(ft, str) and (allowed_fine_types is None or ft in allowed_fine_types):
                yield tid, ft


def _build_fine_type_to_ids(
    entities: List[dict],
    allowed_fine_types: Set[str] | None = None,
) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = {}
    for tid, ft in _iter_allowed_labels(entities, allowed_fine_types):
        mapping.setdefault(ft, set()).add(tid)
    return mapping


def _append_vec(
    rows: List[np.ndarray], id_map: List[str], tid: str, arr: np.ndarray, dim_state: Dict[str, int | None]
) -> None:
    dim = dim_state.get("dim")
    if dim is None:
        dim_state["dim"] = arr.shape[0]
    elif arr.shape[0] != dim:
        raise ValueError(f"Inconsistent embedding dim: got {arr.shape[0]}, expected {dim}")
    rows.append(arr)
    id_map.append(tid)


def _flatten_embeddings(embeds: Dict[str, Sequence[Sequence[float]]]) -> Tuple[np.ndarray, List[str]]:
    rows: List[np.ndarray] = []
    id_map: List[str] = []
    dim_state: Dict[str, int | None] = {"dim": None}
    items = list(embeds.items())
    for tid, vecs in _tqdm(items, desc="Flatten embeddings: texts", dynamic_ncols=True):
        for v in vecs:
            _append_vec(rows, id_map, tid, _to_float32(v), dim_state)
    if not rows:
        raise ValueError("No embeddings found in the provided .pth file")
    return np.stack(rows, axis=0), id_map



def _build_index(embeddings_by_tid: Dict[str, Sequence[Sequence[float]]]):
    base_mat, id_map = _flatten_embeddings(embeddings_by_tid)
    base_mat = _l2_normalize(base_mat)
    index = faiss.IndexFlatIP(base_mat.shape[1])
    index.add(base_mat)
    return index, id_map


def _make_sentences(fine_type_texts: Dict[str, str]) -> List[dict]:
    sents = []
    for k, v in fine_type_texts.items():
        sents.append({"id": k, "sentence": v, "predicted": [{"start": 0, "end": len(v)}]})
    return sents


def _embed_type_queries(fine_type_texts: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, mlp = load_llm_and_mlp(device)
    records = embed_entities_dataset(
        sentences=_make_sentences(fine_type_texts),
        tokenizer=tokenizer,
        model=model,
        mlp=mlp,
        device=device,
        batch_size=32,
    )
    fine_types = list(fine_type_texts.keys())
    rows = [_to_float32(records[ft][0]) for ft in fine_types]
    return _l2_normalize(np.stack(rows, axis=0)), fine_types


def _search_top(index, query_mat: np.ndarray, fine_types: List[str], fine_type_to_ids: Dict[str, Set[str]]):
    k_list = [max(1, len(fine_type_to_ids.get(ft, set()))) for ft in fine_types]
    max_k = max(k_list) if k_list else 1
    return index.search(query_mat, 4 * max_k)


def _dedup_ids(I_row: np.ndarray, id_map: List[str], k: int) -> List[str]:
    seen: Set[str] = set()
    ranking: List[str] = []
    for idx in I_row:
        j = int(idx)
        if 0 <= j < len(id_map):
            tid = id_map[j]
            if tid not in seen:
                seen.add(tid)
                ranking.append(tid)
                if len(ranking) >= k:
                    break
    return ranking[:k]


def compute_r_precision(
    embeddings_by_tid: Dict[str, Sequence[Sequence[float]]],
    fine_type_texts: Dict[str, str],
    fine_type_to_ids: Dict[str, Set[str]],
) -> Dict[str, float]:
    index, id_map = _build_index(embeddings_by_tid)
    query_mat, fine_types = _embed_type_queries(fine_type_texts)
    _, I = _search_top(index, query_mat, fine_types, fine_type_to_ids)
    r_prec: Dict[str, float] = {}
    for row, ft in enumerate(_tqdm(fine_types, desc="Compute R-precision per type", dynamic_ncols=True)):
        relevant = fine_type_to_ids.get(ft, set())
        k = max(1, len(relevant))
        retrieved = _dedup_ids(I[row, :], id_map, k)
        r_prec[ft] = len(set(retrieved) & relevant) / float(k) if k else 0.0
    return r_prec


def _derive_fine_types(entities: List[dict]) -> List[str]:
    s: Set[str] = set()
    for rec in _tqdm(entities, desc="Deriving fine types: records", dynamic_ncols=True):
        for g in rec.get("gold", []):
            lab = g.get("label")
            if isinstance(lab, str):
                s.add(lab)
    return sorted(s)


def _load_fine_type_list(path: str | None, entities: List[dict]) -> List[str]:
    if path:
        with open(path, "r", encoding="utf8") as f:
            types_list = json.load(f)
        if not isinstance(types_list, list):
            raise ValueError("fine_types file must be a JSON list of strings")
        return [str(x) for x in types_list]
    return _derive_fine_types(entities)


def _load_name_map(mapping_path: str | None) -> Dict[str, str]:
    path = mapping_path or os.path.join(os.path.dirname(__file__), "entities_to_names.json")
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _load_fine_types(path: str | None, entities: List[dict], name_map_path: str | None = None) -> Dict[str, str]:
    fine_types = _load_fine_type_list(path, entities)
    name_map = _load_name_map(name_map_path)
    return {ft: name_map.get(ft.split("-")[-1], ft) for ft in fine_types}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R-Precision for FewNERD fine types (FAISS retrieval)")
    p.add_argument("--entities", default="fewnerd_entities.json", help="Path to FewNERD sentence JSON")
    p.add_argument("--embeddings", default="fewnerd_embeddings.pth", help="Path to entity embeddings .pth")
    p.add_argument("--type-name-map", dest="type_name_map", default=None,
                   help="Path to JSON mapping fine_type -> readable name (defaults to demo/entities_to_names.json)")
    return p.parse_args()


def _render_results(fine_type_texts: Dict[str, str], fine_type_to_ids: Dict[str, Set[str]], rp: Dict[str, float]) -> None:
    rows, macro = _build_rows_and_macro(fine_type_texts, fine_type_to_ids, rp)
    header = f"{'fine_type':40s}  {'size':>6s}  {'R-precision':>11s}"
    print(header)
    print("-" * len(header))
    for ft, size, r in _tqdm(rows, desc="Printing rows", dynamic_ncols=True, leave=False):
        print(f"{ft:40.40s}  {size:6d}  {r:11.4f}")
    print("\nMacro R-precision (non-empty types): {:.4f}".format(macro))


def _build_rows_and_macro(
    fine_type_texts: Dict[str, str],
    fine_type_to_ids: Dict[str, Set[str]],
    rp: Dict[str, float],
) -> Tuple[List[Tuple[str, int, float]], float]:
    rows: List[Tuple[str, int, float]] = []
    for ft in _tqdm(sorted(fine_type_texts.keys()), desc="Collecting per-type results", dynamic_ncols=True):
        rows.append((ft, len(fine_type_to_ids.get(ft, set())), rp.get(ft, 0.0)))
    filtered = [r for ft, size, r in rows if size > 0 and str(fine_type_texts.get(ft, "")).strip().lower() != "other"]
    macro = float(np.mean(filtered)) if filtered else 0.0
    return rows, macro


def compute_rprecision_report(
    entities_path: str,
    embeddings_path: str,
    type_name_map_path: str | None = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Compute per-type R-precision and macro average, returning structured rows.

    Parameters
    - entities_path: path to FewNERD sentence JSON (as produced by entity detection phase)
    - embeddings_path: path to entity embeddings .pth (as produced by indexing phase)
    - type_name_map_path: optional JSON mapping fine_type -> readable name used as query text

    Returns
    - rows: list of dicts with keys: 'fine_type', 'size', 'r_precision'
    - macro: macro R-precision over non-empty, non-'other' types
    """
    with open(entities_path, "r", encoding="utf8") as f:
        entities = json.load(f)
    embeds = torch.load(embeddings_path)
    if not isinstance(embeds, dict):
        raise ValueError("Embeddings file must be a dict[text_id] -> List[vector]")
    allowed_ids = {str(rec["id"]) for rec in entities}
    embeds = {str(tid): vecs for tid, vecs in embeds.items() if str(tid) in allowed_ids}

    fine_texts = _load_fine_types(path=None, entities=entities, name_map_path=type_name_map_path)
    ft_to_ids = _build_fine_type_to_ids(entities, allowed_fine_types=set(fine_texts.keys()))
    rp = compute_r_precision(embeds, fine_texts, ft_to_ids)

    tuple_rows, macro = _build_rows_and_macro(fine_texts, ft_to_ids, rp)
    rows: List[Dict[str, Any]] = [
        {"fine_type": ft, "size": size, "r_precision": r} for ft, size, r in tuple_rows
    ]
    return rows, macro


def main() -> None:
    args = _parse_args()
    # Reuse the report-building logic to avoid duplication
    with open(args.entities, "r", encoding="utf8") as f:
        entities = json.load(f)
    embeds = torch.load(args.embeddings)
    if not isinstance(embeds, dict):
        raise ValueError("Embeddings file must be a dict[text_id] -> List[vector]")
    allowed_ids = {str(rec["id"]) for rec in entities}
    embeds = {str(tid): vecs for tid, vecs in embeds.items() if str(tid) in allowed_ids}
    fine_texts = _load_fine_types(path=None, entities=entities, name_map_path=args.type_name_map)
    ft_to_ids = _build_fine_type_to_ids(entities, allowed_fine_types=set(fine_texts.keys()))
    rp = compute_r_precision(embeds, fine_texts, ft_to_ids)
    _render_results(fine_texts, ft_to_ids, rp)


if __name__ == "__main__":
    main()
