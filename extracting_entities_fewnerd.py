"""Extract entities from the full FewNERD dataset and evaluate fuzzy span recall.

This script downloads the complete FewNERD dataset, predicts entity spans for
each sentence with the ``CascadeNER/models_for_CascadeNER`` language model and
stores the results in a JSON file.  The helper functions that interface with
this model live in ``cascade_llm_entity_extractor.py`` so that the extraction
logic is isolated from the dataset handling shown here.

The output JSON is a list of records with the following structure::

    {
        "id": "<uuid>",
        "sentence": "Barack Obama visited Paris in 2015.",
        "gold": [
            {"text": "Barack Obama", "start": 0, "end": 12, "label": "person/actor"},
            {"text": "Paris", "start": 20, "end": 25, "label": "location/city"}
        ],
        "predicted": [
            {"text": "Barack Obama", "start": 0, "end": 12},
            {"text": "Paris", "start": 20, "end": 25}
        ]
    }

Run the script with::

    python extracting_entities_fewnerd.py --output fewnerd_entities.json

Use ``--limit`` to process only the first ``N`` sentences for quicker demos.
"""

from __future__ import annotations

import argparse
import itertools
import json
import uuid
from typing import List, Sequence, Tuple
from tqdm import tqdm
from datasets import load_dataset

from cascade_llm_entity_extractor import load_cascadener, predict_spans_batch
from fuzzy_span_recall import count_fuzzy_matches

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tags_to_spans(tokens: Sequence[str], tags: Sequence[str]) -> List[Tuple[int, int, str]]:
    """Convert token-level tags to character spans.

    The dataset uses a simple tagging scheme where each token is either
    assigned a label or the string ``"O"``.  Consecutive tokens with the same
    non-``"O"`` label belong to the same span.
    """

    spans: List[Tuple[int, int, str]] = []
    current_label: str | None = None
    prev_tag = "O"
    char_pos = 0
    span_start = 0
    span_end = 0

    def flush() -> None:
        nonlocal current_label, span_start, span_end
        if current_label is not None:
            spans.append((span_start, span_end, current_label))
            current_label = None

    for idx, (token, tag) in enumerate(zip(tokens, tags)):
        token_start = char_pos
        char_pos += len(token)
        token_end = char_pos
        if idx < len(tokens) - 1:
            char_pos += 1  # account for joining spaces

        if tag != "O":
            if tag == prev_tag and current_label is not None:
                span_end = token_end
            else:
                flush()
                current_label = tag
                span_start = token_start
                span_end = token_end
            prev_tag = tag
        else:
            flush()
            prev_tag = "O"
    flush()
    return spans



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="fewnerd_entities.json",
        help="Path to write extracted entity dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of sentences to process at once",
    )
    args = parser.parse_args()

    data_files = {
	    "train": "hf://datasets/DFKI-SLT/few-nerd@refs/convert/parquet/supervised/train/*.parquet",
	    "validation": "hf://datasets/DFKI-SLT/few-nerd@refs/convert/parquet/supervised/validation/*.parquet",
	    "test": "hf://datasets/DFKI-SLT/few-nerd@refs/convert/parquet/supervised/test/*.parquet",
    }
    dataset = load_dataset("parquet", data_files=data_files)
    label_names = dataset["train"].features["ner_tags"].feature.names

    tokenizer, model = load_cascadener()

    records: List[dict] = []
    total_gold = 0
    total_matched = 0

    all_examples = list(dataset["train"]) +  list(dataset["validation"]) + list(dataset["test"])
    batches = list(range(0, len(all_examples), args.batch_size))

    for batch_start_index in tqdm(batches):
        batch = all_examples[batch_start_index:batch_start_index + args.batch_size]
        texts = [" ".join(ex["tokens"]) for ex in batch]
        tags_batch = [[label_names[t] for t in ex["ner_tags"]] for ex in batch]
        gold_batch = [
            tags_to_spans(ex["tokens"], tags)
            for ex, tags in zip(batch, tags_batch)
        ]
        pred_batch = predict_spans_batch(texts, tokenizer, model)
        for text, gold_spans, pred_spans in zip(texts, gold_batch, pred_batch):
            gold_texts = [text[start:end] for start, end, _ in gold_spans]
            pred_texts = [p["text"] for p in pred_spans]
            matched, gold_count = count_fuzzy_matches(gold_texts, pred_texts)
            total_matched += matched
            total_gold += gold_count
            record = {
                "id": str(uuid.uuid4()),
                "sentence": text,
                "gold": [
                    {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "label": label,
                    }
                    for start, end, label in gold_spans
                ],
                "predicted": pred_spans,
            }
            records.append(record)

    recall = total_matched / total_gold if total_gold else 0.0

    with open(args.output, "w", encoding="utf8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} sentences to {args.output}")
    print(f"Span recall: {recall:.3f}")


if __name__ == "__main__":
    main()

