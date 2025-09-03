"""Extract entities from the full FewNERD dataset and evaluate recall.

This script downloads the complete FewNERD dataset, locates entity spans for
each sentence with the ``CascadeNER/models_for_CascadeNER`` language model and
stores the results in a JSON file.  The helper functions that interface with
this model live in ``cascade_llm_entity_extractor.py``.

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

"""
from __future__ import annotations
import argparse
import json
import uuid
from typing import Sequence, List, Tuple

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

def _load_fewnerd_dataset():
    data_files = {
        "train": "hf://datasets/DFKI-SLT/few-nerd@refs/convert/parquet/supervised/train/*.parquet",
        "validation": "hf://datasets/DFKI-SLT/few-nerd@refs/convert/parquet/supervised/validation/*.parquet",
        "test": "hf://datasets/DFKI-SLT/few-nerd@refs/convert/parquet/supervised/test/*.parquet",
    }
    dataset = load_dataset("parquet", data_files=data_files)
    label_names = dataset["train"].features["fine_ner_tags"].feature.names
    return dataset, label_names


def _build_record(sentence_text: str, gold_spans, predicted_spans) -> dict:
    record = {
        "id": str(uuid.uuid4()),
        "sentence": sentence_text,
        "gold": [
            {"text": sentence_text[start:end], "start": start, "end": end, "label": label}
            for start, end, label in gold_spans
        ],
        "predicted": predicted_spans,
    }
    return record


def _count_matches(sentence_text: str, gold_spans, predicted_spans) -> Tuple[int, int]:
    gold_texts = [sentence_text[start:end] for start, end, _ in gold_spans]
    predicted_texts = [pred["text"] for pred in predicted_spans]
    matched_count, gold_count = count_fuzzy_matches(gold_texts, predicted_texts)
    return matched_count, gold_count


def _process_dataset(dataset, label_names: List[str], tokenizer, model, batch_size: int):
    records: List[dict] = []
    total_gold = 0
    total_matched = 0
    all_records = list(dataset["train"]) + list(dataset["validation"]) + list(dataset["test"])
    batches = range(0, len(all_records), batch_size)
    for batch_start_index in tqdm(batches, desc="Processing FewNERD"):
        batch_records = all_records[batch_start_index: batch_start_index + batch_size]
        sentence_texts = [" ".join(record["tokens"]) for record in batch_records]
        tags_per_record = [
            [label_names[label_index] for label_index in record["fine_ner_tags"]]
            for record in batch_records
        ]
        gold_spans_per_record = [
            tags_to_spans(record["tokens"], tags_for_record)
            for record, tags_for_record in zip(batch_records, tags_per_record)
        ]
        predicted_spans_per_record = predict_spans_batch(sentence_texts, tokenizer, model)
        records_for_batch = [
            _build_record(sentence_text, gold_spans, predicted_spans)
            for sentence_text, gold_spans, predicted_spans in zip(
                sentence_texts, gold_spans_per_record, predicted_spans_per_record
            )
        ]
        counts_for_batch = [
            _count_matches(sentence_text, gold_spans, predicted_spans)
            for sentence_text, gold_spans, predicted_spans in zip(
                sentence_texts, gold_spans_per_record, predicted_spans_per_record
            )
        ]
        records.extend(records_for_batch)
        total_matched += sum(matched for matched, _ in counts_for_batch)
        total_gold += sum(gold for _, gold in counts_for_batch)
    return records, total_matched, total_gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="fewnerd_entities.json", help="Path to write extracted entity dataset")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of sentences to process at once")
    args = parser.parse_args()
    dataset, label_names = _load_fewnerd_dataset()
    tokenizer, model = load_cascadener()
    records, total_matched, total_gold = _process_dataset(
        dataset=dataset,
        label_names=label_names,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
    )
    recall = total_matched / total_gold if total_gold else 0.0
    with open(args.output, "w", encoding="utf8") as output_file:
        json.dump(records, output_file, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} sentences to {args.output}")
    print(f"Span recall: {recall:.3f}")


if __name__ == "__main__":
    main()
