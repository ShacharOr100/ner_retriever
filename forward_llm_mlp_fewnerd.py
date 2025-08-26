"""Forward FewNERD entities through Llama 3.1 and an MLP head.

This script reads the sentence-level JSON produced by
``extracting_entities_fewnerd.py`` and generates an embedding for every gold
entity span.  The embedding is taken from block 17 of ``Meta‑Llama‑3.1`` and
projected through an MLP whose architecture mirrors the training-time model in
``mlp.py``.  The resulting vectors are written as a binary ``.pth`` file mapping
each sentence identifier to a list of its entity embeddings.

Run the script with::

    python forward_llm_mlp_fewnerd.py \
        --input fewnerd_entities.json --output fewnerd_embeddings.pth
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EMBED_DIM = 500  # dimensionality of the final embedding


class Gate(torch.nn.Module):
    """Per-dimension gating module copied from ``mlp.py``."""

    def __init__(self, size: int):
        super().__init__()
        self.dimension = size
        self.gate = torch.nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return x * torch.sigmoid(self.gate)

    def extra_repr(self) -> str:  # pragma: no cover - diagnostic
        return f"dimension={self.dimension}"


@dataclass
class MLPArgs:
    input_layer: int = 1024
    hidden_layer: int = 500
    output_layer: int = 500
    enable_gate: bool = True
    activation: str = "silu"
    noise: str = "dropout"
    is_hidden_layer: bool = True
    dropout: float = 0.1


class MLP(torch.nn.Module):
    """MLP head matching the training configuration in ``mlp.py``."""

    def __init__(self, args: MLPArgs):
        super().__init__()
        gate = Gate(args.input_layer) if args.enable_gate else torch.nn.Identity()
        activation = self._build_activation(args.activation)
        noise = self._build_noise(args)
        middle = self._build_middle_layer(args.input_layer, args)
        output = self._build_output_layer(args.input_layer, args)

        self.net = torch.nn.Sequential(gate, middle, activation, noise, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x)

    @staticmethod
    def _build_activation(name: str) -> torch.nn.Module:
        if name == "silu":
            return torch.nn.SiLU()
        if name == "leaky_relu":
            return torch.nn.LeakyReLU()
        return torch.nn.ReLU()

    @staticmethod
    def _build_noise(args: MLPArgs) -> torch.nn.Module:
        if args.noise == "dropout":
            return torch.nn.Dropout(args.dropout)
        return torch.nn.Identity()

    @staticmethod
    def _build_middle_layer(input_layer: int, args: MLPArgs) -> torch.nn.Module:
        if args.is_hidden_layer:
            return torch.nn.Linear(input_layer, args.hidden_layer)
        return torch.nn.Identity()

    @staticmethod
    def _build_output_layer(input_layer: int, args: MLPArgs) -> torch.nn.Module:
        if args.is_hidden_layer:
            return torch.nn.Linear(args.hidden_layer, args.output_layer)
        return torch.nn.Linear(input_layer, args.output_layer)


def embed_with_llama(
    sentences: List[Dict],
    tokenizer,
    model,
    mlp: torch.nn.Module,
    device: torch.device,
) -> Dict[str, List[List[float]]]:
    """Embed entities using block 17 of Llama 3.1."""

    records: Dict[str, List[List[float]]] = {}
    for sent in sentences:
        text = sent["sentence"]
        enc = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
        hidden = outputs.hidden_states[17][0]  # block 17 output
        offsets = enc["offset_mapping"][0].tolist()

        embs: List[List[float]] = []
        for ent in sent.get("gold", []):
            start_tok = next(
                (i for i, (s, e) in enumerate(offsets) if s <= ent["start"] < e),
                None,
            )
            end_tok = next(
                (i for i, (s, e) in enumerate(offsets) if s < ent["end"] <= e),
                None,
            )
            if start_tok is None or end_tok is None:
                continue
            start_vec = hidden[start_tok - 1]
            end_vec = hidden[end_tok]
            diff = end_vec - start_vec
            vec = mlp(diff)
            embs.append(vec.cpu().tolist())

        if embs:
            records[str(sent["id"])] = embs

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="fewnerd_entities.json", help="Input JSON file")
    parser.add_argument("--output", default="fewnerd_embeddings.pth", help="Output .pth file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf8") as f:
        sentences = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B",
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B",
        output_hidden_states=True,
        torch_dtype=torch.float32,
    ).to(device).eval()

    mlp_args = MLPArgs()
    mlp = MLP(mlp_args).to(device)
    mlp.load_state_dict(torch.load("entity_head.pth", map_location=device))
    mlp.eval()

    records = embed_with_llama(sentences, tokenizer, model, mlp, device)

    torch.save(records, args.output)
    print(f"Wrote embeddings for {sum(len(v) for v in records.values())} entities to {args.output}")


if __name__ == "__main__":
    main()

