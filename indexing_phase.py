"""Generate an embedding for each data entry in the dataset.
"""
from __future__ import annotations
import argparse
import json
import torch

from embedding_utils import load_llm_and_mlp, embed_entities_dataset


def main() -> None:
	parser = argparse.ArgumentParser(description="Embed entities from Llama 3.1 layer-17 v_proj")
	parser.add_argument("--input", default="fewnerd_entities.json", help="Input JSON file")
	parser.add_argument("--output", default="fewnerd_embeddings.pth", help="Output .pth file")
	parser.add_argument("--batch_size", type=int, default=50, help="Batch size for model forward")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer, model, mlp = load_llm_and_mlp(device)

	with open(args.input, "r", encoding="utf8") as f:
		sentences = json.load(f)

	records = embed_entities_dataset(
		sentences=sentences,
		tokenizer=tokenizer,
		model=model,
		mlp=mlp,
		device=device,
		batch_size=args.batch_size,
	)

	torch.save(records, args.output)
	print(
		f"Wrote embeddings for {sum(len(v) for v in records.values())} entities to {args.output}"
	)


if __name__ == "__main__":
	main()
