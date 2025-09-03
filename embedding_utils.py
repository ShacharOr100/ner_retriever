"""Utility functions for embedding text with the LLM+MLP stack."""
from __future__ import annotations
import torch
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlp import MLP

# ---- hook cache ----
cache: Dict[str, torch.Tensor] = {}


def hooked_layer_function(_mod, _inp, out):
	# out: (B, L, hidden_size)
	cache.clear()
	cache["output"] = out.detach()


def load_llm_and_mlp(device: torch.device):
	"""Initialise tokenizer, Llama model and MLP head with hook registered."""
	mlp = MLP().to(device)
	mlp.load_state_dict(torch.load("contrastive_projection_head.pth", map_location=device))
	mlp = mlp.half()
	mlp.eval()

	# Login for HF gated models
	login()

	tokenizer = AutoTokenizer.from_pretrained(
		"meta-llama/Meta-Llama-3.1-8B",
		use_fast=True,
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"

	model = AutoModelForCausalLM.from_pretrained(
		"meta-llama/Meta-Llama-3.1-8B",
		output_hidden_states=False,
		torch_dtype=torch.float16,
	).to(device).eval()
	model.config.pad_token_id = tokenizer.pad_token_id

	# Register hook on layer-17 v_proj.
	# This is need in order to get to in layer representation.
	v_proj = model.model.layers[17].self_attn.v_proj
	v_proj.register_forward_hook(hooked_layer_function)

	return tokenizer, model, mlp


def _token_indices_from_offsets(offsets_list, start_idx: int, end_idx: int):
	"""
	This function returns the first and last token index of a phrase in the sentence,
	given the location of the phrase in the text.
	"""
	first_token_idx, last_token_idx = None, None
	for i, (token_start, token_end) in enumerate(offsets_list):
		if token_start <= start_idx < token_end and first_token_idx is None:
			first_token_idx = i
		if token_start < end_idx <= token_end:
			last_token_idx = i
			break
	if first_token_idx is None or last_token_idx is None:
		raise ValueError(
			f"Could not map text span ({start_idx}, {end_idx}) to token indices"
		)
	return first_token_idx, last_token_idx


def embed_entities_batch(
		batch,
		tokenizer,
		model,
		mlp: torch.nn.Module,
		device: torch.device,
):
	"""Embed entities for a batch of sentence records.

	Each record is expected to be a dict with at least:
	  - id: unique identifier
	  - sentence: raw text
	  - predicted: list of spans, each having keys "start" and "end" (char indices)

	The representation is taken from the Llama 3.1 layer-17 v_proj at the token
	whose character span contains the entity's end offset, then projected by the MLP.

	Returns a mapping: id -> list of embedding vectors (as Python lists of floats).
	"""
	model.eval()

	texts = [s["sentence"] for s in batch]
	tokens = tokenizer(
		texts,
		return_tensors="pt",
		padding=True,
		truncation=True,
		return_offsets_mapping=True,
		add_special_tokens=True,
	)
	offsets = tokens.pop("offset_mapping")
	tokens = {k: v.to(device) for k, v in tokens.items()}

	with torch.no_grad():
		cache.clear()
		_ = model(
			input_ids=tokens["input_ids"],
			attention_mask=tokens["attention_mask"],
			output_hidden_states=False,
			use_cache=False,
		)

	result = {}
	if "output" not in cache:
		return result

	vproj_batch = cache["output"]  # (B, L, H)

	for b_idx, sent in enumerate(batch):
		vproj = vproj_batch[b_idx]
		offsets_b = offsets[b_idx].tolist()

		embs = []
		for ent in sent.get("predicted", []):
			if ent.get("end") == ent.get("start"):
				continue
			try:
				_, last_tok_index = _token_indices_from_offsets(offsets_b, ent["start"], ent["end"])
			except Exception:
				continue
			# Use the end token representation only (choose_llm_representation = 'end')
			end_token = vproj[last_tok_index]
			embs.append(mlp(end_token).detach().cpu().tolist())

		if embs:
			result[str(sent["id"])] = embs

	return result


def embed_entities_dataset(
		sentences,
		tokenizer,
		model,
		mlp: torch.nn.Module,
		device: torch.device,
		batch_size: int = 8,
):
	"""Embed entities for a full dataset by batching and calling embed_entities_batch."""

	all_results: Dict[str, list] = {}
	for start in tqdm(range(0, len(sentences), batch_size)):
		batch = sentences[start: start + batch_size]
		batch_result = embed_entities_batch(batch, tokenizer, model, mlp, device)
		for k, v in batch_result.items():
			all_results.setdefault(k, []).extend(v)
	return all_results
