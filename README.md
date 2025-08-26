# FewNERD R-Precision Demo

This directory contains a minimal, self-contained demonstration of the
entity retrieval workflow used in this project. The scripts deliberately
avoid importing modules from the rest of the repository so that they can
serve as stand-alone examples. When third-party libraries such as
`datasets`, `transformers` or `torch` are available the code will use them.

The demo follows three steps:

1. **Entity extraction and span evaluation** – `extracting_entities_fewnerd.py`
   downloads the full FewNERD dataset, uses the
   `CascadeNER/models_for_CascadeNER` language model via
   `cascade_llm_entity_extractor.py` to mark entities and prints a
   fuzzy span-level recall (matching logic mirrors the original
   `span_extraction_eval` script). The output `fewnerd_entities.json`
   contains one record per sentence with both the gold and predicted spans.
2. **Forward pass to obtain embeddings** – `forward_llm_mlp_fewnerd.py`
   reads `fewnerd_entities.json`, extracts every gold entity span and encodes
   it using block 17 of the `Meta-Llama-3.1` model followed by an MLP head
   loaded from `entity_head.pth`.  The embeddings are written as a binary file
   `fewnerd_embeddings.pth` that maps each sentence UUID to the list of its
   entity vectors.
3. **R-precision evaluation** – `fewnerd_r_precision_prediction.py` loads
   `fewnerd_embeddings.pth` and the original `fewnerd_entities.json`, groups
   entities by their fine type and reports the R-precision for each type
   together with the macro average.  Cosine similarity search is performed via
   [FAISS](https://github.com/facebookresearch/faiss) when available, falling
   back to a small pure-Python implementation otherwise.

Typical usage::

    python extracting_entities_fewnerd.py --limit 1000
    python forward_llm_mlp_fewnerd.py
    python fewnerd_r_precision_prediction.py --embeddings fewnerd_embeddings.pth --entities fewnerd_entities.json

Each script provides `--help` for additional options.
