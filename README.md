# NER Retriever: Zero-Shot Named Entity Retrieval with Type-Aware Embeddings


![Main Figure](figures/main_figure.png)

This is the official repository of the paper NER Retriever, where a user-defined type
description is used to retrieve documents mentioning entities of that type.
For example, If the user would like to retrieve all documents containing a dinosaur,
all documents containing a name of a dinosaur of any kind would be retrieved.

The workflow of NER Retriever contains three parts:

1. **Entity spans extraction** – `extracting_entities_fewnerd.py`
   downloads the full FewNERD dataset, uses the
   `CascadeNER/models_for_CascadeNER` language model via
   `cascade_llm_entity_extractor.py` to mark entities and prints a
   fuzzy span-level recall (matching logic mirrors the original
   `span_extraction_eval` script). The output `fewnerd_entities.json`
   contains one record per sentence with both the gold and predicted spans.
2. **Embedding of entity instances** – `forward_llm_mlp_fewnerd.py`
   reads `fewnerd_entities.json`, extracts every gold entity span and encodes
   it using block 17 of the `Meta-Llama-3.1` model followed by an MLP head
   loaded from `entity_head.pth`.  The embeddings are written as a binary file
   `fewnerd_embeddings.pth` that maps each sentence UUID to the list of its
   entity vectors.
3. **Metric Evaluation (R-precision)** – `fewnerd_r_precision_prediction.py` loads
   `fewnerd_embeddings.pth` and the original `fewnerd_entities.json`, groups
   entities by their fine type and reports the R-precision for each type
   together with the macro average.  Cosine similarity search is performed via
   [FAISS](https://github.com/facebookresearch/faiss) when available, falling
   back to a small pure-Python implementation otherwise.

![Main Figure](figures/pipeline.pdf)


Typical usage:

    python extracting_entities_fewnerd.py
    python forward_llm_mlp_fewnerd.py
    python fewnerd_r_precision_prediction.py --embeddings fewnerd_embeddings.pth --entities fewnerd_entities.json

Each script provides `--help` for additional options.
