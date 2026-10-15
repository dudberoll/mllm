# NLP and LLM Coursework

A collection of practical NLP/LLM work: sequence modeling, parameter-efficient fine-tuning, and retrieval-augmented generation.

- **Seq2Seq and attention:** GRU encoder-decoder with an attention mechanism.
- **PEFT:** prompt tuning and LoRA-based adaptation of a language model. The notebook is [`notebooks/seq2seq_peft.ipynb`](notebooks/seq2seq_peft.ipynb).
- **RAG:** a question-answering prototype over Russian-history PDFs, with document ingestion, chunking, embeddings, vector search, and answer generation. The implementation is in this directory (`data_ingestion.py`, `preprocessing.py`, `embedding.py`, `vector_db.py`, and `generation.py`); automated checks are in [`tests/`](tests/).

The notebook downloads its model and datasets at runtime. The RAG pipeline uses the documents in [`data/`](data/).
