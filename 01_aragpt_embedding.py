# 01_aragpt_embedding.py
# AraGPT2 document embedding extraction.
#
# Unified settings:
# - Input columns: document, processed_document, id, label
# - Text used for embedding: processed_document by default
# - Output columns: document, processed_document, id, label, embedding
# - Hidden layer: -1
# - Pooling strategy: last_token
# - Max sequence length: 256
# - Tokenization: padding=max_length, truncation=True
#
# CLS vs mean/last-token details:
# Decoder-only GPT models do not use CLS. This script uses the last valid token representation as the document embedding.
#
# Example:
# python 01_aragpt_embedding.py --input input.csv --output embeddings/aragpt_embedding.csv --batch_size 16

from common_embedding_utils import run_embedding_pipeline

CONFIG = {
    "model_label": "AraGPT2",
    "model_checkpoint": "aubmindlab/aragpt2-base",
    "tokenization_settings": {
        "padding": "max_length",
        "truncation": True
    },
    "max_sequence_length": 256,
    "hidden_layer": -1,
    "pooling_strategy": "last_token",
    "cls_vs_mean_details": "Decoder-only GPT models do not use CLS. This script uses the last valid token representation as the document embedding."
}

if __name__ == "__main__":
    run_embedding_pipeline(CONFIG)
