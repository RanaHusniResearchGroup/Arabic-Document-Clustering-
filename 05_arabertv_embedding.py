# 05_arabertv_embedding.py
# AraBERTv2 document embedding extraction.
#
# Unified settings:
# - Input columns: document, processed_document, id, label
# - Text used for embedding: processed_document by default
# - Output columns: document, processed_document, id, label, embedding
# - Hidden layer: -1
# - Pooling strategy: mean
# - Max sequence length: 256
# - Tokenization: padding=max_length, truncation=True
#
# CLS vs mean/last-token details:
# Mean pooling is used over non-padding tokens. This script keeps AraBERTv2 separate from AraBERTv02.
#
# Example:
# python 05_arabertv_embedding.py --input input.csv --output embeddings/arabertv_embedding.csv --batch_size 16

from common_embedding_utils import run_embedding_pipeline

CONFIG = {
    "model_label": "AraBERTv2",
    "model_checkpoint": "aubmindlab/bert-base-arabertv2",
    "tokenization_settings": {
        "padding": "max_length",
        "truncation": True
    },
    "max_sequence_length": 256,
    "hidden_layer": -1,
    "pooling_strategy": "mean",
    "cls_vs_mean_details": "Mean pooling is used over non-padding tokens. This script keeps AraBERTv2 separate from AraBERTv02."
}

if __name__ == "__main__":
    run_embedding_pipeline(CONFIG)
