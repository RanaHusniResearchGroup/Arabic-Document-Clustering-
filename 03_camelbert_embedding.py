# 03_camelbert_embedding.py
# CAMeLBERT document embedding extraction.
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
# Mean pooling is used over non-padding tokens. The mixed CAMeLBERT checkpoint is selected for general Arabic/dialectal coverage.
#
# Example:
# python 03_camelbert_embedding.py --input input.csv --output embeddings/camelbert_embedding.csv --batch_size 16

from common_embedding_utils import run_embedding_pipeline

CONFIG = {
    "model_label": "CAMeLBERT",
    "model_checkpoint": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "tokenization_settings": {
        "padding": "max_length",
        "truncation": True
    },
    "max_sequence_length": 256,
    "hidden_layer": -1,
    "pooling_strategy": "mean",
    "cls_vs_mean_details": "Mean pooling is used over non-padding tokens. The mixed CAMeLBERT checkpoint is selected for general Arabic/dialectal coverage."
}

if __name__ == "__main__":
    run_embedding_pipeline(CONFIG)
