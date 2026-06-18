# 08_arabicmteb_embedding.py
# ArabicMTEB/Swan-style Arabic embedding script

from common_embedding_utils import run_embedding_pipeline

CONFIG = {
    "model_label": "ArabicMTEB_Swan",

    # Replace this checkpoint if you have the official Swan checkpoint.
    # ArabicMTEB is a benchmark; Swan is the related embedding model family.
    "model_checkpoint": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

    "tokenization_settings": {
        "padding": "max_length",
        "truncation": True
    },

    "max_sequence_length": 256,

    "hidden_layer": -1,

    # sentence-transformer style encoder model
    "pooling_strategy": "mean",

    "cls_vs_mean_details": (
        "Mean pooling is used over non-padding tokens. "
        "ArabicMTEB is a benchmark, not a single checkpoint."
    )
}

if __name__ == "__main__":
    run_embedding_pipeline(CONFIG)