import numpy as np
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parents[2]
FASTTEXT_PATH = BASE_DIR / "data" / "embeddings" / "crawl-300d-2M.vec"
TOKENIZER_PATH = BASE_DIR / "data" / "datasets" / "tokenized" / "tokenizer.pkl"
EMBEDDING_MATRIX_PATH = BASE_DIR / "data" / "embeddings" / "embedding_matrix.npy"

EMBEDDING_DIM = 300


def build_and_save_embedding_matrix(
        fasttext_path=FASTTEXT_PATH,
        tokenizer_path=TOKENIZER_PATH,
        save_path=EMBEDDING_MATRIX_PATH,
        embedding_dim=EMBEDDING_DIM,
        scale=0.6
):
    with open(tokenizer_path, "rb") as f:
        tokenizer_obj = pickle.load(f)

    tokenizer_vocab = tokenizer_obj.get_vocab() if hasattr(tokenizer_obj, "get_vocab") else tokenizer_obj.vocab
    vocab_size = len(tokenizer_vocab)

    print(f"Loading FastText vectors from {fasttext_path}...")
    embeddings_index = {}
    with open(fasttext_path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            values = line.rstrip().split(" ")
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    print("Building embedding matrix...")
    embedding_matrix = np.random.normal(scale=scale, size=(vocab_size, embedding_dim))

    for word, idx in tokenizer_vocab.items():
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector

    np.save(save_path, embedding_matrix)
    print(f"Embedding matrix saved to {save_path}")

    return embedding_matrix


if __name__ == "__main__":
    build_and_save_embedding_matrix()
