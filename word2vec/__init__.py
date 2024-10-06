from .load_model import (
    load_embeddings
)

from .embedding_vectorizer import (
    EmbeddingVectorizer
)

import os

embedding_file_path = os.getenv("EMBEDDING_FILE_PATH")
embedding_dict = load_embeddings(os.path.join(os.path.dirname(__file__), embedding_file_path))

vectorizer = EmbeddingVectorizer(embedding_dict)