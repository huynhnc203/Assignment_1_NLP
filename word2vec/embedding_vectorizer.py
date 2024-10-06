from sklearn.base import BaseEstimator, TransformerMixin
from vector_operator.vector_dis import VectorOperator
import numpy as np

class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_dict):
        self.embedding_dict = embedding_dict
        self.embedding_dim = len(next(iter(embedding_dict.values())))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.embedding_dict.get(word, np.zeros(self.embedding_dim)) for word in sentence.split()], axis=0)
            for sentence in X
        ])

    def k_nearest(self, sentence, k=5, metric='cosine'):
        "bai tap 2 :)) "
        pass

