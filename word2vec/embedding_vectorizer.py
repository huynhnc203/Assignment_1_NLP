from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
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
        # Tính vector của câu cần so sánh
        query_vec = np.mean([self.embedding_dict.get(word, np.zeros(self.embedding_dim)) for word in sentence.split()], axis=0)
        
        # Tính vector cho tất cả câu trong embedding_dict
        all_sentences = list(self.embedding_dict.keys())  # Lấy danh sách các câu/từ trong từ điển
        all_vecs = np.array([self.transform([sentence])[0] for sentence in all_sentences])  # Chuyển đổi tất cả các câu/từ thành vector
        
        # Tính độ tương đồng cosine giữa vector của câu query và các vector khác
        if metric == 'cosine':
            similarities = cosine_similarity([query_vec], all_vecs)[0]  # Độ tương đồng cosine
        
        # Tìm k câu có độ tương đồng lớn nhất
        top_k_indices = np.argsort(similarities)[-k:][::-1]  # Sắp xếp theo thứ tự giảm dần và lấy k câu gần nhất
        
        # Trả về các câu tương ứng với các chỉ số tìm được
        return [all_sentences[i] for i in top_k_indices]


