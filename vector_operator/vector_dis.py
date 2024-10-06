from abc import ABC, abstractmethod
import numpy as np

class VectorOperator(ABC):
    @abstractmethod
    def calculate(self, vector1, vector2):
        pass

class CosineSimilarity(VectorOperator):
    def calculate(self, vector1, vector2):
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class EuclideanDistance(VectorOperator):
    def calculate(self, vector1, vector2):
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        return np.linalg.norm(vector1 - vector2)