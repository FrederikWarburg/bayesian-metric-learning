import faiss
import numpy as np


class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.k = k

    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X)

    def kneighbors(self, X):
        distances, indices = self.index.search(X, k=self.k)
        return distances, indices

    def predict(self, X):
        distances, indices = self.index.search(X, k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
