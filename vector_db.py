# vector_db.py
"""
Simple vector database for RAG (Retrieval-Augmented Generation) using FAISS and Sentence Transformers.
Stores input/output pairs, retrieves relevant context for LLM.
"""
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

class VectorDB:
    def __init__(self, db_path="vector_db.pkl", embedding_model="all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model = SentenceTransformer(embedding_model)
        self.data = []  # List of dicts: {"input": str, "output": str, "embedding": np.ndarray}
        self.index = None
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.data = pickle.load(f)
            if self.data:
                embeddings = np.stack([d["embedding"] for d in self.data])
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(embeddings)

    def _save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.data, f)

    def add(self, input_text, output_text):
        embedding = self.model.encode(input_text)
        self.data.append({"input": input_text, "output": output_text, "embedding": embedding})
        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embedding))
        self.index.add(np.expand_dims(embedding, axis=0))
        self._save()

    def query(self, query_text, top_k=3):
        if not self.data or self.index is None:
            return []
        query_emb = self.model.encode(query_text)
        D, I = self.index.search(np.expand_dims(query_emb, axis=0), top_k)
        return [self.data[i] for i in I[0] if i < len(self.data)]

# Example usage:
if __name__ == "__main__":
    db = VectorDB()
    db.add("What is AI?", "AI stands for Artificial Intelligence.")
    db.add("What is Python?", "Python is a programming language.")
    results = db.query("Tell me about Python.")
    for r in results:
        print(f"Input: {r['input']}\nOutput: {r['output']}\n")
