from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from document_processing import load_document

class EmbeddingEngine:
    def __init__(self, model_name:str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
    

class FAISSVectorStore:
    def __init__(self, dim=None, index_path="faiss_index.index", metadata_path="faiss_metadata.pkl", overwrite=False):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.metadata = []

        if not overwrite and os.path.exists(index_path) and os.path.exists(metadata_path):
            self.load()
        else:
            if dim is None:
                raise ValueError("Must provide dimension if creating new index")
            self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings, metadata_list):
        # Auto-create index if not already loaded
        if not hasattr(self, 'index'):
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(np.array(embeddings).astype("float32"))
        self.metadata.extend(metadata_list)

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(np.array([query_embedding]).astype("float32"), top_k)
        results = []
        for i in indices[0]:
            if i < len(self.metadata):
                results.append(self.metadata[i])
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

class ClauseQueryEngine:
    def __init__(self, model_name: str, vector_store_path="faiss_index.index", metadata_path="faiss_metadata.pkl"):
        self.embedder = EmbeddingEngine(model_name=model_name)
        self.store = FAISSVectorStore(index_path=vector_store_path, metadata_path=metadata_path)

    def search(self, query: str, top_k: int = 5):
        query_vector = self.embedder.encode([f"query: {query}"])[0]  # BGE: use "query: " prefix
        results = self.store.search(query_vector, top_k)
        return results
