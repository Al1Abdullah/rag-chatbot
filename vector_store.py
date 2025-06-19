import os
import time
import pickle
import hashlib
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        """Initialize FAISS vector store and embedding model"""
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384
        self.index_path = "faiss_index.bin"
        self.docs_path = "faiss_docs.pkl"

        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, "rb") as f:
                self.docs = pickle.load(f)
            print(f"✅ Loaded existing FAISS index with {len(self.docs)} documents")
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.docs = []
            print("🆕 Created new FAISS index")

    def add_documents(self, chunks: List[Dict]) -> bool:
        try:
            if not chunks:
                print("⚠️ No chunks to add")
                return False

            print(f"📥 Adding {len(chunks)} chunks to FAISS vector store...")
            texts = [chunk['text'] for chunk in chunks]
            vectors = self.embedding_model.encode(texts, show_progress_bar=True)
            self.index.add(np.array(vectors).astype("float32"))

            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk['vector_index'] = len(self.docs) + i
                chunk['chunk_id'] = chunk.get('chunk_id', i)
                self.docs.append(chunk)

            # Save index and docs
            faiss.write_index(self.index, self.index_path)
            with open(self.docs_path, "wb") as f:
                pickle.dump(self.docs, f)

            print(f"✅ Successfully added and saved {len(chunks)} documents.")
            return True
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            return False

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            query_vec = self.embedding_model.encode([query])
            D, I = self.index.search(np.array(query_vec).astype("float32"), top_k)

            similar_docs = []
            for idx in I[0]:
                if idx < len(self.docs):
                    doc = self.docs[idx]
                    similar_docs.append({
                        'id': self._create_chunk_id(doc, idx),
                        'score': float(D[0][list(I[0]).index(idx)]),
                        'text': doc.get('text', ''),
                        'url': doc.get('url', ''),
                        'title': doc.get('title', ''),
                        'chunk_id': doc.get('chunk_id', 0)
                    })
            print(f"🔎 Found {len(similar_docs)} similar documents for: '{query[:50]}...'")
            return similar_docs
        except Exception as e:
            print(f"❌ Error searching: {str(e)}")
            return []

    def get_index_stats(self) -> Dict:
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.embedding_dim
        }

    def delete_all(self) -> bool:
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.docs = []
            if os.path.exists(self.index_path): os.remove(self.index_path)
            if os.path.exists(self.docs_path): os.remove(self.docs_path)
            print("🗑️ All FAISS vectors and docs deleted")
            return True
        except Exception as e:
            print(f"❌ Error deleting vectors: {str(e)}")
            return False

    def _create_chunk_id(self, chunk: Dict, index: int) -> str:
        url = chunk.get('url', 'unknown')
        text = chunk.get('text', '')
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{url.replace('https://', '').replace('http://', '').replace('/', '_')}_{index}_{content_hash}"


# Test run
if __name__ == "__main__":
    vs = VectorStore()

    sample_chunks = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
            'url': 'https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning?hl=en',
            'title': 'Machine Learning Basics',
            'chunk_id': 0
        },
        {
            'text': 'Deep learning uses neural networks with multiple layers to learn complex patterns.',
            'url': 'https://www.ibm.com/think/topics/deep-learning',
            'title': 'Deep Learning Guide',
            'chunk_id': 1
        }
    ]

    if vs.add_documents(sample_chunks):
        results = vs.search_similar("What is machine learning?", top_k=2)
        for r in results:
            print(f"Score: {r['score']:.3f} | Text: {r['text'][:80]}...")

    print("📊 Stats:", vs.get_index_stats())
