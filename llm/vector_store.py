from pinecone import Pinecone, ServerlessSpec
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import pickle
import os
import time
import logging
from document_processor import DocumentChunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        hybrid_weight: float = 0.7,
        bm25_k: int = 3,
        pinecone_api_key: str = os.getenv("PINECONE_API_KEY"),
        index_name: str = os.getenv("PINECONE_INDEX_NAME"),
        project_id: str = os.getenv("PINECONE_PROJECT_ID"),
        region: str = os.getenv("PINECONE_REGION"),
        namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
        
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.hybrid_weight = hybrid_weight
        self.bm25_k = bm25_k
        self.encoder = SentenceTransformer(model_name)
        self.chunk_map = {}
        self.texts = []
        self.chunk_ids = []
        self.bm25_index = None
        self.sparse_vectors = None

        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(
            name=index_name,
            project_id=project_id,
            region=region
        )

        self.namespace = namespace

        logging.info(f"[Pinecone Init] Using namespace: {self.namespace}")


    def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 100):
        if not chunks:
            return
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_texts = [chunk.content for chunk in batch]
            start_time = time.time()
            embeddings = self.encoder.encode(batch_texts, show_progress_bar=False, batch_size=32, convert_to_numpy=True)
            embeddings = embeddings.astype('float32')
            vectors = []
            for j, chunk in enumerate(batch):
                chunk_id = chunk.chunk_id
                vectors.append({
                    "id": chunk_id,
                    "values": embeddings[j].tolist(),
                    "metadata": {
                        "text": batch[j].content,
                        "source": batch[j].source,
                        "section": batch[j].section or "",
                        "page_number": batch[j].page_number or -1
                    }
                })

                self.chunk_map[chunk_id] = chunk
                self.texts.append(chunk.content)
                self.chunk_ids.append(chunk_id)
            resp=self.index.upsert(vectors=vectors, namespace=self.namespace)
            logging.info(f"[Upsert response] {resp}")
            elapsed_time = time.time() - start_time
            logging.info(f"Added batch {i//batch_size+1}/{(len(chunks)//batch_size)+1} ({len(batch)} chunks, {elapsed_time:.2f}s)")
        self._build_sparse_index()

    def _build_sparse_index(self):
        if not self.texts:
            return
        self.bm25_index = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50000)
        self.sparse_vectors = self.bm25_index.fit_transform(self.texts)

    def search(self, query: str, k: int = 5, threshold: float = 0.3, filter_func: Optional[callable] = None) -> List[Tuple[DocumentChunk, float]]:
        logging.info(f"Starting search for query: {query}")
        if not self.chunk_map:
            return []
        
        results = []
        dense_results = self._dense_search(query, k * 3, threshold)
        sparse_results = self._sparse_search(query, self.bm25_k, threshold) if self.bm25_index else []
        
        # The combined results are already sorted by relevance and truncated to k.
        all_results = self._combine_results(dense_results, sparse_results, k)
        
        # The threshold has already been applied in dense/sparse search.
        # Do not filter again on the RRF score.
        for chunk, score in all_results:
            if filter_func and not filter_func(chunk):
                continue
            results.append((chunk, score))
            # The slice [:k] is already handled in _combine_results
            
        return results
    def _dense_search(self, query: str, k: int, threshold: float):
        query_embedding = self.encoder.encode([query], show_progress_bar=False).astype('float32')[0].tolist()
        response = self.index.query(vector=query_embedding, top_k=k, include_metadata=True, namespace=self.namespace)
        results = []
        for match in response.get('matches',[]):
            chunk_id = match['id']
            score = match['score']
            if score < threshold:
                continue
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                results.append((chunk, float(score)))
        return results

    def _sparse_search(self, query: str, k: int, threshold: float):
        query_vec = self.bm25_index.transform([query])
        cos_sim = cosine_similarity(query_vec, self.sparse_vectors).flatten()
        top_indices = np.argsort(cos_sim)[::-1][:k]
        results = []
        for idx in top_indices:
            score = cos_sim[idx]
            if score < threshold:
                continue
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                results.append((chunk, float(score)))
        return results

    def _combine_results(self, dense_results, sparse_results, k: int):
        dense_ranks = {chunk.chunk_id: 1/(rank + 60) for rank, (chunk, _) in enumerate(dense_results)}
        sparse_ranks = {chunk.chunk_id: 1/(rank + 60) for rank, (chunk, _) in enumerate(sparse_results)}
        combined_scores = {}
        for chunk_id in set(dense_ranks.keys()).union(sparse_ranks.keys()):
            combined_score = dense_ranks.get(chunk_id, 0) * self.hybrid_weight + sparse_ranks.get(chunk_id, 0) * (1 - self.hybrid_weight)
            combined_scores[chunk_id] = combined_score
        sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)
        return [(self.chunk_map[chunk_id], combined_scores[chunk_id]) for chunk_id in sorted_ids[:k] if chunk_id in self.chunk_map]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(f"{filepath}.state", 'wb') as f:
            pickle.dump({
                'chunk_map': self.chunk_map,
                'texts': self.texts,
                'chunk_ids': self.chunk_ids,
                'model_name': self.model_name,
                'dimension': self.dimension,
                'hybrid_weight': self.hybrid_weight,
                'bm25_k': self.bm25_k
            }, f)
        if self.bm25_index:
            with open(f"{filepath}.bm25", 'wb') as f:
                pickle.dump(self.bm25_index, f)
        logging.info(f"Saved vector store metadata to {filepath}")

    def load(self, filepath: str):
        try:
            with open(f"{filepath}.state", 'rb') as f:
                state = pickle.load(f)
                self.chunk_map = state['chunk_map']
                self.texts = state['texts']
                self.chunk_ids = state['chunk_ids']
                self.model_name = state['model_name']
                self.dimension = state['dimension']
                self.hybrid_weight = state.get('hybrid_weight', 0.7)
                self.bm25_k = state.get('bm25_k', 3)
            if os.path.exists(f"{filepath}.bm25"):
                with open(f"{filepath}.bm25", 'rb') as f:
                    self.bm25_index = pickle.load(f)
                if self.texts:
                    self.sparse_vectors = self.bm25_index.transform(self.texts)
            logging.info(f"Loaded vector store metadata from {filepath} [Chunks: {len(self.chunk_map)}]")
            return True
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            self.chunk_map = {}
            self.texts = []
            self.chunk_ids = []
            self.bm25_index = None
            return False

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_chunks": len(self.chunk_map),
            "model": self.model_name,
            "dimension": self.dimension,
            "hybrid_weight": self.hybrid_weight,
            "sparse_index": bool(self.bm25_index)
        }
        try:
            logging.info(f"[Pinecone Init] Using namespace: {self.namespace}")
            stats_raw = self.index.describe_index_stats()
            logging.info(f"[Full stats dump] {stats_raw}")
            ns_stats = stats_raw.get("namespaces", {}).get(self.namespace, {})
            vector_count = ns_stats.get("vector_count", stats_raw.get("total_vector_count", 0))
            stats.update({
                "index_type": "pinecone",
                "index_size": vector_count
            })
        except Exception:
            pass
        if self.chunk_map:
            sources = set()
            sections = set()
            for chunk in self.chunk_map.values():
                sources.add(chunk.source)
                if chunk.section:
                    sections.add(chunk.section)
            stats.update({
                "unique_sources": len(sources),
                "unique_sections": len(sections),
                "avg_text_length": sum(len(chunk.content) for chunk in self.chunk_map.values()) / len(self.chunk_map)
            })
        return stats

    def filter_search(self, query: str, k: int = 5, source: Optional[str] = None, section: Optional[str] = None, min_page: Optional[int] = None, max_page: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        def filter_func(chunk):
            if source and chunk.source != source:
                return False
            if section and chunk.section != section:
                return False
            if min_page is not None and chunk.page_number < min_page:
                return False
            if max_page is not None and chunk.page_number > max_page:
                return False
            return True
        return self.search(query, k, filter_func=filter_func)

    def clear(self):
    # Attempt to delete the remote namespace
        try:
            stats = self.index.describe_index_stats()
            if self.namespace in stats.get("namespaces", {}):
                self.index.delete(delete_all=True, namespace=self.namespace)
                logging.info(f"Cleared Pinecone namespace: {self.namespace}")
            else:
                logging.warning(f"Namespace '{self.namespace}' does not exist in Pinecone stats; skipping remote delete.")
        except Exception as e:
            logging.error(f"Error clearing Pinecone namespace: {e}")

        # Always clear the local state
        self.chunk_map = {}
        self.texts = []
        self.chunk_ids = []
        self.bm25_index = None
        self.sparse_vectors = None
        logging.info("Cleared local vector store state.")
