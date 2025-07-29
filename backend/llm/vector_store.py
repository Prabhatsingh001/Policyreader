import faiss
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    """Enhanced FAISS-based vector store with hybrid search capabilities"""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        dimension: int = 384,
        hybrid_weight: float = 0.7,
        bm25_k: int = 3
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.hybrid_weight = hybrid_weight  # Weight for dense vector vs sparse search
        self.bm25_k = bm25_k  # Top BM25 results to consider
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.chunk_map = {}  # chunk_id -> DocumentChunk
        self.texts = []  # For sparse retrieval
        self.chunk_ids = []  # Parallel to texts
        self.bm25_index = None
        
    def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """Add document chunks to the vector store with batching"""
        if not chunks:
            return
            
        # Initialize FAISS index if not exists
        if self.index is None:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_texts = [chunk.content for chunk in batch]
            
            # Create embeddings
            start_time = time.time()
            embeddings = self.encoder.encode(
                batch_texts, 
                show_progress_bar=False, 
                batch_size=32,
                convert_to_numpy=True
            )
            embeddings = embeddings.astype('float32')
            faiss.normalize_L2(embeddings)  # For cosine similarity
            
            # Generate IDs
            ids = np.array([i for i in range(len(self.chunk_ids), len(self.chunk_ids) + len(batch))])
            
            # Add to index
            self.index.add_with_ids(embeddings, ids)
            
            # Update metadata and text stores
            for j, chunk in enumerate(batch):
                self.chunk_map[chunk.chunk_id] = chunk
                self.texts.append(chunk.content)
                self.chunk_ids.append(chunk.chunk_id)
            
            batch_num = i//batch_size+1
            total_batches = (len(chunks)//batch_size)+1
            elapsed_time = time.time()-start_time
            logging.info(f"Added batch {batch_num}/{total_batches} ({len(batch)} chunks, {elapsed_time:.2f}s)")
        
        # Build sparse index after adding all documents
        self._build_sparse_index()
    
    def _build_sparse_index(self):
        """Build TF-IDF index for sparse retrieval"""
        if not self.texts:
            return
            
        self.bm25_index = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=50000
        )
        self.sparse_vectors = self.bm25_index.fit_transform(self.texts)
    
    def search(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = 0.3,
        filter_func: Optional[callable] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Hybrid search combining dense and sparse methods
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity score
            filter_func: Function to filter chunks (chunk -> bool)
        """
        if self.index is None or not self.chunk_map:
            return []
        
        results = []
        
        # 1. Dense vector search
        dense_results = self._dense_search(query, k * 3, threshold)
        
        # 2. Sparse search (if index exists)
        sparse_results = []
        if self.bm25_index:
            sparse_results = self._sparse_search(query, self.bm25_k, threshold)
        
        # 3. Combine results using reciprocal rank fusion
        all_results = self._combine_results(dense_results, sparse_results, k)
        
        # 4. Apply filtering and threshold
        for chunk, score in all_results:
            if score < threshold:
                continue
            if filter_func and not filter_func(chunk):
                continue
            results.append((chunk, score))
            if len(results) >= k:
                break
        
        return results
    
    def _dense_search(self, query: str, k: int, threshold: float):
        """Dense vector search using FAISS"""
        query_embedding = self.encoder.encode([query], show_progress_bar=False)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
            if score < threshold:
                continue
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                results.append((chunk, float(score)))
        return results
    
    def _sparse_search(self, query: str, k: int, threshold: float):
        """Sparse search using TF-IDF"""
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
        """Combine results using reciprocal rank fusion (RRF)"""
        # Create rank dictionaries
        dense_ranks = {chunk.chunk_id: (1/(rank + 60)) for rank, (chunk, _) in enumerate(dense_results)}
        sparse_ranks = {chunk.chunk_id: (1/(rank + 60)) for rank, (chunk, _) in enumerate(sparse_results)}
        
        # Combine scores
        combined_scores = {}
        for chunk_id in set(list(dense_ranks.keys()) + list(sparse_ranks.keys())):
            combined_scores[chunk_id] = (
                dense_ranks.get(chunk_id, 0) * self.hybrid_weight +
                sparse_ranks.get(chunk_id, 0) * (1 - self.hybrid_weight)
            )
        
        # Sort by combined score
        sorted_chunk_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Return top k chunks with scores
        results = []
        for chunk_id in sorted_chunk_ids[:k]:
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                results.append((chunk, combined_scores[chunk_id]))
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk efficiently"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save metadata and other state
        state = {
            'chunk_map': self.chunk_map,
            'texts': self.texts,
            'chunk_ids': self.chunk_ids,
            'model_name': self.model_name,
            'dimension': self.dimension,
            'hybrid_weight': self.hybrid_weight,
            'bm25_k': self.bm25_k
        }
        
        with open(f"{filepath}.state", 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save sparse index separately
        if self.bm25_index:
            with open(f"{filepath}.bm25", 'wb') as f:
                pickle.dump(self.bm25_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logging.info(f"Saved vector store to {filepath} [Index: {self.index.ntotal} chunks]")
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load state
            with open(f"{filepath}.state", 'rb') as f:
                state = pickle.load(f)
                self.chunk_map = state['chunk_map']
                self.texts = state['texts']
                self.chunk_ids = state['chunk_ids']
                self.model_name = state['model_name']
                self.dimension = state['dimension']
                self.hybrid_weight = state.get('hybrid_weight', 0.7)
                self.bm25_k = state.get('bm25_k', 3)
            
            # Load sparse index if exists
            bm25_path = f"{filepath}.bm25"
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                # Rebuild sparse vectors
                if self.texts:
                    self.sparse_vectors = self.bm25_index.transform(self.texts)
            
            logging.info(f"Loaded vector store from {filepath} [Chunks: {len(self.chunk_map)}]")
            return True
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            # Reset state on failure
            self.index = None
            self.chunk_map = {}
            self.texts = []
            self.chunk_ids = []
            self.bm25_index = None
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the vector store"""
        stats = {
            "total_chunks": len(self.chunk_map),
            "model": self.model_name,
            "dimension": self.dimension,
            "hybrid_weight": self.hybrid_weight,
            "sparse_index": bool(self.bm25_index)
        }
        
        if self.index:
            stats.update({
                "index_size": self.index.ntotal,
                "index_type": str(self.index)
            })
        
        # Add metadata statistics
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
    
    def filter_search(
        self, 
        query: str, 
        k: int = 5,
        source: Optional[str] = None,
        section: Optional[str] = None,
        min_page: Optional[int] = None,
        max_page: Optional[int] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search with filtering options"""
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
        """Clear the vector store"""
        self.index = None
        self.chunk_map = {}
        self.texts = []
        self.chunk_ids = []
        self.bm25_index = None
        logging.info("Vector store cleared")

# Example usage
if __name__ == "__main__":
    # Initialize with custom parameters
    vector_store = VectorStore(
        model_name="all-mpnet-base-v2",
        hybrid_weight=0.6,
        bm25_k=5
    )
    
    # Example chunks (in real usage, from DocumentProcessor)
    chunks = [
        DocumentChunk(
            content="Insurance policies cover various types of risks",
            source="policy_doc",
            chunk_id="chunk1",
            page_number=1,
            section="Introduction"
        ),
        DocumentChunk(
            content="Claim process requires proper documentation",
            source="claim_guide",
            chunk_id="chunk2",
            page_number=3,
            section="Procedures"
        )
    ]
    
    # Add documents
    vector_store.add_documents(chunks)
    
    # Perform search
    results = vector_store.search(
        query="How to file an insurance claim?",
        k=3,
        threshold=0.25
    )
    
    # Display results
    print("\nSearch Results:")
    for chunk, score in results:
        print(f"[Score: {score:.3f}] [{chunk.source} - {chunk.section}]")
        print(f"{chunk.content[:100]}...\n")
    
    # Save and load
    vector_store.save("my_vector_store")
    vector_store.load("my_vector_store")
    
    # Get statistics
    print("Vector Store Stats:", vector_store.get_statistics())