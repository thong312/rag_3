from rank_bm25 import BM25Okapi
from ..utils.preprocessing import preprocess_text
from ..utils.cache import CacheManager
from typing import List, Tuple, Dict, Any, Optional
import os

class BM25Search:
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        self.bm25 = None
        self.documents = []
        # self._initialize_index()

    def _initialize_index(self):
        """Initialize BM25 index from cache or prepare for new build"""
        try:
            print("Initializing BM25 index...")
            self.bm25, self.documents = self.cache_manager.load_bm25_cache()
            
            if self.bm25 and self.documents:
                print(f"✓ BM25 index loaded from cache with {len(self.documents)} documents")
            else:
                print("! No valid cache found - will build new index when documents are added")
                
        except Exception as e:
            print(f"! Error loading BM25 cache: {str(e)}")
            self.bm25 = None
            self.documents = []

    def build_index(self, documents: List[Any]):
        """Build BM25 index from documents"""
        try:
            if not documents:
                print("! Warning: Empty documents list provided")
                return

            print(f"Building BM25 index with {len(documents)} documents...")
            self.documents = documents
            
            # Tokenize and build index
            tokenized_docs = [preprocess_text(doc.page_content) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            
            # Try to cache
            cache_success = self.cache_manager.save_bm25_cache(self.bm25, self.documents)
            if cache_success:
                print("✓ BM25 index built and cached successfully")
            else:
                print("! Warning: Index built but caching failed")

        except Exception as e:
            print(f"! Error building BM25 index: {str(e)}")
            self.bm25 = None
            self.documents = []
            raise e

    def search(self, query: str, k: int = 10, 
              metadata_filter: Optional[Dict] = None) -> List[Tuple[Any, float]]:
        """BM25 search with detailed logging"""
        try:
            # print(f"\nBM25 Search:")
            print(f"Query: {query}")
            print(f"Index status: {'Available' if self.bm25 else 'Not initialized'}")
            print(f"Documents: {len(self.documents)}")

            if not self.bm25 or not self.documents:
                print("! Error: BM25 index not initialized")
                return []

            # Tokenize and search
            tokenized_query = preprocess_text(query)
            print(f"Tokenized query: {tokenized_query}")
            
            scores = self.bm25.get_scores(tokenized_query)
            print(f"Got scores for {len(scores)} documents")
            
            # Apply scoring and filtering
            scored_docs = []
            for i, score in enumerate(scores):
                if i < len(self.documents):
                    doc = self.documents[i]
                    if self._matches_filter(doc, metadata_filter):
                        scored_docs.append((doc, float(score)))
            
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            results = scored_docs[:k]
            
            print(f"✓ Returning {len(results)} results")
            return results

        except Exception as e:
            print(f"! Error during BM25 search: {str(e)}")
            return []

    def _matches_filter(self, doc: Any, metadata_filter: Optional[Dict]) -> bool:
        """Check if document matches metadata filter"""
        if not metadata_filter:
            return True
        return all(doc.metadata.get(k) == v for k, v in metadata_filter.items())

    # def clear_index(self):
    #     """Clear BM25 index and cache"""
    #     try:
    #         print("Clearing BM25 index...")
    #         self.bm25 = None
    #         self.documents = []
            
    #         if self.cache_manager:
    #             cache_cleared = self.cache_manager.clear_cache()
    #             if cache_cleared:
    #                 print("✓ BM25 index and cache cleared successfully")
    #             else:
    #                 print("! Warning: Failed to clear cache files")
            
    #     except Exception as e:
    #         print(f"! Error clearing index: {str(e)}")
    def clear_cache(self) -> bool:
        try:
            self.client.delete("bm25_model")
            self.client.delete("bm25_docs")
            print("Redis cache cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing Redis cache: {e}")
            return False


    def get_status(self) -> Dict[str, Any]:
        """Get current status of BM25 index"""
        return {
            "initialized": self.bm25 is not None,
            "document_count": len(self.documents),
            "cache_files_exist": (
                os.path.exists(self.cache_manager.bm25_cache_path),
                os.path.exists(self.cache_manager.docs_cache_path)
            )
        }
    def add_documents(self, documents):
        self.documents.extend(documents)
        tokenized = [doc.page_content.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        # Lưu lại cache để lần sau load không bị mất
        self.cache_manager.save_bm25_cache(self.bm25, self.documents)