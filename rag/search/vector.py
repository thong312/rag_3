from typing import List, Tuple, Dict, Any, Optional
from vector_store import VectorStoreManager

class VectorSearch:
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.vector_store = None
        self._initialize_store()
        self.documents = [] 
    def _initialize_store(self):
        """Initialize vector store"""
        try:
            self.vector_store = self.vector_manager.load_vector_store()
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self.vector_store = None

    def search(self, query: str, k: int = 10, 
              metadata_filter: Optional[Dict] = None) -> List[Tuple[Any, float]]:
        """Vector semantic search"""
        if not self.vector_store:
            return []
            
        try:
            if metadata_filter:
                results = self.vector_store.similarity_search_with_score(
                    query, k=k, filter=metadata_filter
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            return [(doc, float(score)) for doc, score in results]
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []

    def get_all_documents(self) -> List[Any]:
        """Get all documents from vector store"""
        if not self.vector_store:
            return []
            
        try:
            collection_info = self.vector_manager.get_collection_info()
            if collection_info and collection_info.get('points_count', 0) > 0:
                return self.vector_store.similarity_search("", k=collection_info['points_count'])
        except Exception as e:
            print(f"Error getting documents: {e}")
        return []