from typing import Dict, Any, Optional
from .search.bm25 import BM25Search
from .search.vector import VectorSearch
from .search.hybrid import HybridSearch
from .retrieval.reranker import CrossEncoderReranker
from .retrieval.retriever import DocumentRetriever
from .utils.context import ContextFormatter
from config import SIMILARITY_SEARCH_K

class RAGHandler:
    def __init__(self):
        self.vector_search = VectorSearch()
        self.bm25_search = BM25Search()
        self.hybrid_search = HybridSearch(self.bm25_search, self.vector_search)
        self.reranker = CrossEncoderReranker()
        self.retriever = DocumentRetriever()
        self.context_formatter = ContextFormatter()
        self._initialize_indexes()

    def _initialize_indexes(self):
        """Initialize search indexes"""
        try:
            # Get all documents from vector store
            documents = self.vector_search.get_all_documents()
            if documents:
                print(f"Found {len(documents)} documents in vector store")
                # Build BM25 index
                self.bm25_search.build_index(documents)
            else:
                print("No documents found in vector store")
        except Exception as e:
            print(f"Error initializing indexes: {e}")

    def update_indexes(self, documents=None):
        """Update BM25 index incrementally (nếu có documents mới)"""
        try:
            if documents:
                print(f"Incrementally adding {len(documents)} new docs to BM25 index...")
                self.bm25_search.add_documents(documents)
            else:
                print("Rebuilding BM25 index with all docs...")
                all_docs = self.vector_search.get_all_documents()
                self.bm25_search.build_index(all_docs)
            return True
        except Exception as e:
            print(f"Error updating indexes: {e}")
            return False


    def rag_query_hybrid(self, query: str, k: Optional[int] = None, 
                        alpha: float = 0.5, include_sources: bool = True,
                        metadata_filter: Optional[Dict] = None, 
                        use_rerank: bool = True) -> Dict[str, Any]:
        """RAG pipeline with hybrid search"""
        k = k or SIMILARITY_SEARCH_K

        try:
            # Get candidate documents
            candidates = self.hybrid_search.search(
                query=query,
                k=k * 2,
                alpha=alpha,
                metadata_filter=metadata_filter
            )
            documents = [doc for doc, _ in candidates]

            # Rerank if needed
            if use_rerank:
                documents = self.reranker.rerank(query, documents, top_k=k)
            else:
                documents = documents[:k]

            # Format context and get response
            context = self.context_formatter.format_documents(documents)
            response = self.retriever.get_llm_response(query, context)
            
            return {
                "answer": response,
                "sources": self.context_formatter.extract_sources(documents)
            }

        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": []
            }

    def clear_search_indexes(self):
        """Clear all search indexes and caches"""
        try:
            self.bm25_search.clear_index()
            status = self.bm25_search.get_status()
            return {
                "status": "success", 
                "message": "Search indexes cleared",
                "bm25_status": status
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}