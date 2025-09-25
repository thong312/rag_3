from .handler import RAGHandler

def create_hybrid_rag():
    """Create a new RAGHandler instance"""
    return RAGHandler()

def quick_hybrid_query(query, alpha=0.5, k=None):
    """Quick hybrid RAG query"""
    rag = create_hybrid_rag()
    return rag.rag_query_hybrid(query, k=k, alpha=alpha)

__all__ = ['RAGHandler', 'create_hybrid_rag', 'quick_hybrid_query']