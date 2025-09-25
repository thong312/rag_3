from typing import List, Tuple, Dict, Any, Optional
from .bm25 import BM25Search
from .vector import VectorSearch

class HybridSearch:
    def __init__(self, bm25_search: BM25Search, vector_search: VectorSearch):
        self.bm25_search = bm25_search
        self.vector_search = vector_search

    def search(self,
              query: str,
              k: int = 10,
              alpha: float = 0.5,
              metadata_filter: Optional[Dict] = None,
              bm25_k: Optional[int] = None,
              vector_k: Optional[int] = None) -> List[Tuple[Any, float]]:
        """
        Hybrid search combining BM25 and vector search
        Args:
            query: Search query
            k: Final number of results 
            alpha: Weight for vector search (0.0 = only BM25, 1.0 = only vector)
            metadata_filter: Optional metadata filter
            bm25_k: Number of results from BM25 (default: k*2)
            vector_k: Number of results from vector search (default: k*2)
        """
        # Default values
        if bm25_k is None:
            bm25_k = max(k * 2, 20)
        if vector_k is None:
            vector_k = max(k * 2, 20)

        try:
            # Get results from both methods
            bm25_results = self.bm25_search.search(query, bm25_k, metadata_filter)
            vector_results = self.vector_search.search(query, vector_k, metadata_filter)

            print(f"BM25 found {len(bm25_results)} results")
            print(f"Vector found {len(vector_results)} results")

            # Normalize scores
            combined_scores = self._combine_scores(bm25_results, vector_results, alpha)

            # Sort by hybrid score
            final_results = [(info['doc'], info['hybrid_score']) 
                           for info in combined_scores.values()]
            final_results.sort(key=lambda x: x[1], reverse=True)

            print(f"Hybrid search returning top {min(k, len(final_results))} results")
            return final_results[:k]

        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def _combine_scores(self, bm25_results, vector_results, alpha):
        """Combine and normalize BM25 and vector scores"""
        combined_scores = {}

        # Normalize BM25 scores
        bm25_scores = [score for _, score in bm25_results]
        if bm25_scores:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            if max_bm25 > min_bm25:
                normalized_bm25 = [(doc, (score - min_bm25) / (max_bm25 - min_bm25))
                                 for doc, score in bm25_results]
            else:
                normalized_bm25 = bm25_results
        else:
            normalized_bm25 = []

        # Add normalized scores to combined dict
        for doc, score in normalized_bm25:
            doc_key = (doc.page_content, str(doc.metadata))
            combined_scores[doc_key] = {
                'doc': doc,
                'bm25_score': score,
                'vector_score': 0.0
            }

        # Add vector scores
        for doc, score in vector_results:
            doc_key = (doc.page_content, str(doc.metadata))
            if doc_key in combined_scores:
                combined_scores[doc_key]['vector_score'] = score
            else:
                combined_scores[doc_key] = {
                    'doc': doc,
                    'bm25_score': 0.0,
                    'vector_score': score
                }

        # Calculate hybrid scores
        for info in combined_scores.values():
            info['hybrid_score'] = (1 - alpha) * info['bm25_score'] + alpha * info['vector_score']

        return combined_scores