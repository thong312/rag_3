from typing import List, Any
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Any], top_k: int = 10) -> List[Any]:
        """
        Rerank documents using CrossEncoder
        Args:
            query: user query
            docs: list of documents (langchain Document objects) 
            top_k: number of documents to return after reranking
        """
        if not docs:
            return []

        try:
            # Prepare data for CrossEncoder: [(query, doc_text), ...]
            pairs = [(query, doc.page_content) for doc in docs]

            # Calculate relevance scores
            scores = self.reranker.predict(pairs)

            # Combine scores with docs
            scored = list(zip(docs, scores))

            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Return top_k docs
            reranked_docs = [doc for doc, score in scored[:top_k]]

            print(f"Reranking complete: selected {len(reranked_docs)} docs")
            return reranked_docs

        except Exception as e:
            print(f"Error in reranking: {e}")
            return docs[:top_k]
        


# from typing import List, Union, Tuple
# from sentence_transformers import CrossEncoder
# import torch

# class CrossEncoderReranker:
#     def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
#                  device: str | None = None, batch_size: int = 32):
#         # device: 'cuda' / 'cpu' / None (tự phát hiện)
#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.reranker = CrossEncoder(model_name, device=device)
#         self.batch_size = batch_size

#     def _doc_text(self, doc: Union[str, object]) -> str:
#         if isinstance(doc, str):
#             return doc
#         # try common attribute names
#         for attr in ("page_content", "content", "text"):
#             if hasattr(doc, attr):
#                 return getattr(doc, attr)
#         raise AttributeError("Document must be str or have page_content/content/text attribute")

#     def rerank(self, query: str, docs: List[object], top_k: int = 10) -> List[Tuple[object, float]]:
#         if not docs:
#             return []

#         # Optionally limit number of docs to rerank to avoid OOM
#         # docs = docs[:max_rerank_n]

#         pairs = [(query, self._doc_text(doc)) for doc in docs]
#         # predict supports batch_size, show_progress_bar, convert_to_numpy
#         scores = self.reranker.predict(pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)

#         scored = list(zip(docs, [float(s) for s in scores]))
#         scored.sort(key=lambda x: x[1], reverse=True)
#         top = scored[:top_k]

#         # optionally attach score to metadata if doc supports it
#         for doc, score in top:
#             if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
#                 doc.metadata["rerank_score"] = float(score)

#         return top  # list of (doc, score)
