from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
from dataclasses import dataclass

from rag.retrieval.reranker import CrossEncoderReranker
from rag.search.bm25 import BM25Search
from rag.search.hybrid import HybridSearch
from rag.search.vector import VectorSearch


from .retrieval.retriever import DocumentRetriever
from .utils.context import ContextFormatter
from config import SIMILARITY_SEARCH_K

# State class để lưu trữ trạng thái giữa các node
@dataclass 
class RAGState:
    query: str
    chat_history: List
    documents: List = None
    reranked_docs: List = None
    context: str = None
    answer: str = None
    sources: List = None

class RAGGraph:
    def __init__(self):
        # Initialize components
        self.vector_search = VectorSearch()
        self.bm25_search = BM25Search()
        self.hybrid_search = HybridSearch(self.bm25_search, self.vector_search)
        self.reranker = CrossEncoderReranker() 
        self.context_formatter = ContextFormatter()
        self.retriever = DocumentRetriever()
        
        # Create workflow
        self.workflow = StateGraph(RAGState)
        
        # Add nodes
        self.workflow.add_node("search", self.search_node)
        self.workflow.add_node("rerank", self.rerank_node)
        self.workflow.add_node("format_context", self.format_context_node)
        self.workflow.add_node("generate", self.generate_answer_node)

        # Add edges
        self.workflow.set_entry_point("search")
        self.workflow.add_edge("search", "rerank")
        self.workflow.add_edge("rerank", "format_context")
        self.workflow.add_edge("format_context", "generate") 
        self.workflow.add_edge("generate", END)

        self.graph = self.workflow.compile()

    def search_node(self, state: RAGState):
        """Node thực hiện hybrid search"""
        try:
            candidates = self.hybrid_search.search(
                query=state.query,
                k=SIMILARITY_SEARCH_K * 2
            )
            state.documents = [doc for doc, _ in candidates]
            return state
        except Exception as e:
            return {"error": str(e)}

    def rerank_node(self, state: RAGState):
        """Node thực hiện rerank documents"""
        try:
            if state.documents:
                state.reranked_docs = self.reranker.rerank(
                    state.query,
                    state.documents, 
                    top_k=SIMILARITY_SEARCH_K
                )
            else:
                state.reranked_docs = state.documents[:SIMILARITY_SEARCH_K]
            return state
        except Exception as e:
            return {"error": str(e)}

    def format_context_node(self, state: RAGState):
        """Node format context từ documents"""
        try:
            state.context = self.context_formatter.format_documents(state.reranked_docs)
            return state
        except Exception as e:
            return {"error": str(e)}

    def generate_answer_node(self, state: RAGState):
        """Node sinh câu trả lời từ LLM"""
        try:
            response = self.retriever.get_llm_response(state.query, state.context)
            state.answer = response
            state.sources = self.context_formatter.extract_sources(state.reranked_docs)
            return state
        except Exception as e:
            return {"error": str(e)}

    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        state = RAGState(
            query=query,
            chat_history=[]
        )
        
        try:
            result = self.graph.invoke(state)
            
            return {
                "answer": result.answer,
                "sources": result.sources
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "answer": None,
                "sources": []
            }