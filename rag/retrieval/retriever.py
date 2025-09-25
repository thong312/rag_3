from typing import Dict, Any
from models import get_llm, get_rag_prompt

class DocumentRetriever:
    def __init__(self):
        self.llm = get_llm()
        self.rag_prompt = get_rag_prompt()

    def get_llm_response(self, query: str, context: str) -> Dict[str, Any]:
        """Get LLM response using RAG prompt"""
        try:
            prompt = self.rag_prompt.format(
                input=query,
                context=context
            )
            response = self.llm.invoke(prompt)
            
            return response.content if hasattr(response, "content") else str(response)
            
        except Exception as e:
            return f"Error getting LLM response: {str(e)}"