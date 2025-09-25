from typing import List, Dict, Any

class ContextFormatter:
    def format_documents(self, documents: List[Any]) -> str:
        """Format documents thành context cho LLM"""
        if not documents:
            return "Không tìm thấy thông tin liên quan."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            file_name = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page", 0)
            context_parts.append(
                f"[Nguồn {i}: {file_name}, trang {page}]\n{doc.page_content}"
            )
        
        return "\n\n".join(context_parts)

    # def extract_sources(self, documents: List[Any]) -> List[Dict]:
    #     """Extract thông tin nguồn từ documents"""
    #     return [
    #         {
    #             "source": doc.metadata.get("source", "unknown"),
    #             "file_name": doc.metadata.get("file_name", "unknown"), 
    #             "page": doc.metadata.get("page", 0),
    #             "chunk": doc.metadata.get("chunk", 0)
    #         }
    #         for doc in documents
    #     ]
    def extract_sources(self, documents):
    # Trả về list string
        return [
            f"{doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', '?')})"
            for doc in documents
        ]