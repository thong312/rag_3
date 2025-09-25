from typing import Dict, Any
from .service import ChatService

def create_chat_service() -> ChatService:
    """Create a new ChatService instance"""
    return ChatService()

def quick_chat(query: str, use_rag: bool = True) -> Dict[str, Any]:
    """Quick chat function"""
    service = create_chat_service()
    return service.chat_with_history(query, use_rag=use_rag)