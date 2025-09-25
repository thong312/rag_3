from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Optional

class ChatHistory:
    def __init__(self):
        self.history: List[HumanMessage | AIMessage] = []
    
    def add_human_message(self, content: str):
        self.history.append(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        self.history.append(AIMessage(content=content))
    
    def clear(self):
        self.history = []
       
    def get_messages(self) -> List[HumanMessage | AIMessage]:
        return self.history
    
    def get_formatted(self) -> List[Dict]:
        formatted = []
        for i, message in enumerate(self.history):
            formatted.append({
                "type": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content,
                "index": i
            })
        return formatted
    
    # Add this method to get history in the format expected by build_prompt_with_history
    def get_conversation_pairs(self) -> List[Dict[str, str]]:
        """Convert message history to user/assistant pairs"""
        pairs = []
        i = 0
        while i < len(self.history) - 1:
            if isinstance(self.history[i], HumanMessage) and isinstance(self.history[i + 1], AIMessage):
                pairs.append({
                    "user": self.history[i].content,
                    "assistant": self.history[i + 1].content
                })
                i += 2
            else:
                i += 1
        return pairs
    
    def __len__(self):
        return len(self.history)