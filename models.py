from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import ollama
from config import (
    OLLAMA_MODEL, 
    EMBEDDING_MODEL, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP
)

class OllamaWrapper:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def chat(self, messages):
        """
        Gọi non-stream, trả về full text.
        messages: list[dict] [{"role": "user", "content": "..."}]
        """
        resp = ollama.chat(model=self.model, messages=messages)
        return resp["message"]["content"]

    def stream(self, messages):
        """
        Gọi stream, yield từng chunk text.
        """
        stream = ollama.chat(model=self.model, messages=messages, stream=True)
        for chunk in stream:
            # Debug log để xem Ollama trả về gì
            print("OLLAMA RAW:", chunk, flush=True)

            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

            elif "delta" in chunk and chunk["delta"].get("content"):
                yield chunk["delta"]["content"]

            elif chunk.get("done"):
                break

def get_llm_stream():
    return OllamaWrapper(model=OLLAMA_MODEL)

# Initialize LLM
def get_llm():
    return OllamaLLM(model=OLLAMA_MODEL)

# Initialize embeddings
def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

# Initialize text splitter
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

# Prompt template for RAG
# def get_rag_prompt():
#     return PromptTemplate.from_template(
#         """ 
#         <s>[INST] You are a technical assistant that answers strictly based on the given context. 
#         - If the answer to the user's question is in the context, answer it clearly.  
#         - If the answer is not in the context, reply exactly: "I don't know".  
#         - Do not answer unrelated questions or make assumptions. [/INST] </s>
        
#         [INST] Question: {input}
#         Context: {context}
#         Answer:
#         [/INST]
#         """
#     )


# Prompt template for idiom
def get_rag_prompt():
    return PromptTemplate.from_template(
        """
        <s>[INST] <<SYS>>
        You are a technical assistant that answers strictly based on the given context.
        The context contains a list of English idioms with their Vietnamese meanings. 
        <</SYS>>

        Rules:
        - If the user's question is in English, find the idiom in the context that best matches the question
          (it can be semantically similar, not necessarily exact).
        - If the user's question is in Vietnamese, find the idiom in the context that has that meaning or is closest semantically.
        - Return ONLY ONE idiom.
        - Output MUST be exactly one line, containing only the idiom and its meaning. 
        - Do not add any other text, explanation, or multiple answers.
        - If no idiom matches, output exactly: "No match found"
        - When answering, remove extra characters like "/", ":" around idioms.
        - Output must be exactly: idiom - Vietnamese meaning
        - If no exact match is found, return the closest idiom in meaning based on the context.
        - Even if the user's input is not exactly the same words, match the idiom or meaning that is semantically closest.
        - If the idiom text and the meaning text are exactly the same, return only one side (idiom only). 

        Question: {input}
        Context: {context}
        Answer (one line only):
        [/INST]
        """
    )

# Prompt template for retriever
def get_retriever_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", 
        """You are a strict assistant. 
        The context contains English idioms and their Vietnamese meanings. 
        Rules:
        - If the question is in English, return exactly one idiom in this format:
          idiom - Vietnamese meaning
        - If the question is in Vietnamese, return exactly one idiom in this format:
          idiom - Vietnamese meaning
        - Return ONLY ONE idiom.
        - Do not add explanations or extra text.
        - Only use the provided context.
        - If no idiom matches, reply exactly: "I don't know"."""),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # return ChatPromptTemplate.from_messages([
    #     ("system", 
    #      "You are an helpful assistant that provides answers based only on the provided context. "
    #      "Format your response striclty as JSON with keys:'answer', 'sources'."
    #      "Do not add any extra text."  ),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("human", "{input}"),
    #     ("human", "Given the conversation above, please answer the question based on the provided context."),
    # ])
def build_prompt_with_history(query: str, context_docs, history=None) -> str:
    """
    Build final RAG prompt với optional chat history (không dùng LangChain PromptTemplate).
   
    Args:
        query (str): câu hỏi của user
        context_docs (List): danh sách documents từ retriever (BM25/Vector)
        history (List[Dict] hoặc List[HumanMessage|AIMessage]): lịch sử hội thoại
    Returns:
        str: prompt hoàn chỉnh để gửi sang model (ollama.chat)
    """
    # Ghép context lại
    context_text = "\n".join([
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in context_docs
    ])
    
    # Ghép lịch sử hội thoại (nếu có)
    history_text = ""
    if history:
        # Check if history is list of LangChain messages or dict pairs
        if history and isinstance(history[0], (HumanMessage, AIMessage)):
            # Convert LangChain messages to conversation pairs
            pairs = []
            i = 0
            while i < len(history) - 1:
                if isinstance(history[i], HumanMessage) and isinstance(history[i + 1], AIMessage):
                    pairs.append({
                        "user": history[i].content,
                        "assistant": history[i + 1].content
                    })
                    i += 2
                else:
                    i += 1
            
            history_text = "\n".join([
                f"User: {h['user']}\nAssistant: {h['assistant']}"
                for h in pairs
            ])
        else:
            # Already in dict format
            history_text = "\n".join([
                f"User: {h['user']}\nAssistant: {h['assistant']}"
                for h in history
            ])
    
    # Prompt template thuần
    prompt = f"""
<s>[INST] <<SYS>>
You are a technical assistant that answers strictly based on the given context.
The context contains a list of English idioms with their Vietnamese meanings.
<</SYS>>

Rules:
- If the user's input is a greeting (e.g., "Hello", "Hi", "Xin chào"), respond with a greeting back.
- Otherwise:
  - If the user's question is in English, find the idiom in the context that best matches the question.
  - If the user's question is in Vietnamese, find the idiom in the context whose Vietnamese meaning is closest to the question.
  - Output must always contain:
      1. The English idiom
      2. Its Vietnamese meaning
      3. One example usage of the idiom in English, followed by its Vietnamese translation
  - The output format must be:
      idiom - Vietnamese meaning
      Example: <English sentence> | Ví dụ: <Vietnamese sentence>
  - If no idiom matches, output exactly: "Không tìm thấy".


Context:
{context_text}

Conversation history:
{history_text}

Question: {query}

Answer (one line only):
[/INST]""".strip()
    
    return prompt


def build_prompt_with_history_longdoc(query: str, context_docs, history=None) -> str:
    """
    Build RAG prompt với optional chat history cho document dài.
   
    Args:
        query (str): câu hỏi của user
        context_docs (List): danh sách documents từ retriever (BM25/Vector)
        history (List[Dict] hoặc List[HumanMessage|AIMessage]): lịch sử hội thoại
    Returns:
        str: prompt hoàn chỉnh để gửi sang model (ollama.chat)
    """
    # Ghép context lại
    context_text = "\n".join([
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in context_docs
    ])
    
    # Ghép lịch sử hội thoại (nếu có)
    history_text = ""
    if history:
        if isinstance(history[0], (HumanMessage, AIMessage)):
            pairs = []
            i = 0
            while i < len(history) - 1:
                if isinstance(history[i], HumanMessage) and isinstance(history[i + 1], AIMessage):
                    pairs.append({
                        "user": history[i].content,
                        "assistant": history[i + 1].content
                    })
                    i += 2
                else:
                    i += 1
            history_text = "\n".join([
                f"User: {h['user']}\nAssistant: {h['assistant']}"
                for h in pairs
            ])
        else:
            history_text = "\n".join([
                f"User: {h['user']}\nAssistant: {h['assistant']}"
                for h in history
            ])
    
    # Prompt template cho long documents
    prompt = f"""
<s>[INST] <<SYS>>
You are a helpful assistant that answers strictly based on the provided context.
The context may contain long articles, reports, or documents.
<</SYS>>

Rules:
- Use only the context below. Do not hallucinate.
- If multiple context chunks are relevant, synthesize them into a single coherent answer.
- If no relevant information is found, answer exactly: "Không tìm thấy thông tin trong tài liệu".
- Answer in the same language as the user's question.
- Keep the answer concise (2–5 sentences) unless explicitly asked for details.
- If the context has lists, tables, or numbers, preserve them in the answer if relevant.

Context:
{context_text}

Conversation history:
{history_text}

Question: {query}

Answer:
[/INST]""".strip()
    
    return prompt
