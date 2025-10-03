from flask import Blueprint, jsonify, request
from chat.service import ChatService
from vector_store import VectorStoreManager
from config import PDF_FOLDER
import os


# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize services
chat_service = ChatService()
vector_manager = VectorStoreManager()

@api_bp.route("/ai", methods=["POST"])
def ai_post():
    """Simple AI chat endpoint"""
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query", "")
    
    response = chat_service.simple_chat(query)
    return {"answer": response}

# @api_bp.route("/ask_pdf", methods=["POST"])
# def ask_pdf_post():
#     """Ask questions about uploaded PDFs"""
#     print("Post /ask_pdf called")
#     json_content = request.json
#     query = json_content.get("query", "")
    
#     result = chat_service.rag_chat(query)
#     return result
@api_bp.route("/idioms", methods=["POST"])
def idioms_post():
    """Upload and process idioms PDF file"""
    file = request.files.get("file")
    if not file:
        return {"error": "No file part in the request. Make sure to send 'file' as form-data."}, 400

    try:
        # Process idioms directly from file object (MinIO handling is done inside process_idiom)
        source_name = request.form.get("source_name", "idioms")
        docs_len, chunks_len, final_count = vector_manager.process_idiom(
            file,  # Pass file object directly instead of save_file
            source_name=source_name
        )

        return {
            "status": "Successfully Uploaded",
            "filename": file.filename,
            "source": source_name,
            "docs": docs_len,
            "chunks": chunks_len,
            "final_count": final_count,
        }

    except Exception as e:
        return {"error": f"Failed to process idioms: {str(e)}"}, 500

@api_bp.route("/pdf", methods=["POST"])
def pdf_post():
    """Upload and process PDF file"""
    file = request.files.get("file")
    if not file:
        return {"error": "No file part in the request"}, 400

    try:
        # Process PDF directly from file object
        doc_len, chunks_len = vector_manager.process_pdf(file)
        
        return {
            "status": "Successfully Uploaded",
            "filename": file.filename,
            "doc_len": doc_len,
            "chunks": chunks_len,
        }
    except Exception as e:
        return {"error": f"Failed to process PDF: {str(e)}"}, 500

@api_bp.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear chat history"""
    chat_service.clear_history()
    return {"status": "Chat history cleared"}

@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@api_bp.route("/debug/vectorstore", methods=["GET"])
def debug_vectorstore():
    """Debug vector store information"""
    try:
        # Lấy thông tin collection
        collection_info = vector_manager.get_collection_info()
        if not collection_info:
            return {"status": "error", "error": "Cannot get collection info"}, 500

        # Lấy sample metadata từ Qdrant
        points, _ = vector_manager.client.scroll(
            collection_name=vector_manager.collection_name,
            limit=5
        )
        sample_payloads = [p.payload for p in points]

        return {
            "status": "success",
            "collection_info": collection_info,  #  Trả thẳng dict luôn
            "sample_metadata": sample_payloads,
            "message": "Vector store is ready" if collection_info["points_count"] > 0 else "Vector store is empty"
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}, 500



@api_bp.route("/debug/search", methods=["POST"])
def debug_search():
    """Debug search functionality"""
    json_content = request.json
    query = json_content.get("query", "test")
    
    try:
        vector_store = vector_manager.load_vector_store()
        retriever = vector_manager.get_retriever(vector_store)
        
        results = retriever.get_relevant_documents(query)
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "content_preview": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "content_length": len(doc.page_content)
                }
                for doc in results[:3]  # Chỉ trả về 3 kết quả đầu
            ]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}, 500

@api_bp.route("/debug/reset", methods=["POST"])
def reset_collection():
    """Reset collection (xóa và tạo lại)"""
    try:
        vector_manager.delete_collection()
        return {"status": "success", "message": "Collection reset successfully"}
    except Exception as e:
        return {"status": "error", "error": str(e)}, 500
    

# @api_bp.route("/ask_pdf_hybrid", methods=["POST"])
# def ask_pdf_hybrid():
#     """Using hybrid search to answer questions"""
#     print("Post /ask_pdf_hybrid called")
#     data = request.get_json()
    
#     query = data.get("query", "")
#     k = data.get("k", None)
#     alpha = data.get("alpha", 0.5)
#     metadata_filter = data.get("metadata_filter", None)
#     use_rerank = data.get("use_rerank", True)

#     result = chat_service.hybrid_chat(
#         query=query,
#         k=k,
#         alpha=alpha,
#         metadata_filter=metadata_filter,
#         use_rerank=use_rerank
#     )
#     return result

@api_bp.route("/chat", methods=["POST"])
def chat():
    """Unified chat endpoint with search method selection"""
    data = request.get_json()
    query = data.get("query", "")
    search_type = data.get("search_type", "hybrid")  # default to hybrid
    
    result = chat_service.chat_with_history(
        query=query,
        search_type=search_type,
        k=data.get("k"),
        alpha=data.get("alpha", 0.5),
        metadata_filter=data.get("metadata_filter"),
        use_rerank=data.get("use_rerank", True)
    )
    return result

# @api_bp.route("/clear_indexes", methods=["POST"])
# def clear_indexes():
#     """Clear search indexes and caches"""
#     result = chat_service.rag_handler.clear_search_indexes()
#     return result



@api_bp.route("/clear_indexes", methods=["POST"])
def clear_indexes():
    """Clear search indexes and caches"""
    success = chat_service.rag_handler.clear_search_indexes()
    if success:
        return jsonify({"status": "success", "message": "Indexes and caches cleared"}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to clear indexes/caches"}), 500
 