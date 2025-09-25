from flask import Blueprint, Response, request, stream_with_context
import json

from chat.service import ChatService
from vector_store import VectorStoreManager

vector_manager = VectorStoreManager()
stream_bp = Blueprint("stream", __name__)
chat_service = ChatService()

def sse_format(data: dict):
    """Format data as SSE event"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@stream_bp.route("/chat_stream", methods=["POST"])
def chat_stream():
    """Streaming chat endpoint"""

    data = request.get_json()
    query = data.get("query", "")
    search_type = data.get("search_type", "hybrid")
    k = data.get("k")
    alpha = data.get("alpha", 0.5)
    metadata_filter = data.get("metadata_filter")
    use_rerank = data.get("use_rerank", True)

    def generate():
        try:
            # Gửi event bắt đầu
            yield sse_format({"event": "start", "msg": "stream_start"})

            # ChatService cần có hàm stream
            for chunk in chat_service.chat_with_history_stream(
                query=query,
                search_type=search_type,
                k=k,
                alpha=alpha,
                metadata_filter=metadata_filter,
                use_rerank=use_rerank
            ):
                print(f"[DEBUG] stream chunk: {chunk}", flush=True)
                yield sse_format({"text": chunk})

            # Gửi event kết thúc
            yield sse_format({"event": "end", "msg": "stream_end"})
        except Exception as e:
            yield sse_format({"event": "error", "msg": str(e)})
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

# @stream_bp.route("/idioms_stream", methods=["POST"])
# def idioms_stream():
#     file = request.files.get("file")
#     if not file:
#         return Response(
#             sse_format({"event": "error", "msg": "No file provided"}),
#             mimetype="text/event-stream"
#         )

#     def generate():
#         try:
#             yield sse_format({"event": "start", "msg": "Upload started"})

#             source_name = request.form.get("source_name", "idioms")

#             # forward từng event từ process_idiom_stream
#             for event in vector_manager.process_idiom_stream(file, source_name=source_name):
#                 yield sse_format(event)

#         except Exception as e:
#             yield sse_format({"event": "error", "msg": str(e)})

