# import os

# # Model configurations
# OLLAMA_MODEL = "llama3.2:1b"
# EMBEDDING_MODEL = "mxbai-embed-large:latest"

# # Directory configurations
# DB_FOLDER = "db"
# PDF_FOLDER = "pdf"

# # Text splitter configurations
# CHUNK_SIZE = 512
# CHUNK_OVERLAP = 80

# # Vector store configurations
# SIMILARITY_SEARCH_K = 20
# SIMILARITY_THRESHOLD = 0.0

# # Flask configurations
# HOST = "0.0.0.0"
# PORT = 8080
# DEBUG = True

# # Create directories if they don't exist
# os.makedirs(DB_FOLDER, exist_ok=True)
# os.makedirs(PDF_FOLDER, exist_ok=True)

import os

# Model configurations
OLLAMA_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "mxbai-embed-large:latest"

# Directory configurations
DB_FOLDER = "db"
PDF_FOLDER = "pdf"

# Qdrant configurations
QDRANT_URL = "http://localhost:6333"  # URL của Qdrant server
QDRANT_COLLECTION_NAME = "pdf_documents"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Text splitter configurations
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Vector store configurations
SIMILARITY_SEARCH_K = 10
SIMILARITY_THRESHOLD = 0.0  # Giảm threshold để dễ tìm thấy kết quả hơn

# Flask configurations
HOST = "0.0.0.0"
PORT = 8080
DEBUG = True

# Create directories if they don't exist
# os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)