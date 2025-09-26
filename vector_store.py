from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import mlflow
from langchain.schema import Document

from models import get_embeddings, get_text_splitter
from config import (
    QDRANT_HOST, 
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME
)
import os
from rag.search.bm25 import BM25Search
from storage.minio_client import MinioClient
import io
import tempfile

class VectorStoreManager:
    def __init__(self):
        self.embedding = get_embeddings()
        self.text_splitter = get_text_splitter()
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION_NAME
        self.storage = MinioClient()
        self._ensure_collection_exists()
        self.bm25_search = BM25Search() 
        self.documents = []
    
    def _ensure_collection_exists(self):
        """Tạo collection nếu chưa tồn tại"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")
                # Lấy dimension từ embedding model
                test_embedding = self.embedding.embed_query("test")
                vector_size = len(test_embedding)
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    ),
                )
                print(f"Collection created with vector size: {vector_size}")
            else:
                print(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise e
    
    def load_vector_store(self):
        """Load existing vector store"""
        print("Loading Qdrant vector store...")
        try:
            vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.collection_name,
                embedding=self.embedding,
                url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            )
            
            # Kiểm tra số lượng documents
            collection_info = self.client.get_collection(self.collection_name)
            doc_count = collection_info.points_count
            print(f"Qdrant vector store loaded with {doc_count} documents")
            
            if doc_count == 0:
                print("Warning: Vector store is empty!")
                
            return vector_store
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            # Nếu collection chưa tồn tại, tạo mới
            print("Creating new vector store...")
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding,
            )
            return vector_store
    
    # def process_pdf(self, file):
    #     """Process PDF file from upload"""
    #     try:
    #         # Upload to MinIO first
    #         filename = file.filename
    #         if not self.storage.upload_file(file, filename):
    #             raise Exception("Failed to upload file to storage")

    #         # Get file from MinIO for processing
    #         pdf_content = self.storage.get_file(filename)
    #         if not pdf_content:
    #             raise Exception("Failed to retrieve file from storage")

    #         # Create a temporary file for PDFPlumber
    #         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
    #             temp_file.write(pdf_content)
    #             temp_path = temp_file.name

    #         try:
    #             # Process with PDFPlumberLoader using temp file
    #             loader = PDFPlumberLoader(temp_path)
    #             docs = loader.load()
    #             print(f"Initial docs len={len(docs)}")

    #             if len(docs) == 0:
    #                 raise ValueError("No documents found in PDF")

    #             # Split into chunks
    #             chunks = self.text_splitter.split_documents(docs)
    #             print(f"Chunks len={len(chunks)}")

    #             if len(chunks) == 0:
    #                 raise ValueError("No chunks created from documents")

    #             # Add metadata
    #             for i, chunk in enumerate(chunks):
    #                 chunk.metadata.update({
    #                     "file_name": filename,
    #                     "chunk": i + 1,
    #                     "type": "pdf"
    #                 })

    #             # Add to vector store
    #             vector_store = self.load_vector_store()
    #             vector_store.add_documents(chunks)
    #             print(f"Added {len(chunks)} chunks to vector store")

    #             return len(docs), len(chunks)

    #         finally:
    #             # Clean up temp file
    #             os.unlink(temp_path)
    #             print("Temporary file cleaned up")

    #     except Exception as e:
    #         print(f"Error processing PDF: {e}")
    #         raise e


    def process_pdf(self, file):
        try:
            filename = file.filename
            if not self.storage.upload_file(file, filename):
                raise Exception("Failed to upload file to storage")

            pdf_content = self.storage.get_file(filename)
            if not pdf_content:
                raise Exception("Failed to retrieve file from storage")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_content)
                temp_path = temp_file.name

            try:
                reader = PdfReader(temp_path)
                docs = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        docs.append({
                            "page_content": text,
                            "metadata": {
                                "file_name": filename,
                                "page": i + 1,
                                "type": "pdf"
                            }
                        })

                print(f"Initial docs len={len(docs)}")
                if len(docs) == 0:
                    raise ValueError("No text extracted from PDF")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", " ", ""]
                )

                chunks = []
                for doc in docs:
                    content = doc["page_content"]

                    # Heuristic: bảng
                    if content.count("|") > 5 or "\t" in content:
                        for line in content.splitlines():
                            if line.strip():
                                chunks.append(
                                    Document(
                                        page_content=line.strip(),
                                        metadata={**doc["metadata"]}
                                    )
                                )
                    else:
                        split_docs = text_splitter.create_documents(
                            [content], metadatas=[doc["metadata"]]
                        )
                        chunks.extend(split_docs)

                print(f"Chunks len={len(chunks)}")
                if len(chunks) == 0:
                    raise ValueError("No chunks created from documents")

                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({"chunk": i + 1})

                vector_store = self.load_vector_store()
                vector_store.add_documents(chunks)
                print(f"Added {len(chunks)} chunks to vector store")

                return len(docs), len(chunks)

            finally:
                os.unlink(temp_path)
                print("Temporary file cleaned up")

        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise e
        
    def delete_collection(self):
        """Xóa collection (để reset dữ liệu)"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Collection {self.collection_name} deleted")
            self._ensure_collection_exists()
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_info(self):
        """Lấy thông tin về collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None
    
    def add_documents(self, documents):
        """Thêm documents vào vector store và update BM25"""
        try:
            vector_store = self.load_vector_store()
            vector_store.add_documents(documents)
            print(f"Added {len(documents)} documents to vector store")

            #  Update BM25 index bằng instance bm25_search của chính VectorStoreManager
            self.bm25_search.add_documents(documents)
            print("BM25 index updated incrementally")
            return documents
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise e

    def process_idiom(self, file, source_name="idioms"):
        """Process idiom file (PDF) with PyPDF2 and add to Qdrant vector store"""
        try:
            filename = file.filename
            if not self.storage.upload_file(file, filename):
                raise Exception("Failed to upload file to storage")

            pdf_content = self.storage.get_file(filename)
            if not pdf_content:
                raise Exception("Failed to retrieve file from storage")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_content)
                temp_path = temp_file.name

            try:
                #  Load PDF bằng PyPDF2
                reader = PdfReader(temp_path)
                processed_chunks: List[Document] = []

                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if not text:
                        continue

                    for line in text.split("\n"):
                        line = line.strip()
                        if not line or " - " not in line:
                            continue

                        idiom, rest = line.split(" - ", 1)
                        idiom_doc = Document(
                            page_content=f"{idiom.strip()} - {rest.strip()}",
                            metadata={
                                "file_name": filename,
                                "page": page_num,
                                "idiom": idiom.strip(),
                                "meaning": rest.strip(),
                                "type": "idiom",
                                "source": source_name,
                            }
                        )
                        processed_chunks.append(idiom_doc)

                if not processed_chunks:
                    raise ValueError("No idioms extracted from PDF")

                #  Add vào vector store (BM25 sẽ tự cập nhật)
                self.add_documents(processed_chunks)

                #  Lấy số lượng points cuối cùng
                collection_info = self.client.get_collection(self.collection_name)
                final_count = collection_info.points_count

                print(f" Added {len(processed_chunks)} idioms from {filename}")
                print(f"Total points in collection: {final_count}")

                return {
                    "docs": len(reader.pages),
                    "chunks": len(processed_chunks),
                    "total_points": final_count,
                }

            finally:
                os.unlink(temp_path)
                print("Temporary file cleaned up")

        except Exception as e:
            print(f"❌ Error processing idioms: {e}")
            raise e

    # def process_idiom_stream(self, file, source_name="idioms"):
    #     """Process idiom file (PDF) with PyPDF2 and stream progress"""
    #     try:
    #         filename = file.filename

    #         # đọc hết nội dung vào memory buffer (tránh giữ file gốc)
    #         file_bytes = file.read()

    #         # upload vào storage từ buffer
    #         if not self.storage.upload_file(file_bytes, filename):
    #             yield {"event": "error", "msg": "Failed to upload file to storage"}
    #             return

    #         pdf_content = self.storage.get_file(filename)
    #         if not pdf_content:
    #             yield {"event": "error", "msg": "Failed to retrieve file from storage"}
    #             return

    #         # lưu ra temp file
    #         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
    #             temp_file.write(pdf_content)
    #             temp_path = temp_file.name

    #         try:
    #             reader = PdfReader(temp_path)
    #             processed_chunks: List[Document] = []

    #             total_pages = len(reader.pages)
    #             yield {"event": "progress", "msg": f"Start processing {total_pages} pages"}

    #             for page_num, page in enumerate(reader.pages, start=1):
    #                 text = page.extract_text()
    #                 if not text:
    #                     continue

    #                 page_chunks = []
    #                 for line in text.split("\n"):
    #                     line = line.strip()
    #                     if not line or " - " not in line:
    #                         continue

    #                     idiom, rest = line.split(" - ", 1)
    #                     idiom_doc = Document(
    #                         page_content=f"{idiom.strip()} - {rest.strip()}",
    #                         metadata={
    #                             "file_name": filename,
    #                             "page": page_num,
    #                             "idiom": idiom.strip(),
    #                             "meaning": rest.strip(),
    #                             "type": "idiom",
    #                             "source": source_name,
    #                         }
    #                     )
    #                     page_chunks.append(idiom_doc)

    #                 if page_chunks:
    #                     processed_chunks.extend(page_chunks)
    #                     yield {
    #                         "event": "progress",
    #                         "msg": f"Processed page {page_num}/{total_pages}, {len(page_chunks)} idioms"
    #                     }

    #             if not processed_chunks:
    #                 yield {"event": "error", "msg": "No idioms extracted from PDF"}
    #                 return

    #             # Add vào vector store
    #             self.add_documents(processed_chunks)

    #             collection_info = self.client.get_collection(self.collection_name)
    #             final_count = collection_info.points_count

    #             yield {
    #                 "event": "done",
    #                 "docs": total_pages,
    #                 "chunks": len(processed_chunks),
    #                 "total_points": final_count,
    #             }

    #         finally:
    #             os.unlink(temp_path)

    #     except Exception as e:
    #         yield {"event": "error", "msg": str(e)}

