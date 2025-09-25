from minio import Minio
from minio.error import S3Error
from typing import Optional, BinaryIO
import io
import os

class MinioClient:
    def __init__(self):
        self.client = Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False  # Set to True if using HTTPS
        )
        self.bucket_name = "pdfs"
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if it doesn't"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created")
            else:
                print(f"Bucket '{self.bucket_name}' already exists")
        except S3Error as e:
            print(f"Error checking/creating bucket: {e}")
            raise

    def upload_file(self, file_obj: BinaryIO, filename: str) -> bool:
        """Upload a file to MinIO"""
        try:
            # Get file size and reset file pointer
            file_obj.seek(0, os.SEEK_END)
            file_size = file_obj.tell()
            file_obj.seek(0)

            # Upload file
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=filename,
                data=file_obj,
                length=file_size,
                content_type='application/pdf'
            )
            print(f"File {filename} uploaded successfully")
            return True
        except S3Error as e:
            print(f"Error uploading file: {e}")
            return False

    def get_file(self, filename: str) -> Optional[bytes]:
        """Get a file from MinIO"""
        try:
            response = self.client.get_object(
                bucket_name=self.bucket_name,
                object_name=filename
            )
            return response.read()
        except S3Error as e:
            print(f"Error retrieving file: {e}")
            return None

    def delete_file(self, filename: str) -> bool:
        """Delete a file from MinIO"""
        try:
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=filename
            )
            print(f"File {filename} deleted successfully")
            return True
        except S3Error as e:
            print(f"Error deleting file: {e}")
            return False