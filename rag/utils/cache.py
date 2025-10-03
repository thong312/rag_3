# import os
# import pickle
# from typing import Any, Optional

# class CacheManager:
#     def __init__(self, cache_dir: str = "."):
#         self.cache_dir = cache_dir
#         self.bm25_cache_path = os.path.join(cache_dir, "bm25_cache.pkl")
#         self.docs_cache_path = os.path.join(cache_dir, "docs_cache.pkl")

#     def save_bm25_cache(self, bm25_model, documents):
#         try:
#             with open(self.bm25_cache_path, 'wb') as f:
#                 pickle.dump(bm25_model, f)
#             with open(self.docs_cache_path, 'wb') as f:
#                 pickle.dump(documents, f)
#             return True
#         except Exception as e:
#             print(f"Error saving BM25 cache: {e}")
#             return False

#     def load_bm25_cache(self) -> tuple[Optional[Any], Optional[list]]:
#         try:
#             if os.path.exists(self.bm25_cache_path) and os.path.exists(self.docs_cache_path):
#                 with open(self.bm25_cache_path, 'rb') as f:
#                     bm25 = pickle.load(f)
#                 with open(self.docs_cache_path, 'rb') as f:
#                     documents = pickle.load(f)
#                 return bm25, documents
#         except Exception as e:
#             print(f"Error loading BM25 cache: {e}")
#         return None, None

#     def clear_cache(self) -> bool:
#         """Clear all cached data"""
#         try:
#             if os.path.exists(self.bm25_cache_path):
#                 os.remove(self.bm25_cache_path)
#             if os.path.exists(self.docs_cache_path):
#                 os.remove(self.docs_cache_path)
#             print("Cache cleared successfully")
#             return True
#         except Exception as e:
#             print(f"Error clearing cache: {e}")
#             return False
        


import pickle
import redis
from typing import Any, Optional

class CacheManager:
    def __init__(self, host="localhost", port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def save_bm25_cache(self, bm25_model, documents) -> bool:
        try:
            self.client.set("bm25_model", pickle.dumps(bm25_model))
            self.client.set("bm25_docs", pickle.dumps(documents))
            return True
        except Exception as e:
            print(f"Error saving BM25 cache to Redis: {e}")
            return False

    def load_bm25_cache(self) -> tuple[Optional[Any], Optional[list]]:
        try:
            bm25_data = self.client.get("bm25_model")
            docs_data = self.client.get("bm25_docs")
            if bm25_data and docs_data:
                bm25 = pickle.loads(bm25_data)
                documents = pickle.loads(docs_data)
                return bm25, documents
        except Exception as e:
            print(f"Error loading BM25 cache from Redis: {e}")
        return None, None

    def clear_cache(self) -> bool:
        try:
            self.client.delete("bm25_model")
            self.client.delete("bm25_docs")
            print("Redis cache cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing Redis cache: {e}")
            return False
