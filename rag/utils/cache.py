import os
import pickle
from typing import Any, Optional

class CacheManager:
    def __init__(self, cache_dir: str = "."):
        self.cache_dir = cache_dir
        self.bm25_cache_path = os.path.join(cache_dir, "bm25_cache.pkl")
        self.docs_cache_path = os.path.join(cache_dir, "docs_cache.pkl")

    def save_bm25_cache(self, bm25_model, documents):
        try:
            with open(self.bm25_cache_path, 'wb') as f:
                pickle.dump(bm25_model, f)
            with open(self.docs_cache_path, 'wb') as f:
                pickle.dump(documents, f)
            return True
        except Exception as e:
            print(f"Error saving BM25 cache: {e}")
            return False

    def load_bm25_cache(self) -> tuple[Optional[Any], Optional[list]]:
        try:
            if os.path.exists(self.bm25_cache_path) and os.path.exists(self.docs_cache_path):
                with open(self.bm25_cache_path, 'rb') as f:
                    bm25 = pickle.load(f)
                with open(self.docs_cache_path, 'rb') as f:
                    documents = pickle.load(f)
                return bm25, documents
        except Exception as e:
            print(f"Error loading BM25 cache: {e}")
        return None, None

    def clear_cache(self) -> bool:
        """Clear all cached data"""
        try:
            if os.path.exists(self.bm25_cache_path):
                os.remove(self.bm25_cache_path)
            if os.path.exists(self.docs_cache_path):
                os.remove(self.docs_cache_path)
            print("Cache cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False