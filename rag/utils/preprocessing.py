import re
import unicodedata
from typing import List

def preprocess_text(text: str) -> List[str]:
    """Tiền xử lý text cho BM25"""
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return tokens