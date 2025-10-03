import json
import os
from datetime import datetime


class DatasetLogger:
    """
    Lưu dataset dưới dạng JSONL: mỗi dòng = {prompt, response, metadata}
    """

    def __init__(self, file_path="chat_dataset.jsonl"):
        self.file_path = file_path
        # Create file if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                pass

    def log(self, prompt: str, response: str, model: str = "unknown"):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "prompt": prompt,
            "response": response,
        }
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
