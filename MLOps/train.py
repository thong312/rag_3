from typing import List
import mlflow
import mlflow.pyfunc
import pandas as pd
import time
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.callbacks.base import BaseCallbackHandler

from MLOps.dataset_logger import DatasetLogger

from tabulate import tabulate

# ---------------------------
# MLflow Tracker
# ---------------------------
# class MLflowTracker:
#     def __init__(self, experiment_name="chatbot_training", tracking_uri="http://localhost:5000"):
#         mlflow.set_tracking_uri(tracking_uri)
#         mlflow.set_experiment(experiment_name)
        
#     def start_run(self, run_name=None):
#         return mlflow.start_run(run_name=run_name)

#     def log_param(self, key, value):
#         mlflow.log_param(key, value)

#     def log_params(self, params: dict, print_out=False):
#         mlflow.log_params(params)
#         if print_out:
#             table = [(k, v) for k, v in params.items()]
#             print("\nParameters logged:")
#             print(tabulate(table, headers=["Parameter", "Value"], tablefmt="github"))

#     def log_metric(self, key, value, step=None):
#         mlflow.log_metric(key, value, step=step)

#     def log_metrics(self, metrics: dict, step=None):
#         mlflow.log_metrics(metrics, step=step)

#     def log_artifact(self, local_path, artifact_path=None):
#         mlflow.log_artifact(local_path, artifact_path=artifact_path)

#     def log_artifacts(self, local_dir, artifact_path=None):
#         mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

#     def log_dict(self, dictionary, file_name, artifact_path=None):
#         mlflow.log_dict(dictionary, file_name, artifact_path=artifact_path)

#     def log_table(self, data, file_name: str): mlflow.log_table(data, file_name)
class MLflowTracker:
    def __init__(self, experiment_name="chatbot_training", tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name=None):
       return mlflow.start_run(run_name=run_name)

    def log_param(self, key, value):
        mlflow.log_param(key, value)
    def log_params(self, params: dict, print_out=False):
        mlflow.log_params(params)
        if print_out:
            table = [(k, v) for k, v in params.items()]
            print("\nParameters logged:")
            print(tabulate(table, headers=["Parameter", "Value"], tablefmt="github"))
    def log_metrics(self, metrics: dict, step=None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path, artifact_path=None):
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_dict(self, dictionary, file_name, artifact_path=None):
        mlflow.log_dict(dictionary, file_name, artifact_path=artifact_path)

    def log_table(self, data, file_name: str): mlflow.log_table(data, file_name)    
# ---------------------------
# Callback để log prompt/response
# ---------------------------
class MLflowCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracker: MLflowTracker, dataset_logger: DatasetLogger, model: str):
        self.tracker = tracker
        self.dataset_logger = dataset_logger
        self.model = model
        self.step = 0
        self.last_prompt = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        if prompts:
            self.last_prompt = prompts[0]

    def on_llm_end(self, response, **kwargs):
        try:
            if response and response.generations:
                self.step += 1
                output = response.generations[0][0].text

                if self.last_prompt:
                    self.dataset_logger.log(self.last_prompt, output, model=self.model)

                mlflow_data = {
                    "step": self.step,
                    "prompt": self.last_prompt,
                    "response": output,
                }
                self.tracker.log_dict(
                    mlflow_data,
                    file_name=f"llm_response_step{self.step}.json",
                    artifact_path="llm_outputs",
                )
        except Exception as e:
            print(f"[MLflowCallbackHandler] Failed to log response: {e}")


# ---------------------------
# Ollama Pyfunc Wrapper
# ---------------------------
class OllamaPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name: str, embedding_name: str):
        self.model_name = model_name
        self.embedding = embedding_name
    def load_context(self, context):
        self.llm = OllamaLLM(model=self.model_name)
        self.embeddings = OllamaEmbeddings(model=self.embedding)
    def predict(self, context, model_input: List[str]) -> List[str]:
        """
        model_input: List[str] → danh sách các prompt
        return: List[str] → danh sách các response từ Ollama
        """
        if not isinstance(model_input, list):
            raise TypeError("model_input must be a list of strings")
        outputs: List[str] = []
        for query in model_input:
            outputs.append(self.llm.invoke(query))
        return outputs


# ---------------------------
# Main Training Function
# ---------------------------
# def train_and_track():
#     from vector_store import VectorStoreManager

#     tracker = MLflowTracker(experiment_name="chatbot_training")
#     dataset_logger = DatasetLogger("chat_dataset.jsonl")

#     model_name = "qwen2.5:3b"

#     try:
#         with tracker.start_run(run_name=f"training_{int(time.time())}"):

#             # --- Init components ---
#             llm = OllamaLLM(
#                 model=model_name,
#                 callbacks=[MLflowCallbackHandler(tracker, dataset_logger, model_name)],
#             )
#             embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
#             vector_store = VectorStoreManager()

#             # --- Log params + metrics ---
#             tracker.log_param("llm_model", model_name, print_out=True)

#             params = {
#                 "llm_model": model_name,
#                 "embedding_model": "mxbai-embed-large:latest",
#             }
#             tracker.log_params(params, print_out=True)
            
#             tracker.log_metric("training_loss", 0.15, step=1)

#             # --- Run some test prompts ---
#             questions = [
#                 "Xin chào, bạn là ai?",
#                 "MLflow dùng để làm gì?",
#                 "Vector database hoạt động như thế nào?",
#             ]
#             for q in questions:
#                 response = llm.invoke(q)
#                 print(f"\nQ: {q}\nA: {response}\n")

#             # --- Log dataset vào MLflow ---
#             import mlflow.data
#             df = pd.read_json("chat_dataset.jsonl", lines=True)
#             dataset = mlflow.data.from_pandas(
#                 df, source="chat_dataset.jsonl", name="chat_dataset"
#             )
#             mlflow.log_input(dataset, context="training")

#             # Log raw dataset file vào artifact
#             tracker.log_artifact("chat_dataset.jsonl", artifact_path="datasets")

#             # --- Log Ollama model wrapper ---
#             mlflow.pyfunc.log_model(
#                 artifact_path="ollama_model",
#                 python_model=OllamaPyfunc(model_name),
#                 registered_model_name="chatbot_ollama_model",
#             )

#             print("✅ Training completed, dataset + Ollama model wrapper logged to MLflow")

#     except Exception as e:
#         print(f"❌ Error during training: {e}")
#         raise
def train_and_track():
    tracker = MLflowTracker(experiment_name="chatbot_training")
    dataset_logger = DatasetLogger("chat_dataset.jsonl")
    model_name = "qwen2.5:3b"
    with tracker.start_run():
        # Log dataset to MLflow
        import mlflow.data
        df = pd.read_json("chat_dataset.jsonl", lines=True)
        dataset = mlflow.data.from_pandas(
            df, source="chat_dataset.jsonl", name="chat_dataset"
        )
        mlflow.log_input(dataset, context="training")

        # Log raw dataset file as artifact
        tracker.log_artifact("chat_dataset.jsonl", artifact_path="chat_dataset.json")

        # Log Ollama model wrapper
        mlflow.pyfunc.log_model(
            artifact_path="ollama_model",
            python_model=OllamaPyfunc(model_name),
            registered_model_name="chatbot_ollama_model",
        )

if __name__ == "__main__":
    train_and_track()
