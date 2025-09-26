import mlflow
import mlflow.pyfunc
import pandas as pd
import time
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.callbacks.base import BaseCallbackHandler

from MLOps.dataset_logger import DatasetLogger


# ---------------------------
# MLflow Tracker
# ---------------------------
class MLflowTracker:
    def __init__(self, experiment_name="chatbot_training", tracking_uri="mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name)

    # ---------------- Params ----------------
    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    # ---------------- Metrics ----------------
    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step=None):
        mlflow.log_metrics(metrics, step=step)

    # ---------------- Artifacts ----------------
    def log_artifact(self, local_path, artifact_path=None):
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    # ---------------- Dict / Dataset ----------------
    def log_dict(self, dictionary, file_name, artifact_path=None):
        mlflow.log_dict(dictionary, file_name, artifact_path=artifact_path)

    def log_table(self, data, file_name: str):
        import mlflow
        """
        data: pandas.DataFrame
        """
        mlflow.log_table(data, file_name)


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

                # 👉 Lưu dataset thật
                if self.last_prompt:
                    self.dataset_logger.log(self.last_prompt, output, model=self.model)

                # 👉 Log vào MLflow artifact
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
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_context(self, context):
        self.llm = OllamaLLM(model=self.model_name)

    def predict(self, context, model_input):
        """
        model_input: list[str] (prompt)
        return: list[str] (response từ Ollama)
        """
        outputs = []
        for query in model_input:
            outputs.append(self.llm.invoke(query))
        return outputs


# ---------------------------
# Main Training Function
# ---------------------------
def train_and_track():
    # ✅ Import cục bộ để tránh circular import
    from vector_store import VectorStoreManager

    tracker = MLflowTracker(experiment_name="chatbot_training")
    dataset_logger = DatasetLogger("chat_dataset.jsonl")

    model_name = "qwen2.5:3b"

    try:
        with tracker.start_run(run_name=f"training_{int(time.time())}"):

            llm = OllamaLLM(
                model=model_name,
                callbacks=[MLflowCallbackHandler(tracker, dataset_logger, model_name)],
            )
            embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
            vector_store = VectorStoreManager()

            # log params
            tracker.log_param("llm_model", model_name)
            tracker.log_param("embedding_model", "mxbai-embed-large:latest")

            # giả sử metric training
            tracker.log_metric("training_loss", 0.15, step=1)

            # 👉 chạy nhiều prompt thực tế
            questions = [
                "Xin chào, bạn là ai?",
                "MLflow dùng để làm gì?",
                "Vector database hoạt động như thế nào?",
            ]
            for q in questions:
                response = llm.invoke(q)
                print(f"\nQ: {q}\nA: {response}\n")

            # ---------------------------
            # ✅ Log dataset theo chuẩn MLflow
            # ---------------------------
            import mlflow.data

            df = pd.read_json("chat_dataset.jsonl", lines=True)
            dataset = mlflow.data.from_pandas(df, source="chat_dataset.jsonl", name="chat_dataset")
            mlflow.log_input(dataset, context="training")

            # ---------------------------
            # ✅ Log Ollama model bằng Pyfunc
            # ---------------------------
            mlflow.pyfunc.log_model(
                artifact_path="ollama_model",
                python_model=OllamaPyfunc(model_name),
                registered_model_name="chatbot_ollama_model",
            )

            print("✅ Training completed, dataset + Ollama model logged to MLflow")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    train_and_track()
