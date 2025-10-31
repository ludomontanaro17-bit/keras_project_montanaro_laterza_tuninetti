from huggingface_hub import hf_hub_download
import pandas as pd
import joblib

class DataLoader:
    def __init__(self):
        self.repo_id = "julien-c/wine-quality"
        self.model_file = "sklearn_model.joblib"
        self.data_file = "winequality-red.csv"

    def load_model(self):
        """Scarica e carica il modello dal repository Hugging Face."""
        model_path = hf_hub_download(repo_id=self.repo_id, filename=self.model_file)
        model = joblib.load(model_path)
        print("✅ Modello caricato con successo.")
        return model

    def load_data(self):
        """Scarica e carica il dataset come DataFrame Pandas."""
        data_path = hf_hub_download(repo_id=self.repo_id, filename=self.data_file)
        df = pd.read_csv(data_path, sep=";")
        X = df.drop(columns=["quality"])
        y = df["quality"]
        print(f"✅ Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne.")
        return X, y
