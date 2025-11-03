from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os

class WineAPI:
    def __init__(self, model_path="model/wine_quality_model.keras", scaler_path="model/scaler.pkl"):
        """Inizializza il server Flask e carica modello, scaler e feature attese."""
        self.app = Flask(__name__)
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

        # 🔹 Le feature effettive usate nel training (ordine corretto)
        self.expected_features = [
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "total sulfur dioxide",
            "sulphates",
            "alcohol"
        ]

        # Caricamento automatico all'avvio
        self._load_model()
        self._load_scaler()

        # Registrazione endpoints
        self._register_routes()

    # ====== METODI INTERNI ======
    def _load_model(self):
        """Carica il modello Keras."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Modello non trovato in: {self.model_path}")
        self.model = load_model(self.model_path)
        print("✅ Modello Keras caricato con successo")

    def _load_scaler(self):
        """Carica lo scaler salvato con joblib."""
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Scaler caricato con successo")
        else:
            print("⚠️ Nessuno scaler trovato — gli input devono essere già normalizzati")

    def _register_routes(self):
        """Definisce gli endpoint dell’API."""

        @self.app.route("/", methods=["GET"])
        def home():
            return jsonify({
                "message": "API Keras per classificazione vino pronta 🚀",
                "expected_features": self.expected_features
            })

        @self.app.route("/predict", methods=["POST"])
        def predict():
            data = request.get_json()

            if not data or "features" not in data:
                return jsonify({
                    "error": "JSON non valido. Usa 'features': [valori...]",
                    "expected_features": self.expected_features
                }), 400

            features = np.array(data["features"]).reshape(1, -1)

            # ✅ Controllo numero di feature
            if features.shape[1] != len(self.expected_features):
                return jsonify({
                    "error": f"Numero di feature errato: atteso {len(self.expected_features)}, ricevuto {features.shape[1]}",
                    "expected_order": self.expected_features
                }), 400

            # Applica scaler se disponibile
            if self.scaler:
                features = self.scaler.transform(features)

            # Predizione
            prediction = self.model.predict(features)
            predicted_class = int(np.argmax(prediction, axis=1)[0])

            return jsonify({
                "predicted_class": predicted_class,
                "probabilities": prediction.tolist()[0],
                "expected_features": self.expected_features
            })

    # ====== AVVIO SERVER ======
    def run(self, host="0.0.0.0", port=5000, debug=True):
        self.app.run(host=host, port=port, debug=debug)




if __name__ == "__main__":
    api = WineAPI()
    api.run(host="0.0.0.0", port=5000)
