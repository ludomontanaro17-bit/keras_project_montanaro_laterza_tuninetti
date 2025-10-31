# src/api.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# Percorsi modello e preprocessore
MODEL_PATH = "models/wine_model.keras"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

# Carica modello e preprocessore all'avvio
if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    raise RuntimeError("❌ Modello o preprocessore non trovati. Assicurati che 'models/' contenga entrambi.")

model = load_model(MODEL_PATH)
preprocessor_meta = joblib.load(PREPROCESSOR_PATH)
scaler = preprocessor_meta["scaler"]
selected_features = preprocessor_meta["selected_features"]
num_classes = preprocessor_meta["num_classes"]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint per predire la qualità del vino.
    
    Input JSON:
    {
      "volatile acidity": 0.5,
      "citric acid": 0.2,
      "chlorides": 0.08,
      "total sulfur dioxide": 45,
      "density": 0.996,
      "pH": 3.2,
      "sulphates": 0.6,
      "alcohol": 10.5
    }
    
    Output:
    {
      "predicted_class": 1,
      "predicted_label": "medio",
      "probabilities": [0.1, 0.8, 0.1]
    }
    """
    try:
        # Leggi input JSON
        data = request.get_json()
        if not data:
            return jsonify({"error": "Richiesta JSON vuota"}), 400

        # Converti in DataFrame (ordine non importa, ma devono esserci tutte le feature)
        try:
            df = pd.DataFrame([data])
        except Exception as e:
            return jsonify({"error": f"Formato input non valido: {str(e)}"}), 400

        # Verifica che abbia tutte le feature necessarie
        missing = set(selected_features) - set(df.columns)
        if missing:
            return jsonify({"error": f"Feature mancanti: {list(missing)}"}), 400

        # Seleziona e ordina le feature
        X = df[selected_features]

        # Applica scaling
        X_scaled = scaler.transform(X)

        # Predizione
        probs = model.predict(X_scaled, verbose=0)[0]
        predicted_class = int(np.argmax(probs))

        # Mappa classe → etichetta (solo se 3 classi)
        if num_classes == 3:
            label_map = {0: "basso", 1: "medio", 2: "alto"}
            label = label_map.get(predicted_class, str(predicted_class))
        else:
            label = str(predicted_class + 3)  # mappa 0→3, 1→4, ..., 5→8

        return jsonify({
            "predicted_class": predicted_class,
            "predicted_label": label,
            "probabilities": probs.tolist()
        })

    except Exception as e:
        return jsonify({"error": f"Errore interno: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Endpoint per verificare che l'API sia attiva."""
    return jsonify({"status": "ok", "model_loaded": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)