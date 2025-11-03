'''
────────────────────────────────────────────────────────────
📘 DESCRIZIONE DEL FLUSSO COMPLETO DEL PROGETTO
────────────────────────────────────────────────────────────

Questo progetto è diviso in due fasi principali:
1️⃣ TRAINING (eseguito una tantum)
2️⃣ SERVING (eseguito in produzione per predire nuovi vini)

────────────────────────────────────────────────────────────
🔹 FASE 1 — TRAINING (main.py)
────────────────────────────────────────────────────────────
- Si carica il dataset dei vini e si applica il preprocessing.
- I dati vengono standardizzati (StandardScaler) e divisi in train/validation.
- Si addestra il modello Keras (WineQualityNeuralNet).
- Al termine, il modello e lo scaler vengono salvati nella cartella /model:
    - model/wine_quality_model.keras → il modello addestrato
    - model/scaler.pkl               → lo scaler usato per la normalizzazione
  Questi file rappresentano la pipeline "congelata" del sistema.

👉 Questa fase si esegue manualmente solo quando si vuole
   addestrare o aggiornare il modello.

────────────────────────────────────────────────────────────
🔹 FASE 2 — SERVING (Flask API)
────────────────────────────────────────────────────────────
- Si avvia l’API Flask (es. app/app.py o api.py).
- All’avvio, l’API carica automaticamente:
    - il modello Keras salvato,
    - lo scaler salvato.
- Rimane in ascolto su una porta (default: 5000) e attende richieste.

Quando arriva una richiesta POST /predict:
    - legge i dati del nuovo vino (JSON),
    - li trasforma con lo stesso scaler usato nel training,
    - li passa al modello per ottenere la predizione,
    - restituisce la classe di qualità prevista e le probabilità.

👉 Questa fase è continua: serve per fare previsioni “live”
   senza dover riaddestrare ogni volta.

────────────────────────────────────────────────────────────
🔁 FASE 3 — RETRAINING (facoltativa)
────────────────────────────────────────────────────────────
Se arrivano nuovi dati o si vuole migliorare la performance:
    - si riesegue main.py,
    - vengono generati nuovi file .keras e .pkl,
    - si riavvia Flask per usare il nuovo modello.

────────────────────────────────────────────────────────────
💡 In sintesi:
- main.py  → crea e salva il modello addestrato (offline)
- app.py   → lo carica e lo rende disponibile via API (online)
────────────────────────────────────────────────────────────
'''




from WineAPI import WineAPI

if __name__ == "__main__":
    api = WineAPI(
        model_path="../model/wine_quality_model.keras",
        scaler_path="../model/scaler.pkl"
    )
    api.run()


