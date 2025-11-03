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
    - model/scaler.pkl → lo scaler usato per la normalizzazione
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
- main.py → crea e salva il modello addestrato (offline)
- app.py → lo carica e lo rende disponibile via API (online)
────────────────────────────────────────────────────────────
'''


import numpy as np
from classes.Dataloader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from classes.Model import WineQualityNeuralNet
from classes.Datapreprocessing import DataPreprocessing

if __name__ == "__main__":

    ### STEP 1: Caricamento dati dal repository Hugging Face ###
    data_loader = DataLoader()
    X, y = data_loader.load_data()

    # Stampa le prime righe dei dati caricati
    print("\nPrime 5 righe del dataset caricato:")
    print(X.head())
    print("\nPrime 5 etichette del target:")
    print(y.head())

    ### STEP 2: Analisi esplorativa dataset ###
    
    # Combina X e y in un DataFrame
    df = X.copy()
    df['quality'] = y  # aggiungi la colonna target

    # Inizializza la classe di preprocessing
    preprocessing = DataPreprocessing(df, target_column='quality')
    
    # Mostra informazioni generali e statistiche sul dataset
    preprocessing.display_info()
    preprocessing.display_statistics()
    preprocessing.display_missing_values()

    ### STEP 2.2: Visualizzazioni grafiche distribuzioni ###
    preprocessing.plot_quality_distribution()

    ### STEP 2.3: Stampiamo la matrice di correlazione ###
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matrice di Correlazione delle Caratteristiche del Vino')
    plt.show()

    ### STEP 3 DATA PREPROCESSING ###
    # Standardizza le caratteristiche numeriche
    preprocessing.standardize_numeric_features()

    ### Rimozione delle variabili collineari ###
    X_reduced = preprocessing.remove_collinear_features()

    # Suddividi il dataset in training e validation set
    X_train, X_val, y_train, y_val = preprocessing.split_data()


    ### STEP 4 - Creazione e addestramento del modello di rete neurale ###
    model = WineQualityNeuralNet(num_classes=len(np.unique(y_train)))
    model.build(input_dim=X_train.shape[1])

    # Allena il modello
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # Predizioni sul validation set
    y_pred = model.predict_classes(X_val)

    # Report di classificazione
    print("\n📊 Report di classificazione:")
















