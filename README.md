Esempio presentazione: https://www.genspark.ai/agents?id=b4832924-8a9d-4b54-9dfe-d11e2cf55209
https://ludovica-montanaro.gitbook.io/esercitazione-con-keras/


# 🍷 Wine Quality Classifier

Un progetto di **Machine Learning** per predire la qualità del vino rosso utilizzando una **rete neurale Keras** e un'API **Flask** per servire le predizioni in tempo reale.

---

## 📘 Struttura del progetto
```  
keras_project_montanaro_laterza_tuninetti/
├── classes/
│ ├── Dataloader.py
│ ├── Datapreprocessing.py
│ ├── Model.py
│ └── init.py
│
├── model/
│ ├── main.py
│ ├── wine_quality_model.keras
│ └── scaler.pkl
│
├── app/
│ ├── app.py
│ └── wine_api.py
│
├── data/
│ └── winequality-red.csv
│
└── README.md
```  


---


## ⚙️ Fasi principali

### 1️⃣ Addestramento del modello
Il file `model/main.py` gestisce l’intera pipeline:
- Carica e pulisce il dataset.
- Applica la standardizzazione tramite `StandardScaler`.
- Divide i dati in **training** e **validation set**.
- Allena la rete neurale Keras.
- Salva:
  - il modello (`wine_quality_model.keras`)
  - lo scaler (`scaler.pkl`)

```bash
python model/main.py
```

---


## 2️⃣ Avvio dell'API Flask

L'API carica automaticamente il modello e lo scaler salvati, quindi espone due endpoint:

**GET /** → verifica che l'API sia attiva

**POST /predict** → riceve un nuovo vino e restituisce la classe prevista

```bash
python app/app.py
```

---


## 🧠 Esempio di richiesta
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]}'
```

## ✅ Esempio di risposta
```
json
{
  "predicted_class": 1,
  "probabilities": [0.2, 0.7, 0.1]
}
```

---


## 🧩 Requisiti
Assicurati di avere Python ≥ 3.10 e di installare le dipendenze:

```bash
pip install -r requirements.txt

```
