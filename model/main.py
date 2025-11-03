from classes.Dataloader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from classes.Model import WineQualityNeuralNet
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocessing import DataPreprocessing  # Assicurati che il file ecc. avvenga nel contesto giusto

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
















