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

    # Suddividi il dataset in training e validation set
    X_train, X_val, y_train, y_val = preprocessing.split_data()

    ### Rimozione delle variabili collineari ###
    # Calcola la matrice di correlazione (Pearson)
    corr = df.corr(numeric_only=True)
    threshold = 0.6
    to_drop = set()
    cols = X.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = corr.loc[cols[i], cols[j]]
            if abs(corr_val) > threshold:
                corr_with_y_i = abs(corr.loc[cols[i], 'quality'])
                corr_with_y_j = abs(corr.loc[cols[j], 'quality'])
                if corr_with_y_i < corr_with_y_j:
                    to_drop.add(cols[i])
                else:
                    to_drop.add(cols[j])
                print(f"⚠️  {cols[i]} e {cols[j]} sono correlate (ρ = {corr_val:.2f}). "
                      f"Rimuovo '{cols[i] if corr_with_y_i < corr_with_y_j else cols[j]}' "
                      f"perché meno correlata con 'quality'.")

    # Mostra il risultato
    print("\n📉 Variabili da rimuovere per collinearità:")
    print(sorted(to_drop))

    # Crea una versione ridotta di X
    X_reduced = X.drop(columns=list(to_drop))
    df_pair = X_reduced.copy()
    df_pair['quality'] = y

    print("\n📊 Variabili rimaste dopo la rimozione:")
    print(list(X_reduced.columns))

    print("\n📈 Pair Plot delle caratteristiche rimanenti:")
    sns.pairplot(df_pair, hue='quality', palette='viridis')
    plt.title('Pair Plot delle Caratteristiche del Vino')
    plt.show()

    ### STEP 4 - Creazione e addestramento del modello di rete neurale ###
    model = WineQualityNeuralNet(num_classes=len(np.unique(y_train)))
    model.build(input_dim=X_train.shape[1])

    # Allena il modello
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # Predizioni sul validation set
    y_pred = model.predict_classes(X_val)

    # Report di classificazione
    print("\n📊 Report di classificazione:")
















