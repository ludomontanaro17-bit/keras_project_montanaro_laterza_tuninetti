from classes.Dataloader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

### STEP 1: Caricamento dati dal repository Hugging Face ###
    # Inizializza il DataLoader
    data_loader = DataLoader()

    # Carica i dati
    X, y = data_loader.load_data()

    # Stampa le prime righe dei dati caricati
    print("\nPrime 5 righe del dataset caricato:")
    print(X.head())
    print("\nPrime 5 etichette del target:")
    print(y.head())

### STEP 2: Analisi esplorativa dataset ###

    print("\nInformazioni sul dataset:")
    for dtype, cols in X.groupby(X.dtypes, axis=1):
        print(f"\n🔹 Tipo {dtype}: {len(cols.columns)} colonne")
        print(list(cols.columns))

    print("\nStatistiche descrittive del dataset:")
    print(X.describe(include='all'))

    print("\nConteggio dei valori nulli:")
    print(X.isnull().sum().sort_values(ascending=False))


    # Distribuizione della qualità del vino - variabile target
    #plt.figure(figsize=(10, 6))
    #sns.countplot(x = y, palette='viridis')
    #plt.title('Distribuzione della Qualità del Vino Rosso')
    #plt.xlabel('Qualità')
    #plt.ylabel('Conteggio')
    #plt.xticks(rotation=0)
    #plt.grid(axis='y')
    #plt.show()

    # Boxplot delle variabili indipendenti rispetto alla target
    # Combina X e y in un unico DataFrame temporaneo
    df = X.copy()
    df['quality'] = y  # aggiungi la colonna target
#
    #for col in df.columns:
    #    plt.figure(figsize=(10, 5))
    #    sns.boxplot(data=df, x='quality', y=col, palette='Set2')
    #    plt.title(f"Distribuzione di '{col}' rispetto alla Qualità del Vino")
    #    plt.xlabel("Qualità del Vino")
    #    plt.ylabel(col)
    #    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #    plt.tight_layout()
    #    plt.show()































