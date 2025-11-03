import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self, dataframe, target_column):
        self.dataframe = dataframe
        self.target_column = target_column
        self.scaler = StandardScaler()
    
    def display_info(self):
        """Mostra informazioni sul dataset."""
        print("\nInformazioni sul dataset:")
        for dtype, cols in self.X.groupby(self.X.dtypes, axis=1):
            print(f"\n🔹 Tipo {dtype}: {len(cols.columns)} colonne")
            print(list(cols.columns))

    def display_statistics(self):
        """Mostra statistiche descrittive del dataset."""
        print("\nStatistiche descrittive del dataset:")
        print(self.X.describe(include='all'))

    def display_missing_values(self):
        """Mostra il conteggio dei valori nulli nel dataset."""
        print("\nConteggio dei valori nulli:")
        print(self.X.isnull().sum().sort_values(ascending=False))

    def plot_quality_distribution(self):
        """Visualizza la distribuzione della qualità del vino."""
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.y, palette='viridis')
        plt.title('Distribuzione della Qualità del Vino Rosso')
        plt.xlabel('Qualità')
        plt.ylabel('Conteggio')
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        plt.show()

    def plot_boxplots(self):
        """Visualizza boxplot delle variabili indipendenti rispetto alla variabile target."""
        df = self.X.copy()
        df['quality'] = self.y  # aggiungi la colonna target

        for col in df.columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df, x='quality', y=col, palette='Set2')
            plt.title(f"Distribuzione di '{col}' rispetto alla Qualità del Vino")
            plt.xlabel("Qualità del Vino")
            plt.ylabel(col)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
    
    def remove_collinear_features(self, threshold=0.6):
        """Rimuove le variabili collineari usando una soglia specificata."""
        df_corr = self.dataframe.copy()
        corr = df_corr.corr(numeric_only=True)

        to_drop = set()
        cols = self.dataframe.drop(columns=[self.target_column]).columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_val = corr.loc[cols[i], cols[j]]
                if abs(corr_val) > threshold:
                    corr_with_y_i = abs(corr.loc[cols[i], self.target_column])
                    corr_with_y_j = abs(corr.loc[cols[j], self.target_column])
                    if corr_with_y_i < corr_with_y_j:
                        to_drop.add(cols[i])
                    else:
                        to_drop.add(cols[j])
                    print(f"⚠️  {cols[i]} e {cols[j]} sono correlate (ρ = {corr_val:.2f}). "
                          f"Rimuovo '{cols[i] if corr_with_y_i < corr_with_y_j else cols[j]}' "
                          f"perché meno correlata con '{self.target_column}'.")

        print("\n📉 Variabili da rimuovere per collinearità:")
        print(sorted(to_drop))

        X_reduced = self.dataframe.drop(columns=list(to_drop))
        print("\n📊 Variabili rimaste dopo la rimozione:")
        print(list(X_reduced.columns))

        return X_reduced
    
    def standardize_numeric_features(self):
        """Standardizza automaticamente tutte le colonne numeriche."""
        numerical_features = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_features) > 0:
            self.dataframe[numerical_features] = self.scaler.fit_transform(self.dataframe[numerical_features])
            print("Standardizzazione completata per le seguenti colonne numeriche:", list(numerical_features))
        else:
            print("Nessuna colonna numerica trovata per la standardizzazione.")

    def split_data(self, test_size=0.2, random_state=42):
        """Suddivide il dataset in training e validation set."""
        X = self.dataframe.drop(columns=[self.target_column])
        y = self.dataframe[self.target_column]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Suddivisione completata: 
              {len(X_train)} campioni per il training e {len(X_val)} per la validazione.")
        return X_train, X_val, y_train, y_val
        
    



