import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import os

class WineQualityNeuralNet:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.model = None

    def build(self, input_dim):
        """Costruisce l'architettura della rete neurale."""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Allena il modello con pesi per le classi."""
        if self.model is None:
            self.build(X_train.shape[1])

        # Calcola pesi per classi sbilanciate
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def predict_classes(self, X):
        return np.argmax(self.predict(X), axis=1)

    def save(self, path="models/wine_model.keras"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"✅ Modello salvato in: {path}")

    def load(self, path="models/wine_model.keras"):
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        print(f"✅ Modello caricato da: {path}")