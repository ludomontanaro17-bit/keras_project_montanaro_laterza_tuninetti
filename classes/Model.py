# classes/Model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import os

class WineQualityNeuralNet:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.model = None

    def build(self, input_dim):
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
        if self.model is None:
            self.build(X_train.shape[1])

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

    def evaluate(self, X_val, y_val, class_names=None):
        if self.model is None:
            raise ValueError("Il modello non è stato ancora costruito o caricato.")

        y_pred = self.predict_classes(X_val)
        loss, acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\n🔍 Valutazione sul validation set:")
        print(f"   Loss: {loss:.4f}")
        print(f"   Accuracy: {acc:.4f}")

        if class_names is None:
            if self.num_classes == 3:
                class_names = ["basso", "medio", "alto"]
            else:
                class_names = [str(i) for i in range(self.num_classes)]

        print("\n📊 Classification Report:")
        print(classification_report(y_val, y_pred, target_names=class_names))
        cm = confusion_matrix(y_val, y_pred)
        print("\n🧮 Confusion Matrix:")
        print(cm)

        return {
            "accuracy": acc,
            "classification_report": classification_report(y_val, y_pred, target_names=class_names, output_dict=True),
            "confusion_matrix": cm
        }

    def predict(self, X):
        return self.model.predict(X)

    def predict_classes(self, X):
        return np.argmax(self.predict(X), axis=1)

    def save(self, path="model/wine_quality_model.keras"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"✅ Modello salvato in: {path}")

    def load(self, path="model/wine_quality_model.keras"):
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        print(f"✅ Modello caricato da: {path}")