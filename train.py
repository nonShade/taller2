import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

class DataLoader:
    """Carga y gestiona los datos de entrenamiento"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.embeddings_file = os.path.join(data_dir, "embeddings.npy")
        self.labels_file = os.path.join(data_dir, "labels.csv")
    
    def load_embeddings_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Carga embeddings y etiquetas desde archivos"""
        if not os.path.exists(self.embeddings_file) or not os.path.exists(self.labels_file):
            raise FileNotFoundError("Archivos de embeddings o etiquetas no encontrados")
        
        embeddings = np.load(self.embeddings_file)
        labels_df = pd.read_csv(self.labels_file)
        labels = labels_df['label'].values
        
        return embeddings, labels
    
    def check_data_balance(self, labels: np.ndarray) -> Dict[str, int]:
        """Verifica el balance de las clases"""
        unique, counts = np.unique(labels, return_counts=True)
        balance = dict(zip(unique, counts))
        
        print(f"Balance de datos:")
        print(f"  Clase 'YO' (1): {balance.get(1, 0)} muestras")
        print(f"  Clase 'NO-YO' (0): {balance.get(0, 0)} muestras")
        
        return balance

class MeVerifierTrainer:
    """Entrenador principal del modelo de verificación"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.training_history = {}
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Preprocesa los datos (normalización)"""
        print("Preprocesando datos...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   val_size: float = 0.1) -> Tuple[np.ndarray, ...]:
        """Divide los datos en entrenamiento, validación y prueba"""
        print(f"Dividiendo datos: {100*(1-test_size-val_size):.0f}% entrenamiento, "
              f"{100*val_size:.0f}% validación, {100*test_size:.0f}% prueba")
        
        # Primero separar test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Luego separar train y validación
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None) -> LogisticRegression:
        """Entrena el modelo de regresión logística"""
        print("Entrenando modelo de regresión logística...")
        
        # Configurar modelo
        self.model = LogisticRegression(
            max_iter=200,
            random_state=self.random_state,
            class_weight='balanced'  # Para manejar desbalance de clases
        )
        
        # Entrenar
        self.model.fit(X_train, y_train)
        
        # Evaluar en validación si está disponible
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            
            print(f"Accuracy en validación: {val_score:.4f}")
            print(f"AUC en validación: {val_auc:.4f}")
            
            self.training_history['val_accuracy'] = val_score
            self.training_history['val_auc'] = val_auc
        
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evalúa el modelo en el conjunto de prueba"""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        print("Evaluando modelo...")
        
        # Predicciones
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Reporte de clasificación
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'test_size': len(y_test)
        }
        
        print(f"Accuracy en prueba: {accuracy:.4f}")
        print(f"AUC en prueba: {auc:.4f}")
        
        return metrics
    
    def save_model(self, model_path: str = "models/model.joblib"):
        """Guarda el modelo entrenado"""
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Modelo guardado en: {model_path}")
    
    def save_scaler(self, scaler_path: str = "models/scaler.joblib"):
        """Guarda el scaler"""
        if self.scaler is None:
            raise ValueError("No hay scaler para guardar")
        
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler guardado en: {scaler_path}")
    
    def save_metrics(self, metrics: Dict[str, Any], metrics_path: str = "reports/metrics.json"):
        """Guarda las métricas de evaluación"""
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        # Añadir información de entrenamiento
        metrics['training_timestamp'] = datetime.now().isoformat()
        metrics['model_type'] = 'LogisticRegression'
        metrics['training_history'] = self.training_history
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Métricas guardadas en: {metrics_path}")

def main():
    """Función principal de entrenamiento"""
    print("=== Entrenamiento del Verificador de Identidad ===")
    
    # Inicializar componentes
    data_loader = DataLoader()
    trainer = MeVerifierTrainer()
    
    try:
        # Cargar datos
        print("\n1. Cargando datos...")
        X, y = data_loader.load_embeddings_and_labels()
        data_loader.check_data_balance(y)
        
        # Preprocesar
        print("\n2. Preprocesando datos...")
        X_scaled, y, scaler = trainer.preprocess_data(X, y)
        trainer.scaler = scaler
        
        # Dividir datos
        print("\n3. Dividiendo datos...")
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X_scaled, y)
        
        # Entrenar
        print("\n4. Entrenando modelo...")
        trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluar
        print("\n5. Evaluando modelo...")
        metrics = trainer.evaluate_model(X_test, y_test)
        
        # Guardar todo
        print("\n6. Guardando modelo y resultados...")
        trainer.save_model()
        trainer.save_scaler()
        trainer.save_metrics(metrics)
        
        print("\n¡Entrenamiento completado exitosamente!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Asegúrese de haber ejecutado primero el script de extracción de embeddings.")
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main()