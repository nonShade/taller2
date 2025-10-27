import os
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)

class MeVerifierEvaluator:
    """Evaluador completo del modelo de verificación de identidad"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = 0.75
    
    def load_model(self, model_path: str = "models/model.joblib"):
        """Carga el modelo entrenado"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"Modelo cargado desde: {model_path}")
    
    def load_scaler(self, scaler_path: str = "models/scaler.joblib"):
        """Carga el scaler"""
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler no encontrado en: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler cargado desde: {scaler_path}")
    
    def load_test_data(self, embeddings_path: str = "data/embeddings.npy", 
                      labels_path: str = "data/labels.csv") -> Tuple[np.ndarray, np.ndarray]:
        """Carga los datos de prueba"""
        if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Archivos de datos de prueba no encontrados")
        
        embeddings = np.load(embeddings_path)
        labels_df = pd.read_csv(labels_path)
        labels = labels_df['label'].values
        
        return embeddings, labels
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluación completa del modelo"""
        if self.model is None:
            raise ValueError("Modelo no cargado")
        
        print("Realizando evaluación completa...")
        
        # Preprocesar datos de prueba
        if self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Predicciones
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Métricas básicas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'test_samples': len(y_test)
        }
        
        # Reporte de clasificación detallado
        class_report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = conf_matrix.tolist()
        
        # Análisis de umbrales
        threshold_analysis = self.analyze_thresholds(y_test, y_proba)
        metrics['threshold_analysis'] = threshold_analysis
        
        # Imprimir resultados
        self.print_evaluation_results(metrics)
        
        return metrics
    
    def analyze_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Analiza diferentes umbrales para encontrar el óptimo"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            # Calcular métricas para este umbral
            acc = accuracy_score(y_true, y_pred_thresh)
            prec = precision_score(y_true, y_pred_thresh, zero_division=0)
            rec = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            
            results.append({
                'threshold': round(threshold, 2),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            })
        
        # Encontrar umbral óptimo (mayor F1-score)
        best_threshold = max(results, key=lambda x: x['f1_score'])
        
        return {
            'all_thresholds': results,
            'optimal_threshold': best_threshold,
            'recommended_threshold': best_threshold['threshold']
        }
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                save_path: str = "reports/confusion_matrix.png"):
        """Genera y guarda la matriz de confusión"""
        plt.figure(figsize=(8, 6))
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['NO-YO', 'YO'], yticklabels=['NO-YO', 'YO'])
        
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matriz de confusión guardada en: {save_path}")
    
    def generate_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                          save_path: str = "reports/roc_curve.png"):
        """Genera y guarda la curva ROC"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Curva ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Curva ROC guardada en: {save_path}")
    
    def generate_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                      save_path: str = "reports/precision_recall_curve.png"):
        """Genera y guarda la curva Precision-Recall"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall')
        plt.grid(True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Curva Precision-Recall guardada en: {save_path}")
    
    def save_metrics(self, metrics: Dict[str, Any], 
                    filepath: str = "reports/evaluation_metrics.json"):
        """Guarda las métricas de evaluación"""
        metrics['evaluation_timestamp'] = datetime.now().isoformat()
        metrics['threshold_used'] = self.threshold
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Métricas guardadas en: {filepath}")
    
    def print_evaluation_results(self, metrics: Dict[str, Any]):
        """Imprime los resultados de evaluación de forma legible"""
        print("\n=== RESULTADOS DE EVALUACIÓN ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"Muestras de prueba: {metrics['test_samples']}")
        
        if 'threshold_analysis' in metrics:
            optimal = metrics['threshold_analysis']['optimal_threshold']
            print(f"\nUmbral óptimo recomendado: {optimal['threshold']}")
            print(f"  - Accuracy: {optimal['accuracy']:.4f}")
            print(f"  - F1-Score: {optimal['f1_score']:.4f}")

def main():
    """Función principal de evaluación"""
    print("=== Evaluación del Verificador de Identidad ===")
    
    evaluator = MeVerifierEvaluator()
    
    try:
        # Cargar modelo y scaler
        print("\n1. Cargando modelo y scaler...")
        evaluator.load_model()
        evaluator.load_scaler()
        
        # Cargar datos de prueba
        print("\n2. Cargando datos de prueba...")
        X_test, y_test = evaluator.load_test_data()
        
        # Evaluar modelo
        print("\n3. Evaluando modelo...")
        metrics = evaluator.evaluate_model(X_test, y_test)
        
        # Generar visualizaciones
        print("\n4. Generando visualizaciones...")
        # Preprocesar para visualizaciones
        if evaluator.scaler is not None:
            X_test_scaled = evaluator.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        y_pred = evaluator.model.predict(X_test_scaled)
        y_proba = evaluator.model.predict_proba(X_test_scaled)[:, 1]
        
        evaluator.generate_confusion_matrix(y_test, y_pred)
        evaluator.generate_roc_curve(y_test, y_proba)
        evaluator.generate_precision_recall_curve(y_test, y_proba)
        
        # Guardar métricas
        print("\n5. Guardando resultados...")
        evaluator.save_metrics(metrics)
        
        print("\n¡Evaluación completada exitosamente!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Asegúrese de haber entrenado el modelo primero.")
    except Exception as e:
        print(f"Error durante la evaluación: {e}")
        raise

if __name__ == "__main__":
    main()