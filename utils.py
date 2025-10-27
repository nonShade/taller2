import os
import logging
import json
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuración centralizada de la aplicación"""
    
    def __init__(self):
        self.MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.joblib')
        self.SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.joblib')
        self.THRESHOLD = float(os.getenv('THRESHOLD', '0.75'))
        self.PORT = int(os.getenv('PORT', '5000'))
        self.HOST = os.getenv('HOST', '0.0.0.0')
        self.WORKERS = int(os.getenv('WORKERS', '2'))
        self.MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '10'))
        self.ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', 'jpg,jpeg,png').split(',')
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
        self.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
        self.TESTING = os.getenv('TESTING', 'False').lower() == 'true'

class ValidationUtils:
    """Utilidades para validación de requests y archivos"""
    
    @staticmethod
    def is_allowed_file(filename: str, allowed_extensions: list) -> bool:
        """Verifica si el archivo tiene una extensión permitida"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in [ext.lower() for ext in allowed_extensions]
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int) -> bool:
        """Verifica si el archivo no excede el tamaño máximo"""
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
    
    @staticmethod
    def get_file_type(filename: str) -> str:
        """Obtiene el tipo MIME básico del archivo"""
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if ext in ['jpg', 'jpeg']:
            return 'image/jpeg'
        elif ext == 'png':
            return 'image/png'
        return 'unknown'

class ErrorHandler:
    """Manejo centralizado de errores"""
    
    @staticmethod
    def create_error_response(error_message: str, status_code: int = 400) -> tuple:
        """Crea una respuesta de error estandarizada"""
        return {
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }, status_code
    
    @staticmethod
    def file_too_large_error(max_size_mb: int) -> tuple:
        return ErrorHandler.create_error_response(
            f"Archivo demasiado grande. Máximo {max_size_mb}MB permitido",
            413
        )
    
    @staticmethod
    def invalid_file_type_error() -> tuple:
        return ErrorHandler.create_error_response(
            "solo image/jpeg o image/png",
            400
        )
    
    @staticmethod
    def no_file_error() -> tuple:
        return ErrorHandler.create_error_response(
            "No se proporcionó archivo",
            400
        )
    
    @staticmethod
    def model_not_found_error() -> tuple:
        return ErrorHandler.create_error_response(
            "Modelo no encontrado. Entrene el modelo primero",
            503
        )

class ResponseBuilder:
    """Constructor de respuestas estandarizadas"""
    
    @staticmethod
    def build_verify_response(is_me: bool, score: float, threshold: float, 
                            timing_ms: float, model_version: str = "me-verifier-v1") -> Dict[str, Any]:
        """Construye respuesta para el endpoint verify"""
        return {
            "model_version": model_version,
            "is_me": is_me,
            "score": round(score, 3),
            "threshold": threshold,
            "timing_ms": round(timing_ms, 1)
        }
    
    @staticmethod
    def build_health_response() -> Dict[str, Any]:
        """Construye respuesta para el endpoint health"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": "1.0.0"
        }