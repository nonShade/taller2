import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps

class JSONFormatter(logging.Formatter):
    """Formateador personalizado para logs en formato JSON"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Añadir información extra si está disponible
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        if hasattr(record, 'user_ip'):
            log_entry["user_ip"] = record.user_ip
            
        if hasattr(record, 'timing_ms'):
            log_entry["timing_ms"] = record.timing_ms
            
        if hasattr(record, 'file_size'):
            log_entry["file_size"] = record.file_size
            
        if hasattr(record, 'prediction_result'):
            log_entry["prediction_result"] = record.prediction_result
        
        return json.dumps(log_entry, ensure_ascii=False)

class APILogger:
    """Logger especializado para la API"""
    
    def __init__(self, name: str = "me_verifier_api"):
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """Configura el logger con formato JSON"""
        if not self.logger.handlers:
            # Handler para archivo
            file_handler = logging.FileHandler('logs/app.log')
            file_handler.setFormatter(JSONFormatter())
            
            # Handler para consola
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(JSONFormatter())
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
    
    def log_request(self, endpoint: str, method: str, user_ip: str, 
                   file_size: Optional[int] = None, request_id: Optional[str] = None):
        """Registra una petición entrante"""
        extra = {
            'request_id': request_id,
            'user_ip': user_ip,
            'file_size': file_size
        }
        self.logger.info(f"Request: {method} {endpoint}", extra=extra)
    
    def log_prediction(self, result: Dict[str, Any], timing_ms: float, 
                      request_id: Optional[str] = None):
        """Registra el resultado de una predicción"""
        extra = {
            'request_id': request_id,
            'timing_ms': timing_ms,
            'prediction_result': result
        }
        self.logger.info("Prediction completed", extra=extra)
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None,
                 request_id: Optional[str] = None):
        """Registra un error"""
        extra = {'request_id': request_id}
        if exception:
            self.logger.error(f"Error: {error_message} - {str(exception)}", extra=extra)
        else:
            self.logger.error(f"Error: {error_message}", extra=extra)

class PerformanceMonitor:
    """Monitor de rendimiento para endpoints"""
    
    @staticmethod
    def monitor_endpoint(endpoint_name: str):
        """Decorador para monitorear el rendimiento de endpoints"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    timing_ms = (end_time - start_time) * 1000
                    
                    # Log del rendimiento
                    logger = APILogger()
                    logger.logger.info(f"Endpoint {endpoint_name} completed", 
                                      extra={'timing_ms': timing_ms})
                    
                    return result
                    
                except Exception as e:
                    end_time = time.time()
                    timing_ms = (end_time - start_time) * 1000
                    
                    # Log del error con timing
                    logger = APILogger()
                    logger.log_error(f"Endpoint {endpoint_name} failed", e)
                    
                    raise
            
            return wrapper
        return decorator

class RequestIDGenerator:
    """Generador de IDs únicos para requests"""
    
    @staticmethod
    def generate() -> str:
        """Genera un ID único para el request"""
        import uuid
        return str(uuid.uuid4())[:8]

# Configuración global del logging
def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log"):
    """Configura el logging global de la aplicación"""
    
    # Crear directorio de logs si no existe
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configurar logging root
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Limpiar handlers existentes
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Handler para archivo con formato JSON
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter())
    
    # Handler para consola (formato simple para desarrollo)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Silenciar logs de bibliotecas externas
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)