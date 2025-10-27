import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from flask import Flask, jsonify, request

# Añadir el directorio padre al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Config, ValidationUtils, ErrorHandler, ResponseBuilder

class ModelLoader:
    """Carga y gestiona modelos entrenados"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
        self.model_loaded = False
    
    def load_models(self) -> bool:
        """Carga el modelo y scaler desde disco"""
        try:
            # Por ahora simulamos la carga - se implementará cuando tengamos modelos reales
            if os.path.exists(self.config.MODEL_PATH) and os.path.exists(self.config.SCALER_PATH):
                # Aquí iría la lógica real de carga con joblib
                self.model_loaded = True
                return True
            return False
        except Exception as e:
            logging.error(f"Error cargando modelos: {e}")
            return False
    
    def is_model_ready(self) -> bool:
        """Verifica si el modelo está listo para predicciones"""
        return self.model_loaded

class PredictionService:
    """Servicio para realizar predicciones"""
    
    def __init__(self, model_loader: ModelLoader, config: Config):
        self.model_loader = model_loader
        self.config = config
    
    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """Realiza predicción sobre una imagen"""
        start_time = time.time()
        
        if not self.model_loader.is_model_ready():
            raise Exception("Modelo no está cargado")
        
        # Simulación de predicción - se implementará con el modelo real
        # Por ahora retornamos valores mock
        mock_score = 0.85
        is_me = mock_score >= self.config.THRESHOLD
        
        timing_ms = (time.time() - start_time) * 1000
        
        return ResponseBuilder.build_verify_response(
            is_me=is_me,
            score=mock_score,
            threshold=self.config.THRESHOLD,
            timing_ms=timing_ms
        )

class MeVerifierAPI:
    """API principal para verificación de identidad facial"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.config = Config()
        self.model_loader = ModelLoader(self.config)
        self.prediction_service = PredictionService(self.model_loader, self.config)
        self.setup_routes()
        self.setup_logging()
        
        # Intentar cargar modelos al inicializar
        self.model_loader.load_models()
    
    def setup_logging(self):
        """Configura el sistema de logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
    
    def setup_routes(self):
        """Configura las rutas de la API"""
        self.app.route('/healthz', methods=['GET'])(self.health_check)
        self.app.route('/verify', methods=['POST'])(self.verify)
        
        # Configurar límite de tamaño de archivo
        self.app.config['MAX_CONTENT_LENGTH'] = self.config.MAX_FILE_SIZE_MB * 1024 * 1024
    
    def health_check(self):
        """Endpoint de verificación de salud"""
        try:
            response = ResponseBuilder.build_health_response()
            
            # Añadir información del modelo
            response['model_loaded'] = self.model_loader.is_model_ready()
            response['model_path'] = self.config.MODEL_PATH
            
            return jsonify(response), 200
            
        except Exception as e:
            logging.error(f"Error en health check: {e}")
            return ErrorHandler.create_error_response("Error interno del servidor", 500)
    
    def verify(self):
        """Endpoint principal de verificación de identidad"""
        try:
            # Verificar que se envió un archivo
            if 'image' not in request.files:
                return jsonify(*ErrorHandler.no_file_error())
            
            file = request.files['image']
            
            if file.filename == '':
                return jsonify(*ErrorHandler.no_file_error())
            
            # Validar tipo de archivo
            if not ValidationUtils.is_allowed_file(file.filename, self.config.ALLOWED_EXTENSIONS):
                return jsonify(*ErrorHandler.invalid_file_type_error())
            
            # Leer contenido del archivo
            file_content = file.read()
            
            # Validar tamaño
            if not ValidationUtils.validate_file_size(len(file_content), self.config.MAX_FILE_SIZE_MB):
                return jsonify(*ErrorHandler.file_too_large_error(self.config.MAX_FILE_SIZE_MB))
            
            # Verificar que el modelo está disponible
            if not self.model_loader.is_model_ready():
                return jsonify(*ErrorHandler.model_not_found_error())
            
            # Realizar predicción
            result = self.prediction_service.predict(file_content)
            
            # Log de la predicción
            logging.info(f"Predicción realizada: {result}")
            
            return jsonify(result), 200
            
        except Exception as e:
            logging.error(f"Error en verify endpoint: {e}")
            return jsonify(*ErrorHandler.create_error_response("Error procesando imagen", 500))
    
    def run(self, host=None, port=None, debug=None):
        """Ejecuta la aplicación Flask"""
        host = host or self.config.HOST
        port = port or self.config.PORT
        debug = debug if debug is not None else self.config.DEBUG
        
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    api = MeVerifierAPI()
    api.run()
