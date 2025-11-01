from utils import Config, ValidationUtils, ErrorHandler, ResponseBuilder
import os
import sys
import time
import logging
import joblib
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from datetime import datetime
from werkzeug.exceptions import RequestEntityTooLarge

from flask import Flask, jsonify, request, render_template
from facenet_pytorch import MTCNN, InceptionResnetV1

# Añadir el directorio padre al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RealModelLoader:
    """Carga y gestiona el modelo real entrenado"""

    def __init__(self, config: Config):
        self.config = config
        self.classifier = None
        self.scaler = None
        self.facenet_model = None
        self.mtcnn = None
        self.model_loaded = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Inicializando modelo en dispositivo: {self.device}")

    def load_models(self):
        """Carga todos los modelos necesarios"""
        try:
            # 1. Cargar modelo clasificador (LogisticRegression)
            if os.path.exists(self.config.MODEL_PATH):
                self.classifier = joblib.load(self.config.MODEL_PATH)
                logging.info(f"✅ Clasificador cargado: {
                             self.config.MODEL_PATH}")
            else:
                logging.error(f"❌ Clasificador no encontrado: {
                              self.config.MODEL_PATH}")
                return False

            # 2. Cargar scaler
            if os.path.exists(self.config.SCALER_PATH):
                self.scaler = joblib.load(self.config.SCALER_PATH)
                logging.info(f"✅ Scaler cargado: {self.config.SCALER_PATH}")
            else:
                logging.error(f"❌ Scaler no encontrado: {
                              self.config.SCALER_PATH}")
                return False

            # 3. Cargar FaceNet para embeddings
            self.facenet_model = InceptionResnetV1(
                pretrained='vggface2').eval()
            self.facenet_model.to(self.device)
            logging.info("✅ FaceNet cargado")

            # 4. Cargar MTCNN para detección de rostros
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device
            )
            logging.info("✅ MTCNN cargado")

            self.model_loaded = True
            logging.info("🎉 Todos los modelos cargados exitosamente!")
            return True

        except Exception as e:
            logging.error(f"❌ Error cargando modelos: {e}")
            return False

    def is_ready(self):
        """Verifica si todos los modelos están listos"""
        return (self.model_loaded and
                self.classifier is not None and
                self.scaler is not None and
                self.facenet_model is not None and
                self.mtcnn is not None)


class RealPredictionService:
    """Servicio que hace predicciones REALES usando el modelo entrenado"""

    def __init__(self, model_loader: RealModelLoader, config: Config):
        self.model_loader = model_loader
        self.config = config

    def process_image_to_embedding(self, image_bytes):
        """Convierte imagen a embedding usando el pipeline completo"""
        try:
            # 1. Cargar imagen
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            logging.info(f"Imagen cargada: {image.size}")

            # 2. Detectar rostro con MTCNN
            face_tensor = self.model_loader.mtcnn(image)

            if face_tensor is None:
                raise Exception("No se detectó ningún rostro en la imagen")

            logging.info("✅ Rostro detectado y extraído")

            # 3. Extraer embedding con FaceNet
            face_tensor = face_tensor.unsqueeze(0).to(self.model_loader.device)

            with torch.no_grad():
                embedding = self.model_loader.facenet_model(face_tensor)
                embedding_np = embedding.cpu().numpy().flatten()

            logging.info(f"✅ Embedding extraído: dimensión {
                         len(embedding_np)}")
            return embedding_np

        except Exception as e:
            logging.error(f"❌ Error procesando imagen: {e}")
            raise

    def predict(self, image_bytes):
        """Hace predicción REAL usando el modelo entrenado"""
        start_time = time.time()

        if not self.model_loader.is_ready():
            raise Exception("Los modelos no están cargados correctamente")

        try:
            # 1. Procesar imagen y extraer embedding
            embedding = self.process_image_to_embedding(image_bytes)

            # 2. Normalizar con el scaler entrenado
            embedding_scaled = self.model_loader.scaler.transform(
                embedding.reshape(1, -1))

            # 3. Predecir con el clasificador entrenado
            prediction_proba = self.model_loader.classifier.predict_proba(embedding_scaled)[
                0]
            score = float(prediction_proba[1])  # Probabilidad de que sea "YO"

            # 4. Aplicar umbral
            is_me = score >= self.config.THRESHOLD

            timing_ms = (time.time() - start_time) * 1000

            result = {
                "model_version": "me-verifier-v1-REAL",
                "is_me": is_me,
                "score": round(score, 4),
                "threshold": self.config.THRESHOLD,
                "timing_ms": round(timing_ms, 2)
            }

            logging.info(f"🔍 Predicción: {result}")
            return result

        except Exception as e:
            logging.error(f"❌ Error en predicción: {e}")
            raise


class MeVerifierAPI:
    """API de verificación facial con modelo REAL"""

    def __init__(self):
        self.app = Flask(__name__)
        self.config = Config()

        # Inicializar modelo real
        self.model_loader = RealModelLoader(self.config)
        self.prediction_service = RealPredictionService(
            self.model_loader, self.config)

        self.setup_logging()
        self.setup_routes()

        # Cargar modelos al inicializar
        print("🚀 Cargando modelos...")
        if self.model_loader.load_models():
            print("✅ API lista para verificación facial!")
        else:
            print("❌ Error cargando modelos - API en modo degradado")

    def setup_logging(self):
        """Configura logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE),
                logging.StreamHandler()
            ]
        )

    def setup_routes(self):
        """Configura rutas de la API"""
        self.app.route('/', methods=['GET'])(self.index)
        self.app.route('/healthz', methods=['GET'])(self.health_check)
        self.app.route('/verify', methods=['POST'])(self.verify)
        self.app.config['MAX_CONTENT_LENGTH'] = self.config.MAX_FILE_SIZE_MB * 1024 * 1024

    def health_check(self):
        """Health check con información del modelo REAL"""
        try:
            response = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version": "1.0.0-REAL",
                "model_loaded": self.model_loader.is_ready(),
                "model_path": self.config.MODEL_PATH,
                "device": self.model_loader.device,
                "model_type": "LogisticRegression + FaceNet",
                "threshold": self.config.THRESHOLD
            }
            return jsonify(response), 200
        except Exception as e:
            logging.error(f"Error en health check: {e}")
            return jsonify({"error": "Error interno del servidor"}), 500

    def index(self):
        """Página principal con interfaz de upload"""
        return render_template('index.html')

    def verify(self):
        """Endpoint de verificación con modelo REAL"""
        try:
            if 'image' not in request.files:
                return jsonify({"error": "No se envió ningún archivo"}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "Archivo vacío"}), 400

            if file.filename is None or not ValidationUtils.is_allowed_file(file.filename, self.config.ALLOWED_EXTENSIONS):
                return jsonify({"error": "Tipo de archivo no permitido. Use JPG o PNG"}), 400

            file_content = file.read()

            if not ValidationUtils.validate_file_size(len(file_content), self.config.MAX_FILE_SIZE_MB):
                return jsonify({"error": f"Archivo muy grande. Máximo {self.config.MAX_FILE_SIZE_MB}MB"}), 400

            if not self.model_loader.is_ready():
                return jsonify({"error": "Modelo no disponible"}), 503

            result = self.prediction_service.predict(file_content)
            return jsonify(result), 200
        except RequestEntityTooLarge:
            return jsonify({"error": f"Archivo muy grande. Máximo {self.config.MAX_FILE_SIZE_MB}MB"}), 413
        except Exception as e:
            logging.error(f"Error en verify: {e}")
            return jsonify({"error": f"Error procesando imagen: {str(e)}"}), 500

    def run(self, host=None, port=None, debug=None):
        """Ejecuta la aplicación"""
        host = host or self.config.HOST
        port = port or self.config.PORT
        debug = debug if debug is not None else self.config.DEBUG

        print(f"🌐 Iniciando API en http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# Para gunicorn - variable de aplicación WSGI
api = MeVerifierAPI()
application = api.app

if __name__ == '__main__':
    api.run()
