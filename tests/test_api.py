import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from io import BytesIO

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import MeVerifierAPI

class TestMeVerifierAPI(unittest.TestCase):
    """Tests para la API de verificación de identidad"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.api = MeVerifierAPI()
        self.app = self.api.app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Mock del modelo cargado
        self.api.model_loader.model_loaded = True
    
    def tearDown(self):
        """Limpieza después de cada test"""
        pass
    
    def test_health_check_success(self):
        """Test del endpoint de health check exitoso"""
        response = self.client.get('/healthz')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
        self.assertIn('model_loaded', data)
    
    def test_health_check_model_not_loaded(self):
        """Test del health check cuando el modelo no está cargado"""
        self.api.model_loader.model_loaded = False
        
        response = self.client.get('/healthz')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['model_loaded'])
    
    def test_verify_no_file(self):
        """Test del endpoint verify sin archivo"""
        response = self.client.post('/verify')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No se proporcionó archivo')
    
    def test_verify_empty_filename(self):
        """Test del endpoint verify con nombre de archivo vacío"""
        data = {'image': (BytesIO(b'fake image data'), '')}
        response = self.client.post('/verify', data=data)
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data)
        self.assertIn('error', response_data)
    
    def test_verify_invalid_file_type(self):
        """Test del endpoint verify con tipo de archivo inválido"""
        data = {'image': (BytesIO(b'fake data'), 'test.txt')}
        response = self.client.post('/verify', data=data)
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['error'], 'solo image/jpeg o image/png')
    
    def test_verify_file_too_large(self):
        """Test del endpoint verify con archivo demasiado grande"""
        # Crear un archivo grande (simular)
        large_data = b'x' * (11 * 1024 * 1024)  # 11MB
        data = {'image': (BytesIO(large_data), 'test.jpg')}
        response = self.client.post('/verify', data=data)
        
        self.assertEqual(response.status_code, 413)
        response_data = json.loads(response.data)
        self.assertIn('Archivo demasiado grande', response_data['error'])
    
    def test_verify_model_not_loaded(self):
        """Test del endpoint verify cuando el modelo no está cargado"""
        self.api.model_loader.model_loaded = False
        
        data = {'image': (BytesIO(b'fake image data'), 'test.jpg')}
        response = self.client.post('/verify', data=data)
        
        self.assertEqual(response.status_code, 503)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['error'], 'Modelo no encontrado. Entrene el modelo primero')
    
    @patch('api.app.PredictionService.predict')
    def test_verify_success(self, mock_predict):
        """Test del endpoint verify exitoso"""
        # Mock de la predicción
        mock_predict.return_value = {
            "model_version": "me-verifier-v1",
            "is_me": True,
            "score": 0.93,
            "threshold": 0.75,
            "timing_ms": 28.7
        }
        
        data = {'image': (BytesIO(b'fake image data'), 'test.jpg')}
        response = self.client.post('/verify', data=data)
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        
        self.assertEqual(response_data['model_version'], 'me-verifier-v1')
        self.assertEqual(response_data['is_me'], True)
        self.assertEqual(response_data['score'], 0.93)
        self.assertEqual(response_data['threshold'], 0.75)
        self.assertIn('timing_ms', response_data)
    
    @patch('api.app.PredictionService.predict')
    def test_verify_not_me(self, mock_predict):
        """Test del endpoint verify cuando no es la persona"""
        # Mock de la predicción negativa
        mock_predict.return_value = {
            "model_version": "me-verifier-v1",
            "is_me": False,
            "score": 0.62,
            "threshold": 0.75,
            "timing_ms": 25.1
        }
        
        data = {'image': (BytesIO(b'fake image data'), 'test.jpg')}
        response = self.client.post('/verify', data=data)
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        
        self.assertEqual(response_data['is_me'], False)
        self.assertEqual(response_data['score'], 0.62)
    
    def test_verify_png_file(self):
        """Test del endpoint verify con archivo PNG"""
        with patch('api.app.PredictionService.predict') as mock_predict:
            mock_predict.return_value = {
                "model_version": "me-verifier-v1",
                "is_me": True,
                "score": 0.88,
                "threshold": 0.75,
                "timing_ms": 30.2
            }
            
            data = {'image': (BytesIO(b'fake png data'), 'test.png')}
            response = self.client.post('/verify', data=data)
            
            self.assertEqual(response.status_code, 200)
            response_data = json.loads(response.data)
            self.assertEqual(response_data['is_me'], True)
    
    def test_invalid_endpoint(self):
        """Test de endpoint inexistente"""
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
    
    def test_verify_with_different_methods(self):
        """Test del endpoint verify con métodos HTTP incorrectos"""
        # GET no debería funcionar
        response = self.client.get('/verify')
        self.assertEqual(response.status_code, 405)
        
        # PUT no debería funcionar
        response = self.client.put('/verify')
        self.assertEqual(response.status_code, 405)

class MockDataGenerator:
    """Generador de datos mock para testing"""
    
    @staticmethod
    def create_fake_image_data(size_kb: int = 100) -> bytes:
        """Crea datos de imagen falsos"""
        return b'fake_image_data' * (size_kb * 1024 // 15)
    
    @staticmethod
    def create_test_embeddings(num_samples: int = 100, embedding_dim: int = 512):
        """Crea embeddings de prueba"""
        import numpy as np
        return np.random.rand(num_samples, embedding_dim)
    
    @staticmethod
    def create_test_labels(num_positive: int = 20, num_negative: int = 80):
        """Crea etiquetas de prueba"""
        import numpy as np
        labels = np.concatenate([
            np.ones(num_positive),
            np.zeros(num_negative)
        ])
        np.random.shuffle(labels)
        return labels
    
    @staticmethod
    def save_mock_data(embeddings_path: str, labels_path: str):
        """Guarda datos mock para testing"""
        import numpy as np
        import pandas as pd
        
        # Crear datos mock
        embeddings = MockDataGenerator.create_test_embeddings()
        labels = MockDataGenerator.create_test_labels()
        
        # Guardar embeddings
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.save(embeddings_path, embeddings)
        
        # Guardar etiquetas
        labels_df = pd.DataFrame({
            'filename': [f'image_{i}.jpg' for i in range(len(labels))],
            'label': labels
        })
        labels_df.to_csv(labels_path, index=False)
        
        print(f"Datos mock guardados: {embeddings_path}, {labels_path}")

class TestUtils(unittest.TestCase):
    """Tests para utilidades del proyecto"""
    
    def test_validation_utils(self):
        """Test de las utilidades de validación"""
        from utils import ValidationUtils
        
        # Test de extensiones permitidas
        self.assertTrue(ValidationUtils.is_allowed_file('test.jpg', ['jpg', 'png']))
        self.assertTrue(ValidationUtils.is_allowed_file('test.PNG', ['jpg', 'png']))
        self.assertFalse(ValidationUtils.is_allowed_file('test.txt', ['jpg', 'png']))
        self.assertFalse(ValidationUtils.is_allowed_file('test', ['jpg', 'png']))
        
        # Test de tamaño de archivo
        self.assertTrue(ValidationUtils.validate_file_size(1024, 1))  # 1KB < 1MB
        self.assertFalse(ValidationUtils.validate_file_size(2048*1024, 1))  # 2MB > 1MB
        
        # Test de tipo de archivo
        self.assertEqual(ValidationUtils.get_file_type('test.jpg'), 'image/jpeg')
        self.assertEqual(ValidationUtils.get_file_type('test.png'), 'image/png')
        self.assertEqual(ValidationUtils.get_file_type('test.txt'), 'unknown')

if __name__ == '__main__':
    # Configurar testing
    os.environ['TESTING'] = 'True'
    
    # Crear datos mock si no existen
    mock_generator = MockDataGenerator()
    
    # Ejecutar tests
    unittest.main(verbosity=2)