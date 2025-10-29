from api.app import MeVerifierAPI
import unittest
import json
import tempfile
import os
from io import BytesIO

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMeVerifierAPI(unittest.TestCase):

    def setUp(self):
        self.api = MeVerifierAPI()
        self.app = self.api.app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

        self.api.model_loader.model_loaded = True

    def tearDown(self):
        pass

    def test_health_check_success(self):
        response = self.client.get('/healthz')

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
        self.assertIn('model_loaded', data)

    def test_health_check_model_not_loaded(self):
        self.api.model_loader.model_loaded = False

        response = self.client.get('/healthz')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['model_loaded'])

    def test_verify_no_file(self):
        response = self.client.post('/verify')

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_verify_empty_filename(self):
        data = {'image': (BytesIO(b'fake image data'), '')}
        response = self.client.post('/verify', data=data)

        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data)
        self.assertIn('error', response_data)

    def test_verify_invalid_file_type(self):
        data = {'image': (BytesIO(b'fake data'), 'test.txt')}
        response = self.client.post('/verify', data=data)

        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data)

    def test_verify_file_too_large(self):
        large_data = b'x' * (11 * 1024 * 1024)
        data = {'image': (BytesIO(large_data), 'test.jpg')}
        response = self.client.post('/verify', data=data)

        self.assertEqual(response.status_code, 413)
        response_data = json.loads(response.data)
        self.assertIn('error', response_data)

    def test_verify_model_not_loaded(self):
        self.api.model_loader.model_loaded = False

        data = {'image': (BytesIO(b'fake image data'), 'test.jpg')}
        response = self.client.post('/verify', data=data)

        self.assertEqual(response.status_code, 503)
        response_data = json.loads(response.data)
        self.assertIn('error', response_data)

    def test_invalid_endpoint(self):
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)

    def test_verify_with_different_methods(self):
        response = self.client.get('/verify')
        self.assertEqual(response.status_code, 405)

        response = self.client.put('/verify')
        self.assertEqual(response.status_code, 405)


class TestUtils(unittest.TestCase):

    def test_validation_utils(self):
        from utils import ValidationUtils

        self.assertTrue(ValidationUtils.is_allowed_file(
            'test.jpg', ['jpg', 'png']))
        self.assertTrue(ValidationUtils.is_allowed_file(
            'test.PNG', ['jpg', 'png']))
        self.assertFalse(ValidationUtils.is_allowed_file(
            'test.txt', ['jpg', 'png']))
        self.assertFalse(ValidationUtils.is_allowed_file(
            'test', ['jpg', 'png']))

        self.assertTrue(ValidationUtils.validate_file_size(1024, 1))
        self.assertFalse(ValidationUtils.validate_file_size(2048*1024, 1))

        self.assertEqual(ValidationUtils.get_file_type(
            'test.jpg'), 'image/jpeg')
        self.assertEqual(ValidationUtils.get_file_type(
            'test.png'), 'image/png')
        self.assertEqual(ValidationUtils.get_file_type('test.txt'), 'unknown')


if __name__ == '__main__':
    os.environ['TESTING'] = 'True'

    unittest.main(verbosity=2)
