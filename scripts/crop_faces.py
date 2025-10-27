import os
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from typing import List, Tuple, Optional
import argparse

class FaceCropper:
    """Detector y recortador de rostros usando MTCNN"""
    
    def __init__(self, image_size: int = 160, margin: int = 0, device: str = None):
        """
        Inicializa el detector de rostros
        
        Args:
            image_size: Tamaño de salida de las imágenes recortadas
            margin: Margen alrededor del rostro detectado
            device: Dispositivo a usar ('cpu' o 'cuda')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Usando dispositivo: {device}")
        
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device
        )
        
        self.image_size = image_size
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def detect_faces(self, image_path: str) -> List[Tuple[np.ndarray, float]]:
        """
        Detecta rostros en una imagen
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Lista de tuplas (rostro_recortado, confianza)
        """
        try:
            # Cargar imagen
            image = Image.open(image_path).convert('RGB')
            
            # Detectar rostros con confianza
            boxes, probs = self.mtcnn.detect(image)
            
            if boxes is None:
                return []
            
            faces = []
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob > 0.9:  # Solo rostros con alta confianza
                    # Extraer rostro
                    face = self.mtcnn.extract(image, [box], save_path=None)
                    if face is not None and len(face) > 0:
                        face_array = face[0].permute(1, 2, 0).numpy()
                        # Convertir de [-1, 1] a [0, 255]
                        face_array = ((face_array + 1) * 127.5).astype(np.uint8)
                        faces.append((face_array, prob))
            
            return faces
            
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            return []
    
    def save_cropped_face(self, face: np.ndarray, output_path: str):
        """
        Guarda un rostro recortado
        
        Args:
            face: Array numpy del rostro
            output_path: Ruta de salida
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image = Image.fromarray(face)
            image.save(output_path, 'JPEG', quality=95)
            
        except Exception as e:
            print(f"Error guardando {output_path}: {e}")
    
    def crop_faces(self, input_dir: str, output_dir: str) -> dict:
        """
        Procesa todas las imágenes en un directorio
        
        Args:
            input_dir: Directorio con imágenes originales
            output_dir: Directorio para rostros recortados
            
        Returns:
            Diccionario con estadísticas del procesamiento
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Directorio de entrada no encontrado: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        stats = {
            'total_images': 0,
            'images_processed': 0,
            'faces_detected': 0,
            'faces_saved': 0,
            'errors': []
        }
        
        # Obtener todas las imágenes
        image_files = []
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in self.supported_formats):
                image_files.append(file)
        
        stats['total_images'] = len(image_files)
        print(f"Encontradas {len(image_files)} imágenes en {input_dir}")
        
        for filename in image_files:
            input_path = os.path.join(input_dir, filename)
            
            try:
                print(f"Procesando: {filename}")
                faces = self.detect_faces(input_path)
                
                if faces:
                    stats['images_processed'] += 1
                    stats['faces_detected'] += len(faces)
                    
                    # Guardar cada rostro detectado
                    base_name = os.path.splitext(filename)[0]
                    for i, (face, confidence) in enumerate(faces):
                        if len(faces) == 1:
                            output_filename = f"{base_name}.jpg"
                        else:
                            output_filename = f"{base_name}_face_{i+1}.jpg"
                        
                        output_path = os.path.join(output_dir, output_filename)
                        self.save_cropped_face(face, output_path)
                        stats['faces_saved'] += 1
                        
                        print(f"  └─ Rostro guardado: {output_filename} (confianza: {confidence:.3f})")
                else:
                    print(f"  └─ No se detectaron rostros")
                    
            except Exception as e:
                error_msg = f"Error procesando {filename}: {e}"
                print(f"  └─ {error_msg}")
                stats['errors'].append(error_msg)
        
        return stats
    
    def process_directories(self, base_data_dir: str = "data") -> dict:
        """
        Procesa directorios 'me' y 'not_me'
        
        Args:
            base_data_dir: Directorio base de datos
            
        Returns:
            Diccionario con estadísticas completas
        """
        me_dir = os.path.join(base_data_dir, "me")
        not_me_dir = os.path.join(base_data_dir, "not_me")
        cropped_dir = os.path.join(base_data_dir, "cropped")
        
        results = {}
        
        # Procesar fotos 'me'
        if os.path.exists(me_dir):
            print("\n=== Procesando fotos 'YO' ===")
            me_output = os.path.join(cropped_dir, "me")
            results['me'] = self.crop_faces(me_dir, me_output)
        else:
            print(f"Directorio {me_dir} no encontrado")
            results['me'] = None
        
        # Procesar fotos 'not_me'
        if os.path.exists(not_me_dir):
            print("\n=== Procesando fotos 'NO-YO' ===")
            not_me_output = os.path.join(cropped_dir, "not_me")
            results['not_me'] = self.crop_faces(not_me_dir, not_me_output)
        else:
            print(f"Directorio {not_me_dir} no encontrado")
            results['not_me'] = None
        
        return results

def print_summary(results: dict):
    """Imprime resumen del procesamiento"""
    print("\n" + "="*50)
    print("RESUMEN DEL PROCESAMIENTO")
    print("="*50)
    
    total_faces = 0
    total_errors = 0
    
    for category, stats in results.items():
        if stats is not None:
            print(f"\n{category.upper()}:")
            print(f"  Imágenes encontradas: {stats['total_images']}")
            print(f"  Imágenes procesadas: {stats['images_processed']}")
            print(f"  Rostros detectados: {stats['faces_detected']}")
            print(f"  Rostros guardados: {stats['faces_saved']}")
            print(f"  Errores: {len(stats['errors'])}")
            
            total_faces += stats['faces_saved']
            total_errors += len(stats['errors'])
            
            if stats['errors']:
                print("  Errores encontrados:")
                for error in stats['errors'][:3]:  # Mostrar solo los primeros 3
                    print(f"    - {error}")
                if len(stats['errors']) > 3:
                    print(f"    ... y {len(stats['errors']) - 3} errores más")
    
    print(f"\nTOTAL:")
    print(f"  Rostros extraídos: {total_faces}")
    print(f"  Errores totales: {total_errors}")
    
    if total_faces > 0:
        print(f"\n✅ Procesamiento completado exitosamente!")
        print(f"Los rostros recortados están en: data/cropped/")
    else:
        print(f"\n❌ No se encontraron rostros válidos.")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Detector y recortador de rostros')
    parser.add_argument('--data-dir', default='data', 
                       help='Directorio base de datos (default: data)')
    parser.add_argument('--image-size', type=int, default=160,
                       help='Tamaño de imagen de salida (default: 160)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Dispositivo a usar (default: auto)')
    
    args = parser.parse_args()
    
    print("=== DETECTOR Y RECORTADOR DE ROSTROS ===")
    print(f"Directorio de datos: {args.data_dir}")
    print(f"Tamaño de imagen: {args.image_size}x{args.image_size}")
    
    # Verificar dependencias
    try:
        import torch
        print(f"PyTorch versión: {torch.__version__}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch no encontrado. Instale con: pip install torch torchvision")
        return
    
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        print("❌ facenet-pytorch no encontrado. Instale con: pip install facenet-pytorch")
        return
    
    # Inicializar y procesar
    cropper = FaceCropper(
        image_size=args.image_size,
        device=args.device
    )
    
    try:
        results = cropper.process_directories(args.data_dir)
        print_summary(results)
        
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())