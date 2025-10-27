import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm

class FaceDataset(Dataset):
    """Dataset personalizado para cargar rostros recortados"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Cargar imagen
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # Transformación básica para FaceNet
                image = np.array(image)
                image = torch.FloatTensor(image).permute(2, 0, 1)
                image = (image - 127.5) / 128.0  # Normalizar a [-1, 1]
            
            return image, label, image_path
            
        except Exception as e:
            print(f"Error cargando {image_path}: {e}")
            # Retornar tensor vacío en caso de error
            return torch.zeros(3, 160, 160), label, image_path

class FaceEmbeddings:
    """Extractor de embeddings faciales usando FaceNet preentrenado"""
    
    def __init__(self, device: str = None, batch_size: int = 32):
        """
        Inicializa el extractor de embeddings
        
        Args:
            device: Dispositivo a usar ('cpu' o 'cuda')
            batch_size: Tamaño del lote para procesamiento
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Usando dispositivo: {self.device}")
        
        # Cargar modelo FaceNet preentrenado
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.model.to(self.device)
        
        self.batch_size = batch_size
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def load_cropped_images(self, cropped_dir: str) -> Tuple[List[str], List[int], List[str]]:
        """
        Carga las rutas de imágenes recortadas y sus etiquetas
        
        Args:
            cropped_dir: Directorio con rostros recortados
            
        Returns:
            Tupla de (rutas_imagen, etiquetas, nombres_archivo)
        """
        image_paths = []
        labels = []
        filenames = []
        
        # Procesar carpeta 'me' (etiqueta 1)
        me_dir = os.path.join(cropped_dir, "me")
        if os.path.exists(me_dir):
            for filename in os.listdir(me_dir):
                if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                    image_paths.append(os.path.join(me_dir, filename))
                    labels.append(1)  # Etiqueta 'YO'
                    filenames.append(f"me/{filename}")
        
        # Procesar carpeta 'not_me' (etiqueta 0)
        not_me_dir = os.path.join(cropped_dir, "not_me")
        if os.path.exists(not_me_dir):
            for filename in os.listdir(not_me_dir):
                if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                    image_paths.append(os.path.join(not_me_dir, filename))
                    labels.append(0)  # Etiqueta 'NO-YO'
                    filenames.append(f"not_me/{filename}")
        
        print(f"Encontradas {len(image_paths)} imágenes:")
        print(f"  - YO: {sum(labels)} imágenes")
        print(f"  - NO-YO: {len(labels) - sum(labels)} imágenes")
        
        return image_paths, labels, filenames
    
    def extract_embeddings(self, image_paths: List[str], labels: List[int], 
                          filenames: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extrae embeddings de una lista de imágenes
        
        Args:
            image_paths: Lista de rutas de imágenes
            labels: Lista de etiquetas
            filenames: Lista de nombres de archivo
            
        Returns:
            Tupla de (embeddings, etiquetas, nombres_válidos)
        """
        dataset = FaceDataset(image_paths, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        all_embeddings = []
        valid_labels = []
        valid_filenames = []
        
        print("Extrayendo embeddings...")
        
        with torch.no_grad():
            for batch_images, batch_labels, batch_paths in tqdm(dataloader, desc="Procesando lotes"):
                # Mover a dispositivo
                batch_images = batch_images.to(self.device)
                
                # Verificar imágenes válidas (no vacías)
                valid_mask = batch_images.sum(dim=(1,2,3)) != 0
                
                if valid_mask.sum() > 0:
                    valid_images = batch_images[valid_mask]
                    valid_batch_labels = [batch_labels[i] for i in range(len(batch_labels)) if valid_mask[i]]
                    valid_batch_paths = [batch_paths[i] for i in range(len(batch_paths)) if valid_mask[i]]
                    
                    # Extraer embeddings
                    embeddings = self.model(valid_images)
                    
                    # Convertir a numpy y almacenar
                    embeddings_np = embeddings.cpu().numpy()
                    all_embeddings.append(embeddings_np)
                    valid_labels.extend(valid_batch_labels)
                    valid_filenames.extend(valid_batch_paths)
        
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            valid_labels = np.array(valid_labels)
        else:
            all_embeddings = np.array([])
            valid_labels = np.array([])
        
        print(f"Embeddings extraídos: {len(all_embeddings)}")
        print(f"Dimensión de embeddings: {all_embeddings.shape[1] if len(all_embeddings) > 0 else 0}")
        
        return all_embeddings, valid_labels, valid_filenames
    
    def save_embeddings(self, embeddings: np.ndarray, labels: np.ndarray, 
                       filenames: List[str], output_dir: str = "data"):
        """
        Guarda embeddings y etiquetas
        
        Args:
            embeddings: Array de embeddings
            labels: Array de etiquetas
            filenames: Lista de nombres de archivo
            output_dir: Directorio de salida
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar embeddings
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        print(f"Embeddings guardados en: {embeddings_path}")
        
        # Guardar etiquetas y metadatos
        labels_df = pd.DataFrame({
            'filename': filenames,
            'label': labels,
            'embedding_index': range(len(labels))
        })
        
        labels_path = os.path.join(output_dir, "labels.csv")
        labels_df.to_csv(labels_path, index=False)
        print(f"Etiquetas guardadas en: {labels_path}")
        
        # Guardar estadísticas
        stats = {
            'total_samples': len(embeddings),
            'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'positive_samples': int(np.sum(labels == 1)),
            'negative_samples': int(np.sum(labels == 0)),
            'class_balance': float(np.mean(labels)) if len(labels) > 0 else 0.0
        }
        
        stats_path = os.path.join(output_dir, "embedding_stats.json")
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Estadísticas guardadas en: {stats_path}")
        
        return stats
    
    def load_embeddings(self, embeddings_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga embeddings y etiquetas desde archivos
        
        Args:
            embeddings_path: Ruta del archivo de embeddings
            labels_path: Ruta del archivo de etiquetas
            
        Returns:
            Tupla de (embeddings, etiquetas)
        """
        if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Archivos de embeddings o etiquetas no encontrados")
        
        embeddings = np.load(embeddings_path)
        labels_df = pd.read_csv(labels_path)
        labels = labels_df['label'].values
        
        print(f"Embeddings cargados: {embeddings.shape}")
        print(f"Etiquetas cargadas: {len(labels)}")
        
        return embeddings, labels
    
    def process_cropped_faces(self, cropped_dir: str = "data/cropped", 
                            output_dir: str = "data") -> Dict:
        """
        Procesa todos los rostros recortados y extrae embeddings
        
        Args:
            cropped_dir: Directorio con rostros recortados
            output_dir: Directorio de salida
            
        Returns:
            Diccionario con estadísticas
        """
        # Cargar rutas de imágenes
        image_paths, labels, filenames = self.load_cropped_images(cropped_dir)
        
        if len(image_paths) == 0:
            raise ValueError("No se encontraron imágenes recortadas. "
                           "Ejecute scripts/crop_faces.py primero.")
        
        # Extraer embeddings
        embeddings, valid_labels, valid_filenames = self.extract_embeddings(
            image_paths, labels, filenames
        )
        
        if len(embeddings) == 0:
            raise ValueError("No se pudieron extraer embeddings válidos")
        
        # Guardar resultados
        stats = self.save_embeddings(embeddings, valid_labels, valid_filenames, output_dir)
        
        return stats

def print_summary(stats: Dict):
    """Imprime resumen del procesamiento de embeddings"""
    print("\n" + "="*50)
    print("RESUMEN DE EXTRACCIÓN DE EMBEDDINGS")
    print("="*50)
    
    print(f"Total de muestras: {stats['total_samples']}")
    print(f"Dimensión de embeddings: {stats['embedding_dimension']}")
    print(f"Muestras positivas (YO): {stats['positive_samples']}")
    print(f"Muestras negativas (NO-YO): {stats['negative_samples']}")
    print(f"Balance de clases: {stats['class_balance']:.3f}")
    
    # Recomendaciones
    if stats['total_samples'] < 50:
        print("\n⚠️  ADVERTENCIA: Pocas muestras para entrenamiento")
        print("   Recomendado: mínimo 50 muestras (20 YO + 30 NO-YO)")
    
    if stats['class_balance'] > 0.7 or stats['class_balance'] < 0.1:
        print("\n⚠️  ADVERTENCIA: Clases muy desbalanceadas")
        print(f"   Ratio actual: {stats['class_balance']:.3f}")
        print("   Recomendado: entre 0.1 y 0.3 (más negativos que positivos)")
    
    if stats['total_samples'] >= 50 and 0.1 <= stats['class_balance'] <= 0.3:
        print("\n✅ Dataset listo para entrenamiento!")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Extractor de embeddings faciales')
    parser.add_argument('--cropped-dir', default='data/cropped',
                       help='Directorio con rostros recortados (default: data/cropped)')
    parser.add_argument('--output-dir', default='data',
                       help='Directorio de salida (default: data)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamaño del lote (default: 32)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Dispositivo a usar (default: auto)')
    
    args = parser.parse_args()
    
    print("=== EXTRACTOR DE EMBEDDINGS FACIALES ===")
    print(f"Directorio de rostros: {args.cropped_dir}")
    print(f"Directorio de salida: {args.output_dir}")
    print(f"Tamaño de lote: {args.batch_size}")
    
    # Verificar dependencias
    try:
        import torch
        print(f"PyTorch versión: {torch.__version__}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch no encontrado. Instale con: pip install torch torchvision")
        return 1
    
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        print("❌ facenet-pytorch no encontrado. Instale con: pip install facenet-pytorch")
        return 1
    
    # Verificar que existen rostros recortados
    if not os.path.exists(args.cropped_dir):
        print(f"❌ Directorio {args.cropped_dir} no encontrado.")
        print("   Ejecute primero: python scripts/crop_faces.py")
        return 1
    
    # Inicializar y procesar
    embeddings_extractor = FaceEmbeddings(
        device=args.device,
        batch_size=args.batch_size
    )
    
    try:
        stats = embeddings_extractor.process_cropped_faces(
            args.cropped_dir, 
            args.output_dir
        )
        print_summary(stats)
        
        print(f"\n✅ Embeddings extraídos exitosamente!")
        print(f"Siguiente paso: python train.py")
        
    except Exception as e:
        print(f"❌ Error durante la extracción: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())