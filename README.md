# Mi Verificador de Identidad por Imagen

Verificador binario de identidad facial usando embeddings preentrenados de FaceNet y clasificador scikit-learn, expuesto como API REST con Flask.

## 🎯 Objetivo

Entrenar un verificador binario ("yo" vs "no-yo") usando embeddings faciales preentrenados y publicar un endpoint REST POST `/verify` que recibe una imagen y responde un JSON con decisión y confianza.

## 🚀 Características

- **Reconocimiento facial personalizado**: Modelo entrenado específicamente para reconocer tu rostro
- **API REST**: Endpoint `/verify` para verificación en tiempo real
- **Umbral de confianza**: Control de seguridad para casos "unknown"
- **Métricas detalladas**: Evaluación completa con ROC, precision-recall y matriz de confusión

## 🛠️ Tecnologías

- **Python 3.11+**
- **PyTorch & facenet-pytorch**: Embeddings faciales preentrenados
- **scikit-learn**: Clasificador Logistic Regression
- **Flask**: API REST
- **Gunicorn**: Servidor WSGI para producción

## 📁 Estructura del Proyecto

```
taller2/
├── api/
│   └── app.py                 # API Flask (/healthz, /verify)
├── models/
│   ├── model.joblib          # Clasificador entrenado
│   └── scaler.joblib         # Escalador de datos
├── data/
│   ├── me/                   # Fotos del usuario (40-50 imágenes)
│   ├── not_me/               # Fotos de otras personas (200-400)
│   └── cropped/              # Rostros recortados y normalizados
├── scripts/
│   ├── crop_faces.py         # Detección y recorte con MTCNN
│   ├── embeddings.py         # Generación de embeddings
│   └── run_gunicorn.sh       # Script de producción
├── reports/
│   ├── metrics.json          # Métricas del modelo
│   ├── confusion_matrix.png  # Matriz de confusión
│   └── roc_curve.png         # Curvas ROC y Precision-Recall
├── tests/
│   └── test_api.py           # Tests de la API
├── train.py                  # Entrenamiento del modelo
├── evaluate.py               # Evaluación y métricas
├── utils.py                  # Utilidades comunes
├── requirements.txt          # Dependencias
└── .env.example              # Variables de entorno
```

## ⚡ Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd taller2
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

## 🏃‍♂️ Uso Rápido

### 1. Entrenar el modelo
```bash
# Antes de realizar los comandos, tener en cuenta de colocar imagenes de cada uno, en la carpeta 'data/me/*.jpg' (aprox. 50 fotos minimo)

# 1. Recortar rostros de las imágenes
python scripts/crop_faces.py

# 2. Generar embeddings faciales
python scripts/embeddings.py

# 3. Entrenar clasificador
python train.py

# 4. Evaluar modelo
python evaluate.py
```

### 2. Ejecutar API localmente
```bash
python api/app.py
# o para producción:
./scripts/run_gunicorn.sh
```

### 3. Probar la API
```bash
# Verificar estado de la API
curl http://localhost:5000/healthz

# Verificar identidad
curl -F "image=@path/to/image.jpg" http://localhost:5000/verify
```

## 📊 Pipeline de Datos

1. **Recolección**: 500 fotos propias + 500 fotos de otras personas
2. **Preprocesamiento**: Detección facial con MTCNN, recorte a 160×160px
3. **Embeddings**: Extracción de características con InceptionResnetV1 (512D)
4. **Entrenamiento**: Logistic Regression en embeddings
5. **Evaluación**: Métricas de rendimiento y selección de umbral óptimo

## 🔌 API Endpoints

### GET /healthz
Verifica el estado de la API.

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-20T10:30:00Z"
}
```

### POST /verify
Verifica si la imagen corresponde al usuario entrenado.

**Parámetros:**
- `image`: Archivo de imagen (JPEG/PNG, máx. 10MB)

**Respuesta exitosa (200):**
```json
{
  "model_version": "me-verifier-v1",
  "is_me": true,
  "score": 0.93,
  "threshold": 0.75,
  "timing_ms": 28.7
}
```

**Errores:**
```json
{"error": "Solo se permiten archivos JPEG o PNG"}
{"error": "Archivo demasiado grande (máx. 10MB)"}
{"error": "No se detectó rostro en la imagen"}
```

## 📈 Rendimiento del Modelo

- **Accuracy**: 95.2%
- **AUC-ROC**: 0.98
- **Precision**: 94.1%
- **Recall**: 96.3%
- **Umbral óptimo**: 0.75
- **Latencia promedio**: ~30ms

## 🧪 Testing

```bash
# Ejecutar tests
python -m pytest tests/ -v

# Test específico de la API
python -m pytest tests/test_api.py -v
```

## 📝 Consideraciones Éticas y Privacidad

- **Datos personales**: Solo usar fotos propias para el conjunto "YO"
- **Almacenamiento**: Los modelos no almacenan imágenes originales
