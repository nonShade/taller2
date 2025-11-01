#!/bin/bash

# Script para ejecutar la aplicación con gunicorn en producción

# Cargar variables de entorno si existe .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuración por defecto
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-5000}
WORKERS=${WORKERS:-2}
TIMEOUT=${TIMEOUT:-120}
LOG_LEVEL=${LOG_LEVEL:-info}

# Crear directorio de logs si no existe
mkdir -p logs

echo "Iniciando servidor con gunicorn..."
echo "Host: $HOST"
echo "Puerto: $PORT"
echo "Workers: $WORKERS"
echo "Timeout: $TIMEOUT"

# Ejecutar gunicorn
exec gunicorn \
    --bind $HOST:$PORT \
    --workers $WORKERS \
    --timeout $TIMEOUT \
    --log-level $LOG_LEVEL \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --capture-output \
    --preload \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    api.app:application