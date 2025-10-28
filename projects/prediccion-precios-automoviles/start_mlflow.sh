#!/bin/bash
# Script para iniciar MLFlow UI
# Este script es portable y funciona para cualquier usuario
# Uso: ./start_mlflow.sh

# Obtener directorio del script (raíz del proyecto)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Verificar si el puerto 5000 está ocupado
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Puerto 5000 ya está en uso"
    echo ""
    echo "Opciones:"
    echo "  1) Detener el proceso existente: pkill -f 'mlflow ui'"
    echo "  2) Usar otro puerto: mlflow ui --port 5001"
    echo ""
    read -p "¿Quieres detener el proceso existente? (s/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[SsYy]$ ]]; then
        pkill -f "mlflow ui"
        echo "✅ Proceso anterior detenido"
        sleep 1
    else
        echo "❌ Abortando..."
        exit 1
    fi
fi

echo "=========================================="
echo "  🚀 Iniciando MLFlow UI"
echo "=========================================="
echo "📁 Proyecto: $(basename $SCRIPT_DIR)"
echo "📊 MLFlow: mlruns/"
echo ""
echo "🌐 Abriendo en: http://localhost:5000"
echo "⏹️  Presiona Ctrl+C para detener"
echo "=========================================="
echo ""

# Cambiar al directorio del proyecto
cd "$SCRIPT_DIR"

# Iniciar MLFlow UI (usa mlruns/ automáticamente en el directorio actual)
mlflow ui
