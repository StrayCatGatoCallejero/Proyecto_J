#!/bin/bash

# Script de instalación automática para Proyecto J
# Compatible con Python 3.11

echo "🚀 Instalando Proyecto J con Python 3.11..."

# Verificar si Python 3.11 está instalado
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 no está instalado."
    echo "📥 Instalando Python 3.11..."
    
    # Detectar el sistema operativo
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt update
        sudo apt install python3.11 python3.11-venv python3.11-pip
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install python@3.11
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Windows
        echo "⚠️  Por favor instala Python 3.11 manualmente desde https://www.python.org/downloads/"
        exit 1
    fi
fi

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
python3.11 -m venv venv

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

# Verificar instalación
echo "✅ Verificando instalación..."
python -c "
import sys
import streamlit
import pandas
import plotly
import scipy
import sklearn

print(f'✅ Python {sys.version}')
print('✅ Streamlit instalado')
print('✅ Pandas instalado')
print('✅ Plotly instalado')
print('✅ SciPy instalado')
print('✅ Scikit-learn instalado')
print('🎉 ¡Instalación completada exitosamente!')
"

echo ""
echo "🎯 Para ejecutar el proyecto:"
echo "   source venv/bin/activate"
echo "   streamlit run app_front.py"
echo ""
echo "🌐 La aplicación estará disponible en: http://localhost:8501" 