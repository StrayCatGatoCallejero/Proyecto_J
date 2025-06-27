@echo off
REM Script de instalación automática para Proyecto J en Windows
REM Compatible con Python 3.11

echo 🚀 Instalando Proyecto J con Python 3.11...

REM Verificar si Python 3.11 está instalado
python --version 2>nul
if errorlevel 1 (
    echo ❌ Python no está instalado o no está en el PATH
    echo 📥 Por favor instala Python 3.11 desde https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Verificar versión de Python
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo ❌ Se requiere Python 3.11 o superior
    echo 📥 Por favor actualiza Python desde https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Crear entorno virtual
echo 📦 Creando entorno virtual...
python -m venv venv

REM Activar entorno virtual
echo 🔧 Activando entorno virtual...
call venv\Scripts\activate.bat

REM Actualizar pip
echo ⬆️ Actualizando pip...
python -m pip install --upgrade pip

REM Instalar dependencias
echo 📚 Instalando dependencias...
pip install -r requirements.txt

REM Verificar instalación
echo ✅ Verificando instalación...
python -c "import sys; import streamlit; import pandas; import plotly; import scipy; import sklearn; print('✅ Python', sys.version); print('✅ Streamlit instalado'); print('✅ Pandas instalado'); print('✅ Plotly instalado'); print('✅ SciPy instalado'); print('✅ Scikit-learn instalado'); print('🎉 ¡Instalación completada exitosamente!')"

echo.
echo 🎯 Para ejecutar el proyecto:
echo    venv\Scripts\activate.bat
echo    streamlit run app_front.py
echo.
echo 🌐 La aplicación estará disponible en: http://localhost:8501
pause 