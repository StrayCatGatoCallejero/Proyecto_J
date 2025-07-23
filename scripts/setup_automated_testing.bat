@echo off
REM Script de configuración para pruebas automatizadas en Windows

echo 🔧 Configurando sistema de pruebas automatizadas...

REM Crear directorio de logs si no existe
if not exist "logs" mkdir logs

REM Verificar que Python esté disponible
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no está disponible en el PATH
    echo Por favor, instala Python y agrégalo al PATH
    pause
    exit /b 1
)

echo ✅ Python detectado

REM Instalar dependencias de prueba si no están instaladas
echo 📦 Verificando dependencias de prueba...
python -c "import pytest" >nul 2>&1
if errorlevel 1 (
    echo 📦 Instalando pytest...
    pip install pytest pytest-mock pytest-cov
)

python -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo 📦 Instalando pandas...
    pip install pandas
)

echo ✅ Dependencias verificadas

REM Crear script de prueba rápida
echo 🔧 Creando script de prueba rápida...
(
echo @echo off
echo echo 🧪 Ejecutando pruebas rápidas...
echo python scripts/run_automated_tests.py
echo if errorlevel 1 ^(
echo     echo ❌ Algunas pruebas fallaron
echo     pause
echo     exit /b 1
echo ^)
echo echo ✅ Todas las pruebas pasaron
echo pause
) > "run_tests.bat"

echo ✅ Script de prueba rápida creado: run_tests.bat

REM Crear tarea programada para Windows (opcional)
echo.
echo 📅 ¿Deseas crear una tarea programada para ejecutar pruebas automáticamente?
echo    Esto ejecutará las pruebas diariamente a las 6:00 AM
set /p create_task="¿Crear tarea programada? (s/n): "

if /i "%create_task%"=="s" (
    echo 📅 Creando tarea programada...
    
    REM Crear script para la tarea programada
    (
    echo @echo off
    echo cd /d "%~dp0"
    echo python scripts/run_automated_tests.py
    echo if errorlevel 1 ^(
    echo     echo ❌ Pruebas fallaron en ejecución automática
    echo     echo Fecha: %%date%% %%time%%
    echo     echo. ^>^> logs/automated_test_failures.log
    echo ^)
    ) > "scripts/run_scheduled_tests.bat"
    
    REM Crear la tarea programada
    schtasks /create /tn "ProyectoJ_AutomatedTests" /tr "%~dp0scripts\run_scheduled_tests.bat" /sc daily /st 06:00 /f
    
    if errorlevel 1 (
        echo ⚠️ No se pudo crear la tarea programada. Ejecuta como administrador.
    ) else (
        echo ✅ Tarea programada creada exitosamente
    )
)

echo.
echo 🎉 Configuración completada
echo.
echo 📋 Comandos disponibles:
echo    • run_tests.bat - Ejecutar pruebas rápidas
echo    • python scripts/run_automated_tests.py - Ejecutar pruebas completas
echo    • python -m pytest tests/ - Ejecutar pruebas específicas
echo.
echo 📁 Archivos creados:
echo    • logs/ - Directorio para logs de pruebas
echo    • run_tests.bat - Script de prueba rápida
echo    • scripts/run_automated_tests.py - Ejecutor de pruebas automatizadas
echo    • tests/test_file_upload_integration.py - Pruebas de carga de archivos
echo.
echo 💡 Consejos:
echo    • Ejecuta run_tests.bat antes de hacer commit
echo    • Revisa logs/automated_tests.log para detalles
echo    • Las pruebas se ejecutan automáticamente en GitHub Actions
echo.
pause 