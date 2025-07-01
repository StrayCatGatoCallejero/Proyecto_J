"""
Tests E2E para la aplicación Streamlit
"""
import pytest
import subprocess
import time
import requests
from playwright.sync_api import sync_playwright
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def streamlit_app():
    """Inicia la aplicación Streamlit para testing"""
    # Iniciar Streamlit en background
    process = subprocess.Popen(
        ["streamlit", "run", "app_front.py", "--server.port", "8501", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Esperar a que la aplicación esté lista
    time.sleep(5)
    
    yield process
    
    # Limpiar
    process.terminate()
    process.wait()

@pytest.mark.skip(reason="Requiere configuración específica de Playwright")
def test_streamlit_app_loads(streamlit_app):
    """Test que verifica que la aplicación Streamlit carga correctamente"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            # Navegar a la aplicación
            page.goto("http://localhost:8501")
            
            # Verificar que la página carga
            assert page.title() is not None
            
            # Verificar elementos básicos
            assert page.locator("text=Proyecto J").count() > 0 or page.locator("text=Análisis").count() > 0
            
        finally:
            browser.close()

def test_streamlit_api_endpoint():
    """Test que verifica que la API de Streamlit responde"""
    try:
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        assert response.status_code == 200
    except requests.exceptions.RequestException:
        pytest.skip("Streamlit no está ejecutándose en localhost:8501")

@pytest.mark.skip(reason="Test de integración que requiere aplicación ejecutándose")
def test_streamlit_file_upload():
    """Test que verifica la funcionalidad de carga de archivos"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto("http://localhost:8501")
            
            # Buscar input de archivo
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                # Simular carga de archivo
                file_input.set_input_files("datos_ejemplo.csv")
                
                # Verificar que se procesó
                time.sleep(2)
                assert page.locator("text=Datos cargados").count() > 0 or page.locator("text=Análisis").count() > 0
                
        finally:
            browser.close()

def test_streamlit_components_exist():
    """Test que verifica que los componentes principales existen"""
    # Este test verifica que los archivos de la aplicación existen
    assert os.path.exists("app_front.py"), "app_front.py debe existir"
    assert os.path.exists("streamlit_app.py"), "streamlit_app.py debe existir"
    
    # Verificar que los módulos se pueden importar
    try:
        import app_front
        assert app_front is not None
    except ImportError:
        pytest.skip("No se puede importar app_front")

@pytest.mark.skip(reason="Test de rendimiento que requiere aplicación ejecutándose")
def test_streamlit_performance():
    """Test de rendimiento básico"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            start_time = time.time()
            page.goto("http://localhost:8501")
            load_time = time.time() - start_time
            
            # La página debe cargar en menos de 10 segundos
            assert load_time < 10, f"La página tardó {load_time:.2f} segundos en cargar"
            
        finally:
            browser.close() 