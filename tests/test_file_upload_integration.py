#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruebas de integración para la funcionalidad de carga de archivos
Estas pruebas se ejecutan automáticamente para verificar que la carga de archivos funciona correctamente
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import io

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Importar la función load_file del streamlit_app
try:
    from proyecto_j.streamlit_app import load_file
    STREAMLIT_APP_AVAILABLE = True
except ImportError:
    STREAMLIT_APP_AVAILABLE = False
    print("⚠️ No se pudo importar streamlit_app, usando función simulada")

class TestFileUploadIntegration:
    """Pruebas de integración para la carga de archivos"""
    
    @pytest.fixture
    def sample_data(self):
        """Crear datos de muestra para las pruebas"""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'id': range(1, n+1),
            'nombre': [f'Usuario {i}' for i in range(1, n+1)],
            'edad': np.random.randint(18, 80, n),
            'genero': np.random.choice(['Masculino', 'Femenino', 'No binario'], n),
            'ingresos': np.random.normal(50000, 20000, n),
            'satisfaccion': np.random.randint(1, 11, n),
            'fecha_registro': pd.date_range('2023-01-01', periods=n, freq='D')
        })
    
    @pytest.fixture
    def temp_files(self, sample_data):
        """Crear archivos temporales para las pruebas"""
        temp_dir = tempfile.mkdtemp()
        files = {}
        
        # CSV UTF-8
        csv_utf8_path = Path(temp_dir) / "test_utf8.csv"
        sample_data.to_csv(csv_utf8_path, index=False, encoding='utf-8')
        files['csv_utf8'] = csv_utf8_path
        
        # CSV Latin-1
        csv_latin_path = Path(temp_dir) / "test_latin1.csv"
        sample_data.to_csv(csv_latin_path, index=False, encoding='latin-1')
        files['csv_latin'] = csv_latin_path
        
        # Excel
        excel_path = Path(temp_dir) / "test.xlsx"
        sample_data.to_excel(excel_path, index=False)
        files['excel'] = excel_path
        
        yield files
        
        # Limpiar archivos temporales
        import shutil
        shutil.rmtree(temp_dir)
    
    def create_mock_uploaded_file(self, file_path):
        """Crear un mock de archivo subido de Streamlit"""
        mock_file = Mock()
        mock_file.name = Path(file_path).name
        mock_file.size = Path(file_path).stat().st_size
        
        # Leer el contenido del archivo
        with open(file_path, 'rb') as f:
            content = f.read()
        
        mock_file.getbuffer.return_value = content
        return mock_file
    
    @pytest.mark.skipif(not STREAMLIT_APP_AVAILABLE, reason="streamlit_app no disponible")
    @patch('streamlit.error')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_csv_utf8_loading(self, mock_warning, mock_info, mock_success, mock_error, temp_files):
        """Probar carga de archivo CSV con encoding UTF-8"""
        mock_file = self.create_mock_uploaded_file(temp_files['csv_utf8'])
        
        result = load_file(mock_file)
        
        # Verificar que no hubo errores
        mock_error.assert_not_called()
        
        # Verificar que se cargó correctamente
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50
        assert len(result.columns) == 7
        
        # Verificar que se mostraron mensajes de éxito
        mock_success.assert_called()
        mock_info.assert_called()
    
    @pytest.mark.skipif(not STREAMLIT_APP_AVAILABLE, reason="streamlit_app no disponible")
    @patch('streamlit.error')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_csv_latin1_loading(self, mock_warning, mock_info, mock_success, mock_error, temp_files):
        """Probar carga de archivo CSV con encoding Latin-1"""
        mock_file = self.create_mock_uploaded_file(temp_files['csv_latin'])
        
        result = load_file(mock_file)
        
        # Verificar que no hubo errores
        mock_error.assert_not_called()
        
        # Verificar que se cargó correctamente
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50
        assert len(result.columns) == 7
    
    @pytest.mark.skipif(not STREAMLIT_APP_AVAILABLE, reason="streamlit_app no disponible")
    @patch('streamlit.error')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_excel_loading(self, mock_warning, mock_info, mock_success, mock_error, temp_files):
        """Probar carga de archivo Excel"""
        mock_file = self.create_mock_uploaded_file(temp_files['excel'])
        
        result = load_file(mock_file)
        
        # Verificar que no hubo errores
        mock_error.assert_not_called()
        
        # Verificar que se cargó correctamente
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50
        assert len(result.columns) == 7
    
    @pytest.mark.skipif(not STREAMLIT_APP_AVAILABLE, reason="streamlit_app no disponible")
    @patch('streamlit.error')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_invalid_file_handling(self, mock_warning, mock_info, mock_success, mock_error):
        """Probar manejo de archivos inválidos"""
        # Crear un mock de archivo inválido
        mock_file = Mock()
        mock_file.name = "invalid.txt"
        mock_file.size = 100
        mock_file.getbuffer.return_value = b"invalid content"
        
        result = load_file(mock_file)
        
        # Verificar que se manejó el error correctamente
        assert result is None
        mock_error.assert_called()
    
    @pytest.mark.skipif(not STREAMLIT_APP_AVAILABLE, reason="streamlit_app no disponible")
    @patch('streamlit.error')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_none_file_handling(self, mock_warning, mock_info, mock_success, mock_error):
        """Probar manejo cuando no se sube ningún archivo"""
        result = load_file(None)
        
        # Verificar que se manejó correctamente
        assert result is None
        mock_error.assert_called()
    
    @pytest.mark.skipif(not STREAMLIT_APP_AVAILABLE, reason="streamlit_app no disponible")
    @patch('streamlit.error')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_corrupted_csv_handling(self, mock_warning, mock_info, mock_success, mock_error):
        """Probar manejo de archivos CSV corruptos"""
        # Crear un mock de archivo CSV corrupto
        mock_file = Mock()
        mock_file.name = "corrupted.csv"
        mock_file.size = 100
        mock_file.getbuffer.return_value = b"invalid,csv,content\nwith,broken,format"
        
        result = load_file(mock_file)
        
        # Verificar que se manejó el error correctamente
        assert result is not None  # Debería intentar cargar con manejo de errores
        mock_warning.assert_called()  # Debería mostrar advertencias sobre encoding

class TestFileUploadUnit:
    """Pruebas unitarias para componentes específicos de carga de archivos"""
    
    def test_file_extension_detection(self):
        """Probar detección de extensiones de archivo"""
        from pathlib import Path
        
        test_cases = [
            ("test.csv", ".csv"),
            ("data.xlsx", ".xlsx"),
            ("file.xls", ".xls"),
            ("spss.sav", ".sav"),
            ("stata.dta", ".dta"),
            ("no_extension", ""),
            ("UPPER.XLSX", ".xlsx"),
        ]
        
        for filename, expected_ext in test_cases:
            ext = Path(filename).suffix.lower()
            assert ext == expected_ext, f"Para {filename}, esperaba {expected_ext}, obtuve {ext}"
    
    def test_encoding_detection(self):
        """Probar detección de encoding (si chardet está disponible)"""
        try:
            import chardet
            
            # Crear datos de prueba con diferentes encodings
            test_data = "áéíóú,ñ,ü\n1,2,3"
            
            # UTF-8
            utf8_bytes = test_data.encode('utf-8')
            result = chardet.detect(utf8_bytes)
            assert result['encoding'] in ['utf-8', 'ascii']
            
            # Latin-1
            latin1_bytes = test_data.encode('latin-1')
            result = chardet.detect(latin1_bytes)
            assert result['encoding'] in ['iso-8859-1', 'latin-1']
            
        except ImportError:
            pytest.skip("chardet no disponible")
    
    def test_pandas_loading_methods(self):
        """Probar métodos de carga de pandas"""
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        # Probar guardar y cargar CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            loaded_data = pd.read_csv(f.name)
            assert loaded_data.equals(test_data)
            os.unlink(f.name)
        
        # Probar guardar y cargar Excel
        try:
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
                test_data.to_excel(f.name, index=False)
                loaded_data = pd.read_excel(f.name)
                assert loaded_data.equals(test_data)
                os.unlink(f.name)
        except ImportError:
            pytest.skip("openpyxl no disponible para Excel")

def run_file_upload_tests():
    """Función para ejecutar las pruebas de carga de archivos"""
    print("🧪 Ejecutando pruebas de carga de archivos...")
    
    # Ejecutar pruebas con pytest
    import pytest
    import sys
    
    # Configurar argumentos de pytest
    args = [
        __file__,
        "-v",
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ]
    
    # Ejecutar pruebas
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("✅ Todas las pruebas de carga de archivos pasaron")
    else:
        print("❌ Algunas pruebas de carga de archivos fallaron")
    
    return exit_code == 0

if __name__ == "__main__":
    success = run_file_upload_tests()
    sys.exit(0 if success else 1) 