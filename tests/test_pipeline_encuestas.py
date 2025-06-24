"""
Pruebas unitarias para el pipeline de encuestas
==============================================

Este módulo contiene pruebas para todas las funciones del pipeline
de procesamiento de datos de encuestas.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import subprocess
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Agregar el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar funciones a probar
from estadistica.pipeline_encuestas import (
    cargar_datos, explorar_datos, visualizar_nulos, imputar_nulos,
    analisis_ponderado, muestreo_estratificado, exportar_limpio,
    generar_reporte_pdf, validar_archivo, obtener_estadisticas_basicas
)

class TestCargarDatos:
    """Pruebas para la función cargar_datos."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        # Crear datos de prueba
        self.datos_prueba = pd.DataFrame({
            'edad': [25, 30, 35, 40, 45],
            'ingresos': [30000, 35000, 40000, 45000, 50000],
            'educacion': ['primaria', 'secundaria', 'universidad', 'posgrado', 'doctorado'],
            'peso': [1.0, 1.2, 0.8, 1.1, 0.9]
        })
        
        # Crear archivo temporal
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, 'datos_prueba.csv')
        self.datos_prueba.to_csv(self.csv_path, index=False)
    
    def teardown_method(self):
        """Limpieza después de cada prueba."""
        shutil.rmtree(self.temp_dir)
    
    def test_cargar_csv_exitoso(self):
        """Prueba carga exitosa de archivo CSV."""
        df = cargar_datos(self.csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ['edad', 'ingresos', 'educacion', 'peso']
    
    def test_cargar_archivo_inexistente(self):
        """Prueba error cuando el archivo no existe."""
        with pytest.raises(FileNotFoundError):
            cargar_datos('archivo_inexistente.csv')
    
    def test_validar_columnas_exitoso(self):
        """Prueba validación de columnas requeridas."""
        columnas_requeridas = ['edad', 'ingresos']
        df = cargar_datos(self.csv_path, validar_columnas=columnas_requeridas)
        assert isinstance(df, pd.DataFrame)
    
    def test_validar_columnas_faltantes(self):
        """Prueba error cuando faltan columnas requeridas."""
        columnas_requeridas = ['edad', 'columna_inexistente']
        with pytest.raises(ValueError, match="Columnas requeridas faltantes"):
            cargar_datos(self.csv_path, validar_columnas=columnas_requeridas)

class TestExplorarDatos:
    """Pruebas para la función explorar_datos."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        self.df = pd.DataFrame({
            'numerica': [1, 2, 3, np.nan, 5],
            'categorica': ['A', 'B', 'A', 'B', 'A'],
            'peso': [1.0, 1.2, 0.8, 1.1, 0.9]
        })
    
    def test_explorar_datos_estructura(self):
        """Prueba que la función retorna la estructura correcta."""
        info = explorar_datos(self.df)
        
        assert 'shape' in info
        assert 'dtypes' in info
        assert 'memory_usage' in info
        assert 'null_counts' in info
        assert 'null_percentage' in info
        assert 'numeric_columns' in info
        assert 'categorical_columns' in info
        assert 'duplicate_rows' in info
    
    def test_explorar_datos_valores(self):
        """Prueba que los valores calculados son correctos."""
        info = explorar_datos(self.df)
        
        assert info['shape'] == (5, 3)
        assert info['duplicate_rows'] == 0
        assert 'numerica' in info['numeric_columns']
        assert 'categorica' in info['categorical_columns']
        assert info['null_counts']['numerica'] == 1

class TestImputarNulos:
    """Pruebas para la función imputar_nulos."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        self.df = pd.DataFrame({
            'numerica': [1, 2, np.nan, 4, 5],
            'categorica': ['A', 'B', np.nan, 'A', 'B'],
            'peso': [1.0, 1.2, 0.8, 1.1, 0.9]
        })
    
    def test_imputar_media(self):
        """Prueba imputación con media."""
        df_limpio = imputar_nulos(self.df, estrategia='media')
        assert df_limpio['numerica'].isnull().sum() == 0
        assert df_limpio['numerica'].mean() == pytest.approx(3.0, rel=1e-2)
    
    def test_imputar_mediana(self):
        """Prueba imputación con mediana."""
        df_limpio = imputar_nulos(self.df, estrategia='mediana')
        assert df_limpio['numerica'].isnull().sum() == 0
    
    def test_imputar_moda(self):
        """Prueba imputación con moda para categóricos."""
        df_limpio = imputar_nulos(self.df, estrategia='moda')
        assert df_limpio['categorica'].isnull().sum() == 0
    
    def test_imputar_knn(self):
        """Prueba imputación con KNN."""
        df_limpio = imputar_nulos(self.df, estrategia='knn')
        assert df_limpio['numerica'].isnull().sum() == 0
    
    def test_columnas_especificas(self):
        """Prueba imputación en columnas específicas."""
        df_limpio = imputar_nulos(
            self.df, 
            estrategia='media',
            columnas_numericas=['numerica']
        )
        assert df_limpio['numerica'].isnull().sum() == 0
        # La columna categórica debe mantener sus valores faltantes
        assert df_limpio['categorica'].isnull().sum() == 1

class TestAnalisisPonderado:
    """Pruebas para la función analisis_ponderado."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        self.df = pd.DataFrame({
            'variable': [10, 20, 30, 40, 50],
            'peso': [1.0, 1.2, 0.8, 1.1, 0.9]
        })
    
    def test_analisis_ponderado_exitoso(self):
        """Prueba análisis ponderado exitoso."""
        resultados = analisis_ponderado(self.df, 'variable', 'peso')
        
        assert 'media_ponderada' in resultados
        assert 'varianza_ponderada' in resultados
        assert 'desv_est_ponderada' in resultados
        assert 'n_efectivo' in resultados
        assert 'suma_pesos' in resultados
        assert isinstance(resultados['media_ponderada'], (int, float))
    
    def test_analisis_ponderado_variable_inexistente(self):
        """Prueba error cuando la variable no existe."""
        with pytest.raises(ValueError, match="Variable 'inexistente' no encontrada"):
            analisis_ponderado(self.df, 'inexistente', 'peso')
    
    def test_analisis_ponderado_peso_inexistente(self):
        """Prueba error cuando el peso no existe."""
        with pytest.raises(ValueError, match="Variable de peso 'inexistente' no encontrada"):
            analisis_ponderado(self.df, 'variable', 'inexistente')
    
    def test_analisis_ponderado_datos_vacios(self):
        """Prueba error cuando no hay datos válidos."""
        df_vacio = pd.DataFrame({
            'variable': [np.nan, np.nan],
            'peso': [1.0, 1.0]
        })
        with pytest.raises(ValueError, match="No hay datos válidos"):
            analisis_ponderado(df_vacio, 'variable', 'peso')

class TestMuestreoEstratificado:
    """Pruebas para la función muestreo_estratificado."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        self.df = pd.DataFrame({
            'variable': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'estrato': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
        })
    
    def test_muestreo_estratificado_exitoso(self):
        """Prueba muestreo estratificado exitoso."""
        train_df, test_df = muestreo_estratificado(self.df, 'estrato')
        
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) + len(test_df) == len(self.df)
        assert len(train_df) > 0 and len(test_df) > 0
    
    def test_muestreo_estratificado_columna_inexistente(self):
        """Prueba error cuando la columna de estratificación no existe."""
        with pytest.raises(ValueError, match="Columna de estratificación 'inexistente' no encontrada"):
            muestreo_estratificado(self.df, 'inexistente')
    
    def test_muestreo_estratificado_diferentes_test_sizes(self):
        """Prueba diferentes tamaños de conjunto de prueba."""
        for test_size in [0.1, 0.2, 0.3]:
            train_df, test_df = muestreo_estratificado(self.df, 'estrato', test_size=test_size)
            expected_test_size = int(len(self.df) * test_size)
            assert len(test_df) == expected_test_size

class TestExportarLimpio:
    """Pruebas para la función exportar_limpio."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        self.df = pd.DataFrame({
            'columna1': [1, 2, 3],
            'columna2': ['A', 'B', 'C']
        })
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Limpieza después de cada prueba."""
        shutil.rmtree(self.temp_dir)
    
    def test_exportar_csv(self):
        """Prueba exportación a CSV."""
        ruta_salida = os.path.join(self.temp_dir, 'test.csv')
        ruta_resultado = exportar_limpio(self.df, ruta_salida, formato='csv')
        
        assert ruta_resultado == ruta_salida
        assert os.path.exists(ruta_salida)
        
        # Verificar contenido
        df_cargado = pd.read_csv(ruta_salida)
        pd.testing.assert_frame_equal(self.df, df_cargado)
    
    def test_exportar_excel(self):
        """Prueba exportación a Excel."""
        ruta_salida = os.path.join(self.temp_dir, 'test.xlsx')
        ruta_resultado = exportar_limpio(self.df, ruta_salida, formato='excel')
        
        assert ruta_resultado == ruta_salida
        assert os.path.exists(ruta_salida)
    
    def test_exportar_formato_invalido(self):
        """Prueba error con formato inválido."""
        ruta_salida = os.path.join(self.temp_dir, 'test.txt')
        with pytest.raises(ValueError, match="Formato no soportado"):
            exportar_limpio(self.df, ruta_salida, formato='txt')

class TestFuncionesAuxiliares:
    """Pruebas para funciones auxiliares."""
    
    def test_validar_archivo_existente(self):
        """Prueba validación de archivo existente."""
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b'test')
            temp_path = f.name
        
        try:
            assert validar_archivo(temp_path) == True
        finally:
            os.unlink(temp_path)
    
    def test_validar_archivo_inexistente(self):
        """Prueba validación de archivo inexistente."""
        assert validar_archivo('archivo_inexistente.csv') == False
    
    def test_validar_archivo_formato_invalido(self):
        """Prueba validación de archivo con formato inválido."""
        assert validar_archivo('archivo.txt') == False
    
    def test_obtener_estadisticas_basicas(self):
        """Prueba obtención de estadísticas básicas."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'A']
        })
        
        stats = obtener_estadisticas_basicas(df)
        
        assert stats['filas'] == 3
        assert stats['columnas'] == 2
        assert stats['valores_faltantes'] == 0
        assert stats['filas_duplicadas'] == 0
        assert 'memoria_mb' in stats

class TestGenerarReportePDF:
    """Pruebas para la función generar_reporte_pdf."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        self.df = pd.DataFrame({
            'columna1': [1, 2, 3],
            'columna2': ['A', 'B', 'C']
        })
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Limpieza después de cada prueba."""
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    def test_generar_reporte_pdf_exitoso(self, mock_run):
        """Prueba generación exitosa de reporte PDF."""
        # Mock del subprocess.run
        mock_run.return_value = MagicMock()
        
        ruta_pdf = os.path.join(self.temp_dir, 'test.pdf')
        resultado = generar_reporte_pdf(self.df, ruta_pdf)
        
        assert resultado == ruta_pdf
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_generar_reporte_pdf_error(self, mock_run):
        """Prueba error en generación de reporte PDF."""
        # Mock del subprocess.run para simular error
        mock_run.side_effect = subprocess.CalledProcessError(1, 'cmd', stderr='Error')
        
        ruta_pdf = os.path.join(self.temp_dir, 'test.pdf')
        resultado = generar_reporte_pdf(self.df, ruta_pdf)
        
        assert resultado is None

# Pruebas de integración
class TestIntegracion:
    """Pruebas de integración del pipeline completo."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        self.df = pd.DataFrame({
            'edad': [25, 30, 35, np.nan, 45],
            'ingresos': [30000, 35000, np.nan, 45000, 50000],
            'educacion': ['primaria', 'secundaria', 'universidad', np.nan, 'doctorado'],
            'peso': [1.0, 1.2, 0.8, 1.1, 0.9]
        })
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, 'datos_integracion.csv')
        self.df.to_csv(self.csv_path, index=False)
    
    def teardown_method(self):
        """Limpieza después de cada prueba."""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_completo_basico(self):
        """Prueba pipeline completo básico."""
        # Cargar datos
        df_cargado = cargar_datos(self.csv_path)
        assert len(df_cargado) == 5
        
        # Explorar datos
        info = explorar_datos(df_cargado)
        assert info['shape'] == (5, 4)
        
        # Imputar valores faltantes - usar estrategia que funcione para ambos tipos
        df_limpio = imputar_nulos(df_cargado, estrategia='knn')
        assert df_limpio.isnull().sum().sum() == 0
        
        # Análisis ponderado
        resultados = analisis_ponderado(df_limpio, 'edad', 'peso')
        assert 'media_ponderada' in resultados
        
        # Exportar
        ruta_export = os.path.join(self.temp_dir, 'export.csv')
        ruta_resultado = exportar_limpio(df_limpio, ruta_export)
        assert os.path.exists(ruta_resultado)

if __name__ == "__main__":
    pytest.main([__file__]) 