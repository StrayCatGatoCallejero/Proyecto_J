"""
Tests de integración para el pipeline completo
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))


class TestPipelineIntegration:
    """Tests de integración para el pipeline completo"""
    
    def test_pipeline_core_integration(self, sample_data, temp_dir):
        """Test de integración del pipeline core"""
        from proyecto_j.src.core import Pipeline
        
        # Crear configuración de prueba
        config = {
            "input_path": str(Path(temp_dir) / "test_data.csv"),
            "output_report": str(Path(temp_dir) / "test_report.pdf")
        }
        
        # Guardar datos de prueba
        sample_data.to_csv(config["input_path"], index=False)
        
        # Crear y ejecutar pipeline
        pipeline = Pipeline(config)
        result = pipeline.run()
        
        # Verificar que el pipeline devuelve resultados
        assert result is not None, "El pipeline debe devolver resultados"
        assert len(result) == 4, "El pipeline debe devolver 4 elementos (df, model, results, metadata)"
        
        df, model, results, metadata = result
        
        # Verificar que los datos se procesaron correctamente
        assert isinstance(df, pd.DataFrame), "El resultado debe ser un DataFrame"
        assert len(df) > 0, "El DataFrame no debe estar vacío"
        assert len(df.columns) > 0, "El DataFrame debe tener columnas"
    
    def test_pipeline_steps_integration(self, sample_data, temp_dir):
        """Test de integración de los pasos del pipeline"""
        from proyecto_j.src.steps import (
            cargar_datos, limpiar_datos, transformar_datos, 
            modelar, visualizar, generar_reporte
        )
        
        # Crear archivo de datos
        data_path = Path(temp_dir) / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        # Ejecutar pasos individuales
        df_loaded = cargar_datos(str(data_path))
        assert isinstance(df_loaded, pd.DataFrame), "cargar_datos debe devolver un DataFrame"
        
        df_cleaned = limpiar_datos(df_loaded)
        assert isinstance(df_cleaned, pd.DataFrame), "limpiar_datos debe devolver un DataFrame"
        
        df_transformed = transformar_datos(df_cleaned)
        assert isinstance(df_transformed, pd.DataFrame), "transformar_datos debe devolver un DataFrame"
        
        # Verificar que los datos se mantienen consistentes
        assert len(df_loaded) == len(df_cleaned), "La limpieza no debe eliminar filas sin razón"
        assert len(df_cleaned) == len(df_transformed), "La transformación no debe eliminar filas sin razón"
    
    def test_estadistica_integration(self, sample_data):
        """Test de integración del módulo estadística"""
        from proyecto_j.src.estadistica import (
            calcular_media, calcular_moda, calcular_percentiles,
            generar_histograma, calcular_correlacion_pearson
        )
        
        # Probar funciones estadísticas
        media_edad = calcular_media(sample_data, 'edad')
        assert isinstance(media_edad, (int, float)), "calcular_media debe devolver un número"
        assert media_edad > 0, "La media de edad debe ser positiva"
        
        moda_genero = calcular_moda(sample_data, 'genero')
        # La moda puede devolver una lista si hay empate
        if isinstance(moda_genero, list):
            assert all(m in ['M', 'F'] for m in moda_genero), "Todos los valores de la moda deben ser M o F"
        else:
            assert moda_genero in ['M', 'F'], "La moda del género debe ser M o F"
        
        percentiles_ingresos = calcular_percentiles(sample_data, 'ingresos', [25, 50, 75])
        assert isinstance(percentiles_ingresos, dict), "calcular_percentiles debe devolver un diccionario"
        assert len(percentiles_ingresos) == 3, "Debe devolver 3 percentiles"
        assert all(isinstance(p, (int, float)) for p in percentiles_ingresos.values()), "Todos los percentiles deben ser números"
        
        # Probar correlación
        correlacion = calcular_correlacion_pearson(sample_data, ['edad', 'ingresos'])
        assert isinstance(correlacion, pd.DataFrame), "La correlación debe ser un DataFrame"
        assert correlacion.shape == (2, 2), "La matriz de correlación debe ser 2x2"
        # Verificar que los valores están entre -1 y 1
        for i in range(2):
            for j in range(2):
                assert -1 <= correlacion.iloc[i, j] <= 1, "Los valores de correlación deben estar entre -1 y 1"
    
    def test_ciencias_sociales_integration(self, sample_data):
        """Test de integración del módulo ciencias sociales"""
        from proyecto_j.src.ciencias_sociales import (
            clasificar_variable, analisis_descriptivo_cs,
            calcular_indice_gini_simple, analizar_valores_perdidos
        )
        
        # Probar clasificación de variables
        clasificacion = clasificar_variable(sample_data, 'edad')
        assert isinstance(clasificacion, dict), "clasificar_variable debe devolver un diccionario"
        # Verificar que tiene al menos algunos campos esperados
        expected_fields = ['dominio', 'es_continua', 'es_binaria', 'es_likert']
        assert any(field in clasificacion for field in expected_fields), "La clasificación debe incluir campos de clasificación"
        
        # Probar análisis descriptivo
        analisis = analisis_descriptivo_cs(sample_data, 'edad')
        assert isinstance(analisis, dict), "analisis_descriptivo_cs debe devolver un diccionario"
        assert 'estadisticas_basicas' in analisis, "El análisis debe incluir estadísticas básicas"
        # Verificar que las estadísticas básicas contienen media y mediana
        stats_basicas = analisis.get('estadisticas_basicas', {})
        assert 'media' in stats_basicas or 'promedio' in stats_basicas, "Las estadísticas básicas deben incluir la media"
        assert 'mediana' in stats_basicas, "Las estadísticas básicas deben incluir la mediana"
        
        # Probar índice de Gini
        gini = calcular_indice_gini_simple(sample_data, 'ingresos')
        assert isinstance(gini, (int, float)), "El índice de Gini debe ser un número"
        assert 0 <= gini <= 1, "El índice de Gini debe estar entre 0 y 1"
    
    def test_validacion_chile_integration(self, sample_data):
        """Test de integración del módulo validación Chile"""
        from proyecto_j.src.validacion_chile import validar_datos_chile
        
        # Probar validación de datos chilenos
        resultado = validar_datos_chile(sample_data)
        assert isinstance(resultado, dict), "validar_datos_chile debe devolver un diccionario"
        assert 'valido' in resultado, "El resultado debe incluir el campo 'valido'"
        assert 'errores' in resultado, "El resultado debe incluir el campo 'errores'"
        assert 'advertencias' in resultado, "El resultado debe incluir el campo 'advertencias'"
    
    def test_nl_query_integration(self, sample_data):
        """Test de integración del módulo consultas naturales"""
        from proyecto_j.src.nl_query import parse_and_execute
        
        # Probar consulta en lenguaje natural
        query = "¿Cuál es la edad promedio por género?"
        df_filtrado, resultado, df_grafico, tipo_grafico, interpretacion = parse_and_execute(sample_data, query)
        
        assert isinstance(df_filtrado, pd.DataFrame), "parse_and_execute debe devolver un DataFrame filtrado"
        assert isinstance(interpretacion, str), "parse_and_execute debe devolver una interpretación"
    
    def test_complex_grouping_integration(self, sample_data):
        """Test de integración del módulo agrupaciones complejas"""
        from proyecto_j.src.complex_grouping import execute_complex_grouping_from_question
        
        # Probar agrupación compleja
        query = "Agrupa por región y calcula el ingreso promedio"
        resultado, mensaje = execute_complex_grouping_from_question(query, sample_data)
        
        assert isinstance(resultado, pd.DataFrame), "execute_complex_grouping debe devolver un DataFrame"
        assert isinstance(mensaje, str), "execute_complex_grouping debe devolver un mensaje"
    
    def test_streamlit_integration(self, sample_data, mock_streamlit):
        """Test de integración con Streamlit"""
        # Simular carga de datos en Streamlit
        mock_streamlit.session_state['df'] = sample_data
        
        # Verificar que los datos se pueden acceder
        assert 'df' in mock_streamlit.session_state, "Los datos deben estar en session_state"
        assert isinstance(mock_streamlit.session_state['df'], pd.DataFrame), "Los datos deben ser un DataFrame"
        
        # Simular operaciones de Streamlit
        mock_streamlit.title("Test de Integración")
        mock_streamlit.dataframe(sample_data)
        
        # Verificar que las operaciones se registraron
        assert len(mock_streamlit.calls) == 2, "Debe haber 2 llamadas registradas"
        assert mock_streamlit.calls[0][0] == 'title', "La primera llamada debe ser title"
        assert mock_streamlit.calls[1][0] == 'dataframe', "La segunda llamada debe ser dataframe"


class TestDataFlowIntegration:
    """Tests de integración para el flujo de datos"""
    
    def test_data_loading_integration(self, sample_csv_file, sample_excel_file):
        """Test de integración de carga de datos"""
        from proyecto_j.src.estadistica import cargar_archivo
        
        # Probar carga de CSV
        df_csv = cargar_archivo(sample_csv_file)
        assert isinstance(df_csv, pd.DataFrame), "Carga de CSV debe devolver DataFrame"
        assert len(df_csv) > 0, "CSV cargado no debe estar vacío"
        
        # Probar carga de Excel
        df_excel = cargar_archivo(sample_excel_file)
        assert isinstance(df_excel, pd.DataFrame), "Carga de Excel debe devolver DataFrame"
        assert len(df_excel) > 0, "Excel cargado no debe estar vacío"
    
    def test_data_processing_integration(self, sample_data_missing):
        """Test de integración de procesamiento de datos con valores faltantes"""
        from proyecto_j.src.ciencias_sociales import analizar_valores_perdidos, sugerir_imputacion
        
        # Analizar valores perdidos
        analisis = analizar_valores_perdidos(sample_data_missing)
        assert isinstance(analisis, dict), "analizar_valores_perdidos debe devolver un diccionario"
        # Verificar que tiene campos esperados del análisis
        expected_fields = ['porcentaje_total_perdidos', 'porcentajes_por_variable', 'conteo_por_variable']
        assert any(field in analisis for field in expected_fields), "El análisis debe incluir campos de análisis de valores perdidos"
        
        # Sugerir imputación
        sugerencias = sugerir_imputacion(sample_data_missing, 'edad')
        assert isinstance(sugerencias, dict), "sugerir_imputacion debe devolver un diccionario"
        assert 'metodos_recomendados' in sugerencias, "Las sugerencias deben incluir métodos recomendados"
    
    def test_visualization_integration(self, sample_data):
        """Test de integración de visualización"""
        from proyecto_j.src.estadistica import generar_histograma
        
        # Probar generación de histograma
        fig = generar_histograma(sample_data, 'edad')
        assert fig is not None, "generar_histograma debe devolver una figura"
        # La figura puede ser de plotly, matplotlib, etc., dependiendo de la implementación


class TestErrorHandlingIntegration:
    """Tests de integración para manejo de errores"""
    
    def test_error_handling_integration(self, temp_dir):
        """Test de integración del manejo de errores"""
        from proyecto_j.src.estadistica import cargar_archivo
        
        # Probar carga de archivo inexistente
        archivo_inexistente = Path(temp_dir) / "archivo_inexistente.csv"
        
        try:
            cargar_archivo(str(archivo_inexistente))
            pytest.fail("Debería haber lanzado una excepción")
        except (FileNotFoundError, Exception):
            # Esperado que lance una excepción
            pass
    
    def test_invalid_data_handling(self, temp_dir):
        """Test de manejo de datos inválidos"""
        # Crear archivo con datos inválidos
        archivo_invalido = Path(temp_dir) / "datos_invalidos.csv"
        with open(archivo_invalido, 'w') as f:
            f.write("columna1,columna2\n")
            f.write("valor1,valor2\n")
            f.write("valor3,valor4\n")
        
        # El sistema debería manejar esto graciosamente
        from proyecto_j.src.estadistica import cargar_archivo
        try:
            df = cargar_archivo(str(archivo_invalido))
            assert isinstance(df, pd.DataFrame), "Debería devolver un DataFrame"
        except Exception as e:
            # Si falla, debe ser un error manejado
            assert "error" in str(e).lower() or "invalid" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 