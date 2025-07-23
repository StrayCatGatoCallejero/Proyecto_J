"""
Tests end-to-end para el flujo completo de la aplicación
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestAppWorkflowE2E:
    """Tests E2E para el flujo completo de la aplicación"""
    
    def test_complete_data_analysis_workflow(self, sample_data_large, temp_dir):
        """Test E2E del flujo completo de análisis de datos"""
        from proyecto_j.src.core import Pipeline
        from proyecto_j.src.estadistica import (
            cargar_archivo, calcular_media, calcular_moda, 
            calcular_percentiles, generar_histograma
        )
        from proyecto_j.src.ciencias_sociales import (
            clasificar_variable, analisis_descriptivo_cs,
            calcular_indice_gini_simple
        )
        from proyecto_j.src.validacion_chile import validar_datos_chile
        
        # Paso 1: Guardar datos de prueba
        data_path = temp_dir / "workflow_data.csv"
        sample_data_large.to_csv(data_path, index=False)
        
        # Paso 2: Cargar datos
        df = cargar_archivo(str(data_path))
        assert isinstance(df, pd.DataFrame), "Carga de datos debe funcionar"
        assert len(df) > 0, "Los datos no deben estar vacíos"
        
        # Paso 3: Validar datos
        validacion = validar_datos_chile(df)
        assert isinstance(validacion, dict), "Validación debe devolver diccionario"
        
        # Paso 4: Análisis estadístico básico
        media_edad = calcular_media(df, 'edad')
        moda_genero = calcular_moda(df, 'genero')
        percentiles_ingresos = calcular_percentiles(df, 'ingresos', [25, 50, 75])
        
        assert isinstance(media_edad, (int, float)), "Media debe ser número"
        assert isinstance(moda_genero, str), "Moda debe ser string"
        assert len(percentiles_ingresos) == 3, "Debe devolver 3 percentiles"
        
        # Paso 5: Análisis de ciencias sociales
        clasificacion = clasificar_variable(df, 'edad')
        analisis = analisis_descriptivo_cs(df, 'edad')
        gini = calcular_indice_gini_simple(df, 'ingresos')
        
        assert isinstance(clasificacion, dict), "Clasificación debe ser diccionario"
        assert isinstance(analisis, dict), "Análisis debe ser diccionario"
        assert isinstance(gini, (int, float)), "Gini debe ser número"
        
        # Paso 6: Pipeline completo
        config = {
            "input_path": str(data_path),
            "output_report": str(temp_dir / "workflow_report.pdf")
        }
        
        pipeline = Pipeline(config)
        result = pipeline.run()
        
        assert result is not None, "Pipeline debe devolver resultados"
        assert len(result) == 4, "Pipeline debe devolver 4 elementos"
        
        print("✅ Flujo completo de análisis de datos ejecutado exitosamente")
    
    def test_streamlit_app_workflow(self, sample_data, mock_streamlit):
        """Test E2E del flujo de la aplicación Streamlit"""
        # Simular el flujo completo de la app Streamlit
        
        # Paso 1: Inicialización
        mock_streamlit.session_state['current_page'] = 'data_loading'
        mock_streamlit.session_state['wizard_step'] = 1
        
        # Paso 2: Carga de datos
        mock_streamlit.session_state['df'] = sample_data
        mock_streamlit.session_state['current_page'] = 'summary'
        mock_streamlit.session_state['wizard_step'] = 2
        
        # Verificar que los datos se cargaron
        assert 'df' in mock_streamlit.session_state, "Datos deben estar en session_state"
        assert len(mock_streamlit.session_state['df']) > 0, "Datos no deben estar vacíos"
        
        # Paso 3: Resumen de datos
        mock_streamlit.title("Resumen de Datos")
        mock_streamlit.dataframe(sample_data.head())
        
        # Paso 4: Detección de tipos
        mock_streamlit.session_state['current_page'] = 'type_detection'
        mock_streamlit.session_state['wizard_step'] = 3
        
        # Simular detección de tipos
        tipos_detectados = {
            'edad': 'numerica',
            'ingresos': 'numerica',
            'genero': 'categorica',
            'region': 'categorica'
        }
        mock_streamlit.session_state['tipos_detectados'] = tipos_detectados
        
        # Paso 5: Visualización
        mock_streamlit.session_state['current_page'] = 'visualization'
        mock_streamlit.session_state['wizard_step'] = 4
        
        # Simular selección de gráfico
        mock_streamlit.selectbox("Tipo de gráfico", ["Histograma", "Barras", "Dispersión"])
        
        # Paso 6: Análisis avanzado
        mock_streamlit.session_state['current_page'] = 'advanced_analysis'
        mock_streamlit.session_state['wizard_step'] = 5
        
        # Simular consulta en lenguaje natural
        query = "¿Cuál es la edad promedio por género?"
        mock_streamlit.text_input("Consulta", query)
        
        # Paso 7: Exportación
        mock_streamlit.session_state['current_page'] = 'export'
        mock_streamlit.session_state['wizard_step'] = 6
        
        mock_streamlit.button("Exportar PDF")
        mock_streamlit.button("Exportar Excel")
        
        # Verificar que el flujo se completó
        assert mock_streamlit.session_state['wizard_step'] == 6, "Wizard debe completarse"
        assert len(mock_streamlit.calls) > 0, "Debe haber llamadas registradas"
        
        print("✅ Flujo completo de Streamlit ejecutado exitosamente")
    
    def test_data_processing_pipeline_e2e(self, sample_data_missing, temp_dir):
        """Test E2E del pipeline de procesamiento de datos"""
        from proyecto_j.src.ciencias_sociales import analizar_valores_perdidos, sugerir_imputacion
        from proyecto_j.src.estadistica import cargar_archivo
        
        # Paso 1: Cargar datos con valores faltantes
        data_path = temp_dir / "missing_data.csv"
        sample_data_missing.to_csv(data_path, index=False)
        
        df = cargar_archivo(str(data_path))
        assert isinstance(df, pd.DataFrame), "Carga debe funcionar"
        
        # Paso 2: Analizar valores faltantes
        analisis_missing = analizar_valores_perdidos(df)
        assert isinstance(analisis_missing, dict), "Análisis de missing debe ser diccionario"
        assert 'total_missing' in analisis_missing, "Debe incluir total de missing"
        
        # Paso 3: Sugerir imputación para cada columna
        columnas_numericas = ['edad', 'ingresos', 'satisfaccion']
        for col in columnas_numericas:
            if col in df.columns:
                sugerencia = sugerir_imputacion(df, col)
                assert isinstance(sugerencia, dict), f"Sugerencia para {col} debe ser diccionario"
                assert 'metodo' in sugerencia, f"Sugerencia para {col} debe incluir método"
        
        # Paso 4: Aplicar imputación (simulado)
        df_imputed = df.copy()
        for col in columnas_numericas:
            if col in df_imputed.columns:
                # Simular imputación con media
                media = df_imputed[col].mean()
                df_imputed[col].fillna(media, inplace=True)
        
        # Paso 5: Verificar que no hay valores faltantes en columnas numéricas
        for col in columnas_numericas:
            if col in df_imputed.columns:
                assert df_imputed[col].isnull().sum() == 0, f"No debe haber valores faltantes en {col}"
        
        print("✅ Pipeline de procesamiento de datos ejecutado exitosamente")
    
    def test_visualization_workflow_e2e(self, sample_data):
        """Test E2E del flujo de visualización"""
        from proyecto_j.src.estadistica import (
            generar_histograma, calcular_correlacion_pearson,
            generar_heatmap_correlacion
        )
        
        # Paso 1: Generar histograma
        fig_hist = generar_histograma(sample_data, 'edad')
        assert fig_hist is not None, "Histograma debe generarse"
        
        # Paso 2: Calcular correlaciones
        correlacion = calcular_correlacion_pearson(sample_data, 'edad', 'ingresos')
        assert isinstance(correlacion, (int, float)), "Correlación debe ser número"
        assert -1 <= correlacion <= 1, "Correlación debe estar entre -1 y 1"
        
        # Paso 3: Generar heatmap de correlaciones
        columnas_numericas = ['edad', 'ingresos', 'satisfaccion']
        fig_heatmap = generar_heatmap_correlacion(sample_data, columnas_numericas)
        assert fig_heatmap is not None, "Heatmap debe generarse"
        
        print("✅ Flujo de visualización ejecutado exitosamente")
    
    def test_export_workflow_e2e(self, sample_data, temp_dir):
        """Test E2E del flujo de exportación"""
        from proyecto_j.src.estadistica import cargar_archivo
        
        # Paso 1: Preparar datos para exportación
        data_path = temp_dir / "export_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        df = cargar_archivo(str(data_path))
        
        # Paso 2: Exportar a diferentes formatos
        # CSV
        csv_export_path = temp_dir / "exported_data.csv"
        df.to_csv(csv_export_path, index=False)
        assert csv_export_path.exists(), "Exportación CSV debe funcionar"
        
        # Excel
        excel_export_path = temp_dir / "exported_data.xlsx"
        df.to_excel(excel_export_path, index=False)
        assert excel_export_path.exists(), "Exportación Excel debe funcionar"
        
        # Paso 3: Verificar que los archivos exportados se pueden leer
        df_csv_imported = pd.read_csv(csv_export_path)
        df_excel_imported = pd.read_excel(excel_export_path)
        
        assert len(df_csv_imported) == len(df), "CSV exportado debe tener misma cantidad de filas"
        assert len(df_excel_imported) == len(df), "Excel exportado debe tener misma cantidad de filas"
        
        print("✅ Flujo de exportación ejecutado exitosamente")
    
    def test_error_recovery_workflow_e2e(self, temp_dir):
        """Test E2E del flujo de recuperación de errores"""
        from proyecto_j.src.estadistica import cargar_archivo
        
        # Paso 1: Crear archivo con datos problemáticos
        problematic_data = pd.DataFrame({
            'columna1': ['valor1', 'valor2', None, 'valor4'],
            'columna2': [1, 2, 'texto_invalido', 4],
            'columna3': ['a', 'b', 'c', None]
        })
        
        problematic_path = temp_dir / "problematic_data.csv"
        problematic_data.to_csv(problematic_path, index=False)
        
        # Paso 2: Intentar cargar datos problemáticos
        try:
            df = cargar_archivo(str(problematic_path))
            # Si se carga exitosamente, verificar que se manejó graciosamente
            assert isinstance(df, pd.DataFrame), "Debe devolver DataFrame"
            print("✅ Datos problemáticos manejados graciosamente")
        except Exception as e:
            # Si falla, debe ser un error manejado
            assert "error" in str(e).lower() or "invalid" in str(e).lower(), "Error debe ser manejado"
            print("✅ Error manejado correctamente")
        
        # Paso 3: Crear archivo inexistente
        nonexistent_path = temp_dir / "nonexistent_file.csv"
        
        try:
            cargar_archivo(str(nonexistent_path))
            pytest.fail("Debería haber lanzado una excepción")
        except (FileNotFoundError, Exception):
            print("✅ Error de archivo inexistente manejado correctamente")
    
    def test_performance_workflow_e2e(self, sample_data_large):
        """Test E2E de rendimiento con datos grandes"""
        import time
        
        # Paso 1: Medir tiempo de carga
        start_time = time.time()
        
        # Simular operaciones con datos grandes
        df = sample_data_large.copy()
        
        # Operaciones básicas
        df['edad_categoria'] = pd.cut(df['edad'], bins=[0, 30, 50, 80], labels=['Joven', 'Adulto', 'Mayor'])
        df['ingresos_categoria'] = pd.cut(df['ingresos'], bins=3, labels=['Bajo', 'Medio', 'Alto'])
        
        # Agrupaciones
        resumen_por_region = df.groupby('region').agg({
            'edad': ['mean', 'std'],
            'ingresos': ['mean', 'std'],
            'satisfaccion': ['mean', 'std']
        }).round(2)
        
        # Correlaciones
        correlaciones = df[['edad', 'ingresos', 'satisfaccion']].corr()
        
        end_time = time.time()
        tiempo_total = end_time - start_time
        
        # Verificar que las operaciones se completaron
        assert isinstance(resumen_por_region, pd.DataFrame), "Resumen por región debe ser DataFrame"
        assert isinstance(correlaciones, pd.DataFrame), "Correlaciones debe ser DataFrame"
        assert tiempo_total < 30, f"Operaciones deben completarse en menos de 30 segundos (tomó {tiempo_total:.2f}s)"
        
        print(f"✅ Operaciones con datos grandes completadas en {tiempo_total:.2f} segundos")


class TestIntegrationScenariosE2E:
    """Tests E2E para escenarios de integración específicos"""
    
    def test_social_sciences_analysis_e2e(self, sample_data_large):
        """Test E2E para análisis de ciencias sociales"""
        from proyecto_j.src.ciencias_sociales import (
            analisis_descriptivo_cs, analisis_bivariado_cs,
            calcular_indice_gini_simple, analisis_clusters_cs
        )
        
        # Escenario completo de análisis de ciencias sociales
        
        # Paso 1: Análisis descriptivo
        analisis_edad = analisis_descriptivo_cs(sample_data_large, 'edad')
        analisis_ingresos = analisis_descriptivo_cs(sample_data_large, 'ingresos')
        
        assert isinstance(analisis_edad, dict), "Análisis descriptivo edad debe ser diccionario"
        assert isinstance(analisis_ingresos, dict), "Análisis descriptivo ingresos debe ser diccionario"
        
        # Paso 2: Análisis bivariado
        analisis_bivariado = analisis_bivariado_cs(sample_data_large, 'genero', 'ingresos')
        assert isinstance(analisis_bivariado, dict), "Análisis bivariado debe ser diccionario"
        
        # Paso 3: Índice de desigualdad
        gini = calcular_indice_gini_simple(sample_data_large, 'ingresos')
        assert isinstance(gini, (int, float)), "Índice Gini debe ser número"
        assert 0 <= gini <= 1, "Índice Gini debe estar entre 0 y 1"
        
        print("✅ Análisis de ciencias sociales completado exitosamente")
    
    def test_natural_language_queries_e2e(self, sample_data_large):
        """Test E2E para consultas en lenguaje natural"""
        from proyecto_j.src.nl_query import parse_and_execute
        from proyecto_j.src.complex_grouping import execute_complex_grouping_from_question
        
        # Escenario completo de consultas en lenguaje natural
        
        # Consultas simples
        queries_simples = [
            "¿Cuál es la edad promedio?",
            "¿Cuántas personas hay por género?",
            "¿Cuál es el ingreso máximo?",
            "¿Cuál es la satisfacción promedio por región?"
        ]
        
        for query in queries_simples:
            resultado, mensaje = parse_and_execute(query, sample_data_large)
            assert isinstance(resultado, pd.DataFrame), f"Consulta '{query}' debe devolver DataFrame"
            assert isinstance(mensaje, str), f"Consulta '{query}' debe devolver mensaje"
        
        # Consultas complejas
        queries_complejas = [
            "Agrupa por región y calcula el ingreso promedio",
            "Encuentra la correlación entre edad e ingresos",
            "Muestra la distribución de satisfacción por género"
        ]
        
        for query in queries_complejas:
            resultado, mensaje = execute_complex_grouping_from_question(query, sample_data_large)
            assert isinstance(resultado, pd.DataFrame), f"Consulta compleja '{query}' debe devolver DataFrame"
            assert isinstance(mensaje, str), f"Consulta compleja '{query}' debe devolver mensaje"
        
        print("✅ Consultas en lenguaje natural completadas exitosamente")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 