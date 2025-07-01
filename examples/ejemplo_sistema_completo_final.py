"""
Ejemplo Completo del Sistema con Serialización Robusta
=====================================================

Demuestra el sistema completo funcionando con serialize_for_json en un escenario real.
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# Importar componentes del sistema
from processing.json_logging import JsonLogger, LogLevel, LogCategory, serialize_for_json


def crear_datos_reales():
    """Crear datos reales de una encuesta de satisfacción"""
    
    np.random.seed(42)
    
    # Datos de encuesta de satisfacción
    data = {
        'id_encuesta': range(1, 101),
        'edad': np.random.randint(18, 80, 100),
        'genero': np.random.choice(['M', 'F'], 100),
        'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universidad', 'Postgrado'], 100),
        'ingresos_mensuales': np.random.normal(50000, 15000, 100),
        'satisfaccion_servicio': np.random.randint(1, 11, 100),
        'satisfaccion_atencion': np.random.randint(1, 11, 100),
        'satisfaccion_precio': np.random.randint(1, 11, 100),
        'recomendaria': np.random.choice(['Sí', 'No', 'Tal vez'], 100),
        'fecha_encuesta': pd.date_range('2023-01-01', periods=100, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Calcular estadísticas reales
    stats = {
        'resumen_general': {
            'total_encuestas': np.int64(len(df)),
            'promedio_edad': np.float64(df['edad'].mean()),
            'desviacion_edad': np.float64(df['edad'].std()),
            'satisfaccion_promedio': np.float64(df[['satisfaccion_servicio', 'satisfaccion_atencion', 'satisfaccion_precio']].mean().mean())
        },
        'por_genero': {
            'masculino': {
                'count': np.int64(len(df[df['genero'] == 'M'])),
                'satisfaccion_promedio': np.float64(df[df['genero'] == 'M'][['satisfaccion_servicio', 'satisfaccion_atencion', 'satisfaccion_precio']].mean().mean())
            },
            'femenino': {
                'count': np.int64(len(df[df['genero'] == 'F'])),
                'satisfaccion_promedio': np.float64(df[df['genero'] == 'F'][['satisfaccion_servicio', 'satisfaccion_atencion', 'satisfaccion_precio']].mean().mean())
            }
        },
        'por_educacion': df.groupby('educacion').agg({
            'satisfaccion_servicio': ['mean', 'count'],
            'satisfaccion_atencion': 'mean',
            'satisfaccion_precio': 'mean'
        }).to_dict(),
        'correlaciones': {
            'edad_satisfaccion': np.float64(df['edad'].corr(df['satisfaccion_servicio'])),
            'ingresos_satisfaccion': np.float64(df['ingresos_mensuales'].corr(df['satisfaccion_servicio'])),
            'matriz_correlacion': df[['edad', 'ingresos_mensuales', 'satisfaccion_servicio', 'satisfaccion_atencion', 'satisfaccion_precio']].corr().to_dict()
        }
    }
    
    return df, stats


def simular_pipeline_completo():
    """Simular un pipeline completo de análisis de encuestas"""
    
    print("🚀 Iniciando Pipeline Completo de Análisis de Encuestas...")
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, 'encuesta_analysis.json')
    
    try:
        # Crear logger
        logger = JsonLogger(
            file_path=log_file,
            level='INFO',
            session_id=f'encuesta_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            console_output=True
        )
        
        # 1. Carga de datos
        print("📊 Cargando datos de encuesta...")
        df, stats = crear_datos_reales()
        
        logger.log_event(
            level=LogLevel.INFO,
            message="Datos de encuesta cargados exitosamente",
            module="data_loader",
            function="load_survey_data",
            step="data_loading",
            category=LogCategory.DATA_LOAD.value,
            parameters={"source": "simulated_survey", "format": "dataframe"},
            before_metrics={"expected_rows": 100},
            after_metrics={
                "actual_rows": np.int64(len(df)),
                "actual_columns": np.int64(len(df.columns)),
                "memory_usage_mb": np.float64(df.memory_usage(deep=True).sum() / (1024 * 1024)),
                "data_types": df.dtypes.value_counts().to_dict()
            },
            execution_time=0.25,
            tags=["survey_data", "loading"],
            metadata={"survey_period": "2023-01-01 to 2023-04-10"}
        )
        
        # 2. Limpieza de datos
        print("🧹 Limpiando datos...")
        
        # Simular limpieza
        df_clean = df.copy()
        df_clean = df_clean.dropna()
        
        cleaning_stats = {
            "rows_original": np.int64(len(df)),
            "rows_after_cleaning": np.int64(len(df_clean)),
            "rows_removed": np.int64(len(df) - len(df_clean)),
            "missing_values_removed": np.int64(df.isnull().sum().sum())
        }
        
        logger.log_event(
            level=LogLevel.INFO,
            message="Limpieza de datos completada",
            module="data_cleaner",
            function="clean_survey_data",
            step="data_cleaning",
            category=LogCategory.PROCESSING.value,
            parameters={"methods": ["remove_missing", "validate_ranges"]},
            before_metrics={"rows": len(df)},
            after_metrics=cleaning_stats,
            execution_time=0.15,
            tags=["data_cleaning", "missing_values"],
            metadata=cleaning_stats
        )
        
        # 3. Análisis estadístico
        print("📈 Ejecutando análisis estadístico...")
        
        logger.log_event(
            level=LogLevel.INFO,
            message="Análisis estadístico completado",
            module="statistical_analyzer",
            function="analyze_survey_data",
            step="statistical_analysis",
            category=LogCategory.ANALYSIS.value,
            parameters={"methods": ["descriptive", "correlation", "group_analysis"]},
            before_metrics={"input_size": len(df_clean)},
            after_metrics={
                "output_size": len(stats),
                "analysis_time": np.float64(0.45),
                "statistics_generated": list(stats.keys())
            },
            execution_time=0.45,
            tags=["statistical_analysis", "survey"],
            metadata=stats
        )
        
        # 4. Generación de visualizaciones
        print("📊 Generando visualizaciones...")
        
        try:
            import plotly.graph_objs as go
            import plotly.express as px
            
            # Crear visualizaciones
            fig1 = px.histogram(df_clean, x='edad', title='Distribución de Edades')
            fig2 = px.box(df_clean, y=['satisfaccion_servicio', 'satisfaccion_atencion', 'satisfaccion_precio'], 
                         title='Satisfacción por Dimensiones')
            
            visualization_info = {
                "charts_created": np.int64(2),
                "chart_types": ["histogram", "boxplot"],
                "figure_info": {
                    "histogram": {
                        "type": "plotly_figure",
                        "title": "Distribución de Edades",
                        "data_points": len(df_clean)
                    },
                    "boxplot": {
                        "type": "plotly_figure", 
                        "title": "Satisfacción por Dimensiones",
                        "variables": 3
                    }
                }
            }
            
            logger.log_event(
                level=LogLevel.INFO,
                message="Visualizaciones generadas exitosamente",
                module="visualization_generator",
                function="create_survey_charts",
                step="visualization_generation",
                category=LogCategory.VISUALIZATION.value,
                parameters={"chart_types": ["histogram", "boxplot"]},
                before_metrics={"data_points": len(df_clean)},
                after_metrics=visualization_info,
                execution_time=0.35,
                tags=["visualization", "charts"],
                metadata=visualization_info
            )
            
        except ImportError:
            logger.log_event(
                level=LogLevel.WARNING,
                message="Plotly no disponible, saltando visualizaciones",
                module="visualization_generator",
                function="create_survey_charts",
                step="visualization_generation",
                category=LogCategory.VISUALIZATION.value,
                parameters={"chart_types": "none"},
                before_metrics={"data_points": len(df_clean)},
                after_metrics={"charts_created": 0},
                execution_time=0.01,
                tags=["visualization", "skipped"],
                metadata={"reason": "plotly_not_installed"}
            )
        
        # 5. Generación de reporte
        print("📋 Generando reporte final...")
        
        report_data = {
            "resumen_ejecutivo": {
                "total_encuestas": np.int64(len(df_clean)),
                "satisfaccion_promedio": np.float64(stats['resumen_general']['satisfaccion_promedio']),
                "tendencia": "Positiva" if stats['resumen_general']['satisfaccion_promedio'] > 7 else "Neutral",
                "recomendacion_rate": np.float64(len(df_clean[df_clean['recomendaria'] == 'Sí']) / len(df_clean) * 100)
            },
            "hallazgos_clave": [
                "La satisfacción promedio es de {:.1f}/10".format(stats['resumen_general']['satisfaccion_promedio']),
                "El {}% de los encuestados recomendaría el servicio".format(
                    len(df_clean[df_clean['recomendaria'] == 'Sí']) / len(df_clean) * 100
                ),
                "La edad promedio de los encuestados es {:.1f} años".format(stats['resumen_general']['promedio_edad'])
            ],
            "metricas_detalladas": stats
        }
        
        # Calcular tamaño del reporte serializado
        report_data_serialized = serialize_for_json(report_data)
        report_size_kb = len(json.dumps(report_data_serialized)) / 1024
        
        logger.log_event(
            level=LogLevel.INFO,
            message="Reporte final generado exitosamente",
            module="report_generator",
            function="generate_survey_report",
            step="report_generation",
            category=LogCategory.EXPORT.value,
            parameters={"report_type": "survey_analysis"},
            before_metrics={"data_size": len(df_clean)},
            after_metrics={
                "report_sections": len(report_data),
                "report_size_kb": report_size_kb
            },
            execution_time=0.20,
            tags=["report", "survey"],
            metadata=report_data
        )
        
        # 6. Pipeline completado
        logger.log_event(
            level=LogLevel.INFO,
            message="Pipeline de análisis de encuestas completado exitosamente",
            module="pipeline_orchestrator",
            function="run_survey_pipeline",
            step="pipeline_complete",
            category=LogCategory.SYSTEM.value,
            parameters={"pipeline_type": "survey_analysis"},
            before_metrics={"start_time": datetime.now().isoformat()},
            after_metrics={
                "end_time": datetime.now().isoformat(),
                "total_steps": np.int64(5),
                "successful_steps": np.int64(5),
                "failed_steps": np.int64(0),
                "total_execution_time": np.float64(1.40)
            },
            execution_time=1.40,
            tags=["pipeline", "complete", "success"],
            metadata={"pipeline_summary": "Análisis completo de encuesta de satisfacción"}
        )
        
        print("✅ Pipeline completado exitosamente!")
        
        # Verificar y mostrar logs
        verificar_logs_completos(log_file, report_data)
        
        return log_file, report_data
        
    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    finally:
        # No limpiar archivos para poder revisarlos
        pass


def verificar_logs_completos(log_file, report_data):
    """Verificar que los logs son válidos y mostrar estadísticas completas"""
    
    print(f"\n📋 Verificando logs completos en: {log_file}")
    
    if not os.path.exists(log_file):
        print("❌ Archivo de log no encontrado")
        return
    
    # Leer logs
    with open(log_file, 'r', encoding='utf-8') as f:
        log_lines = f.readlines()
    
    print(f"📄 Total de líneas de log: {len(log_lines)}")
    
    # Validar cada línea
    valid_logs = 0
    log_types = {}
    log_levels = {}
    total_execution_time = 0
    
    for i, line in enumerate(log_lines):
        line = line.strip()
        if line:
            try:
                log_event = json.loads(line)
                valid_logs += 1
                
                # Contar tipos de eventos
                step = log_event.get('step', 'unknown')
                log_types[step] = log_types.get(step, 0) + 1
                
                # Contar niveles
                level = log_event.get('level', 'unknown')
                log_levels[level] = log_levels.get(level, 0) + 1
                
                # Sumar tiempo de ejecución
                execution_time = log_event.get('execution_time', 0)
                if isinstance(execution_time, (int, float)):
                    total_execution_time += execution_time
                
            except json.JSONDecodeError as e:
                print(f"❌ Error en línea {i+1}: {e}")
    
    print(f"✅ Logs válidos: {valid_logs}/{len(log_lines)}")
    print(f"⏱️ Tiempo total de ejecución: {total_execution_time:.3f} segundos")
    
    print(f"\n📊 Distribución por pasos:")
    for step, count in log_types.items():
        print(f"  {step}: {count} eventos")
    
    print(f"\n📊 Distribución por niveles:")
    for level, count in log_levels.items():
        print(f"  {level}: {count} eventos")
    
    # Mostrar resumen del reporte
    print(f"\n📋 Resumen del Reporte:")
    if report_data:
        resumen = report_data.get('resumen_ejecutivo', {})
        print(f"  Total encuestas: {resumen.get('total_encuestas', 'N/A')}")
        print(f"  Satisfacción promedio: {resumen.get('satisfaccion_promedio', 'N/A'):.2f}/10")
        print(f"  Tendencia: {resumen.get('tendencia', 'N/A')}")
        print(f"  Tasa de recomendación: {resumen.get('recomendacion_rate', 'N/A'):.1f}%")
    
    print(f"\n🎉 ¡Sistema completo funcionando correctamente!")
    print(f"💡 Los logs están listos para dashboards de monitoreo")
    print(f"📊 El reporte está listo para la UI")


if __name__ == "__main__":
    print("🚀 Sistema Completo con Serialización Robusta")
    print("=" * 60)
    
    # Ejecutar ejemplo completo
    log_file, report_data = simular_pipeline_completo()
    
    if log_file and report_data:
        print(f"\n📁 Archivos creados:")
        print(f"  - Log: {log_file}")
        print(f"  - Reporte: Generado en memoria")
        print("💡 Puedes abrir el log en cualquier dashboard de monitoreo")
    else:
        print("\n❌ No se pudo completar el ejemplo") 