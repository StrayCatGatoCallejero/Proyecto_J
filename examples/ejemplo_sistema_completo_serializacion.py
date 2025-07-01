"""
Ejemplo Completo del Sistema de Logging JSON con Serialización Robusta
=====================================================================

Este ejemplo demuestra el sistema completo funcionando con:
- JsonLogger con serialización robusta
- Datos complejos (numpy, pandas, plotly)
- Logs estructurados y válidos
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# Importar solo el JsonLogger
from processing.json_logging import JsonLogger, LogLevel, LogCategory


def crear_datos_ejemplo():
    """Crear datos de ejemplo con tipos complejos"""
    np.random.seed(42)
    
    # DataFrame con tipos mixtos
    data = {
        'edad': np.random.randint(18, 80, 100),
        'ingresos': np.random.normal(50000, 15000, 100),
        'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universidad'], 100),
        'genero': np.random.choice(['M', 'F'], 100),
        'satisfaccion': np.random.randint(1, 11, 100),
        'fecha_registro': pd.date_range('2023-01-01', periods=100, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Estadísticas con numpy
    stats = {
        'media_edad': np.float64(df['edad'].mean()),
        'std_ingresos': np.float64(df['ingresos'].std()),
        'percentiles': np.array([25, 50, 75]),
        'correlacion': np.array([[1.0, 0.5], [0.5, 1.0]]),
        'conteo_educacion': df['educacion'].value_counts().to_dict()
    }
    
    return df, stats


def simular_pipeline_completo():
    """Simular un pipeline completo con logging detallado"""
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, 'pipeline_complete.json')
    
    try:
        print("🚀 Iniciando Pipeline Completo con Logging JSON...")
        
        # Crear logger
        logger = JsonLogger(
            file_path=log_file,
            level='INFO',
            session_id=f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            console_output=True
        )
        
        # Crear datos
        df, stats = crear_datos_ejemplo()
        
        print(f"📊 Datos creados: {len(df)} filas, {len(df.columns)} columnas")
        
        # Simular pasos del pipeline con logging detallado
        
        # 1. Carga de datos
        logger.log_event(
            level=LogLevel.INFO,
            message="Iniciando carga de datos",
            module="data_loader",
            function="load_data",
            step="data_loading",
            category=LogCategory.DATA_LOAD.value,
            parameters={"source": "memory", "format": "dataframe"},
            before_metrics={"expected_rows": 100},
            after_metrics={
                "actual_rows": np.int64(len(df)),
                "actual_columns": np.int64(len(df.columns)),
                "memory_usage_mb": np.float64(df.memory_usage(deep=True).sum() / (1024 * 1024)),
                "data_types": df.dtypes.value_counts().to_dict()
            },
            execution_time=0.15,
            tags=["data_load", "memory"],
            metadata={"dataframe_info": str(df.info())}
        )
        
        # 2. Validación de esquema
        logger.log_event(
            level=LogLevel.INFO,
            message="Validando esquema de datos",
            module="schema_validator",
            function="validate_schema",
            step="schema_validation",
            category=LogCategory.VALIDATION.value,
            parameters={"schema_version": "1.0"},
            before_metrics={"columns_to_validate": len(df.columns)},
            after_metrics={
                "valid_columns": np.int64(len(df.columns)),
                "invalid_columns": np.int64(0),
                "validation_score": np.float64(1.0)
            },
            execution_time=0.08,
            tags=["validation", "schema"],
            metadata={"column_types": df.dtypes.to_dict()}
        )
        
        # 3. Análisis estadístico
        logger.log_event(
            level=LogLevel.INFO,
            message="Ejecutando análisis estadístico",
            module="statistical_analyzer",
            function="analyze_data",
            step="statistical_analysis",
            category=LogCategory.ANALYSIS.value,
            parameters={"methods": ["descriptive", "correlation"]},
            before_metrics={"input_size": len(df)},
            after_metrics={
                "output_size": len(stats),
                "analysis_time": np.float64(0.25),
                "statistics_generated": list(stats.keys())
            },
            execution_time=0.25,
            tags=["analysis", "statistics"],
            metadata={"statistics": stats}
        )
        
        # 4. Generación de visualizaciones
        try:
            import plotly.graph_objs as go
            import plotly.express as px
            
            # Crear figura de Plotly
            fig = px.histogram(df, x='edad', title='Distribución de Edades')
            
            logger.log_event(
                level=LogLevel.INFO,
                message="Generando visualizaciones",
                module="visualization_generator",
                function="create_visualizations",
                step="visualization_generation",
                category=LogCategory.VISUALIZATION.value,
                parameters={"chart_type": "histogram", "variable": "edad"},
                before_metrics={"data_points": len(df)},
                after_metrics={
                    "charts_created": np.int64(1),
                    "chart_types": ["histogram"],
                    "figure_size": "800x600"
                },
                execution_time=0.12,
                tags=["visualization", "plotly"],
                metadata={
                    "figure_info": {
                        "type": "plotly_figure",
                        "layout_title": fig.layout.title.text if fig.layout.title else None,
                        "data_count": len(fig.data),
                        "data_types": [trace.type for trace in fig.data] if fig.data else []
                    }
                }
            )
            
        except ImportError:
            logger.log_event(
                level=LogLevel.WARNING,
                message="Plotly no disponible, saltando visualizaciones",
                module="visualization_generator",
                function="create_visualizations",
                step="visualization_generation",
                category=LogCategory.VISUALIZATION.value,
                parameters={"chart_type": "none"},
                before_metrics={"data_points": len(df)},
                after_metrics={"charts_created": 0},
                execution_time=0.01,
                tags=["visualization", "skipped"],
                metadata={"reason": "plotly_not_installed"}
            )
        
        # 5. Exportación de resultados
        logger.log_event(
            level=LogLevel.INFO,
            message="Exportando resultados",
            module="exporter",
            function="export_results",
            step="export",
            category=LogCategory.EXPORT.value,
            parameters={"formats": ["json", "csv"]},
            before_metrics={"data_size": len(df)},
            after_metrics={
                "files_created": np.int64(2),
                "total_size_mb": np.float64(0.05),
                "export_formats": ["json", "csv"]
            },
            execution_time=0.18,
            tags=["export", "results"],
            metadata={"export_path": temp_dir}
        )
        
        # 6. Pipeline completado
        logger.log_event(
            level=LogLevel.INFO,
            message="Pipeline completado exitosamente",
            module="pipeline_orchestrator",
            function="run_pipeline",
            step="pipeline_complete",
            category=LogCategory.SYSTEM.value,
            parameters={"pipeline_version": "1.0"},
            before_metrics={"start_time": datetime.now().isoformat()},
            after_metrics={
                "end_time": datetime.now().isoformat(),
                "total_steps": np.int64(5),
                "successful_steps": np.int64(5),
                "failed_steps": np.int64(0)
            },
            execution_time=0.68,
            tags=["pipeline", "complete", "success"],
            metadata={"pipeline_summary": "Todos los pasos completados exitosamente"}
        )
        
        print("✅ Pipeline completado!")
        
        # Verificar y mostrar logs
        verificar_logs(log_file)
        
        return log_file
        
    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Limpiar archivos temporales (opcional)
        # shutil.rmtree(temp_dir)
        pass


def verificar_logs(log_file):
    """Verificar que los logs son válidos y mostrar estadísticas"""
    
    print(f"\n📋 Verificando logs en: {log_file}")
    
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
    
    # Mostrar ejemplo de log
    if log_lines:
        print(f"\n📝 Ejemplo de log (primeros 200 caracteres):")
        example = log_lines[0][:200] + "..." if len(log_lines[0]) > 200 else log_lines[0]
        print(f"  {example}")
    
    print(f"\n🎉 ¡Sistema de logging JSON funcionando correctamente!")
    print(f"💡 Los logs están listos para ser consumidos por dashboards como Kibana, Grafana, etc.")


if __name__ == "__main__":
    print("🚀 Sistema de Logging JSON con Serialización Robusta")
    print("=" * 60)
    
    # Ejecutar ejemplo completo
    log_file = simular_pipeline_completo()
    
    if log_file:
        print(f"\n📁 Archivo de log creado: {log_file}")
        print("💡 Puedes abrir este archivo en cualquier editor de JSON o dashboard")
    else:
        print("\n❌ No se pudo crear el archivo de log") 