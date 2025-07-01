"""
Ejemplo de Integración del Logging JSON con el Pipeline Existente
================================================================

Este ejemplo muestra cómo integrar el sistema de logging JSON avanzado
con el pipeline de análisis existente, manteniendo compatibilidad
con el sistema actual.
"""

import pandas as pd
import time
from typing import Dict, Any, Optional
from datetime import datetime

# Importar componentes del pipeline existente
from processing.json_logging import (
    JSONLogger,
    LogLevel,
    LogCategory,
    create_json_logger
)
from processing.business_rules import validate_business_rules
from processing.data_validators import validate_data_integrity
from processing.features import extract_features
from processing.stats import calculate_statistics


class PipelineWithJSONLogging:
    """
    Pipeline que integra el logging JSON con el sistema existente.
    
    Esta clase demuestra cómo mantener la funcionalidad existente
    mientras se agrega logging JSON estructurado.
    """
    
    def __init__(self, config_path: str = "config/config.yml"):
        """
        Inicializa el pipeline con logging JSON.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.json_logger = create_json_logger(config_path)
        self.session_id = self.json_logger.session_id
        
        # Log de inicialización
        self.json_logger.log_system_event(
            level=LogLevel.INFO,
            message="Pipeline initialized with JSON logging",
            metadata={"config_path": config_path}
        )
        
        print(f"🚀 Pipeline inicializado con session_id: {self.session_id}")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga datos con logging JSON detallado.
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            DataFrame cargado
        """
        start_time = time.time()
        
        try:
            # Simular carga de datos
            print(f"📁 Cargando datos desde: {file_path}")
            
            # Crear datos de ejemplo (en un caso real, cargaríamos el archivo)
            df = pd.DataFrame({
                'edad': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
                'genero': ['Masculino', 'Femenino', 'Masculino', 'Femenino', 'Masculino', 
                          'Femenino', 'Masculino', 'Femenino', 'Masculino', 'Femenino'],
                'nivel_educacion': ['Primaria', 'Secundaria', 'Universitaria', 'Postgrado', 'Doctorado',
                                   'Primaria', 'Secundaria', 'Universitaria', 'Postgrado', 'Doctorado'],
                'ingresos': [500000, 800000, 1200000, 1500000, 2000000, 
                            600000, 900000, 1300000, 1600000, 2200000],
                'region': ['Metropolitana', 'Valparaíso', 'Antofagasta', 'Tarapacá', 'Atacama',
                          'Metropolitana', 'Valparaíso', 'Antofagasta', 'Tarapacá', 'Atacama'],
                'pregunta_1_likert': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                'pregunta_2_likert': [2, 3, 4, 5, 1, 2, 3, 4, 5, 1]
            })
            
            execution_time = time.time() - start_time
            
            # Log del evento de carga
            self.json_logger.log_data_load(
                function="load_data",
                file_path=file_path,
                file_size=1024 * 50,  # Simulado
                rows=len(df),
                columns=len(df.columns),
                execution_time=execution_time,
                success=True
            )
            
            print(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
            return df
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log del error
            self.json_logger.log_error(
                function="load_data",
                error=e,
                context="data_loading",
                execution_time=execution_time,
                additional_data={"file_path": file_path}
            )
            
            print(f"❌ Error cargando datos: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida datos con logging JSON detallado.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Resultados de validación
        """
        start_time = time.time()
        
        try:
            print(f"🔍 Validando datos...")
            
            # Ejecutar validaciones
            validation_results = validate_data_integrity(df)
            
            execution_time = time.time() - start_time
            
            # Contar resultados
            total_checks = len(validation_results)
            passed_checks = sum(1 for result in validation_results.values() if result.get('is_valid', False))
            failed_checks = total_checks - passed_checks
            
            # Log del evento de validación
            self.json_logger.log_validation(
                function="validate_data",
                validation_type="data_integrity",
                total_checks=total_checks,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                execution_time=execution_time,
                details=validation_results
            )
            
            print(f"✅ Validación completada: {passed_checks}/{total_checks} checks pasaron")
            return validation_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log del error
            self.json_logger.log_error(
                function="validate_data",
                error=e,
                context="data_validation",
                execution_time=execution_time,
                additional_data={"data_shape": df.shape}
            )
            
            print(f"❌ Error en validación: {e}")
            raise
    
    def validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida reglas de negocio con logging JSON detallado.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Resultados de validación de reglas de negocio
        """
        start_time = time.time()
        
        try:
            print(f"📋 Validando reglas de negocio...")
            
            # Ejecutar validaciones de reglas de negocio
            metadata = {'dataset_type': 'social_sciences'}
            validation_results = validate_business_rules(df, metadata)
            
            execution_time = time.time() - start_time
            
            # Contar resultados
            total_rules = len(validation_results)
            failed_rules = sum(1 for r in validation_results if not r.is_valid)
            warning_rules = len([r for r in validation_results if r.details.get('alertas_generadas', 0) > 0])
            
            # Log del evento de validación de reglas de negocio
            self.json_logger.log_business_rules(
                function="validate_business_rules",
                rules_executed=total_rules,
                rules_failed=failed_rules,
                rules_warnings=warning_rules,
                execution_time=execution_time,
                details={
                    "validation_results": [r.rule_name for r in validation_results],
                    "failed_rules": [r.rule_name for r in validation_results if not r.is_valid],
                    "warning_rules": [r.rule_name for r in validation_results if r.details.get('alertas_generadas', 0) > 0]
                }
            )
            
            print(f"✅ Reglas de negocio validadas: {total_rules - failed_rules}/{total_rules} pasaron")
            return {"results": validation_results, "summary": {"total": total_rules, "failed": failed_rules, "warnings": warning_rules}}
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log del error
            self.json_logger.log_error(
                function="validate_business_rules",
                error=e,
                context="business_rules_validation",
                execution_time=execution_time,
                additional_data={"data_shape": df.shape}
            )
            
            print(f"❌ Error en validación de reglas de negocio: {e}")
            raise
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae características con logging JSON detallado.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame con características extraídas
        """
        start_time = time.time()
        
        try:
            print(f"🔧 Extrayendo características...")
            
            # Extraer características
            df_features = extract_features(df)
            
            execution_time = time.time() - start_time
            
            # Log del evento de extracción de características
            self.json_logger.log_analysis(
                function="extract_features",
                analysis_type="feature_extraction",
                input_size=len(df.columns),
                output_size=len(df_features.columns),
                execution_time=execution_time,
                success=True,
                results={
                    "original_features": len(df.columns),
                    "new_features": len(df_features.columns) - len(df.columns),
                    "total_features": len(df_features.columns)
                }
            )
            
            print(f"✅ Características extraídas: {len(df_features.columns)} características totales")
            return df_features
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log del error
            self.json_logger.log_error(
                function="extract_features",
                error=e,
                context="feature_extraction",
                execution_time=execution_time,
                additional_data={"input_features": len(df.columns)}
            )
            
            print(f"❌ Error extrayendo características: {e}")
            raise
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcula estadísticas con logging JSON detallado.
        
        Args:
            df: DataFrame para análisis
            
        Returns:
            Estadísticas calculadas
        """
        start_time = time.time()
        
        try:
            print(f"📊 Calculando estadísticas...")
            
            # Calcular estadísticas
            stats = calculate_statistics(df)
            
            execution_time = time.time() - start_time
            
            # Log del evento de cálculo de estadísticas
            self.json_logger.log_analysis(
                function="calculate_statistics",
                analysis_type="descriptive_statistics",
                input_size=len(df),
                output_size=len(stats),
                execution_time=execution_time,
                success=True,
                results={
                    "variables_analyzed": len(stats),
                    "statistics_types": list(stats.keys()) if stats else []
                }
            )
            
            print(f"✅ Estadísticas calculadas para {len(stats)} variables")
            return stats
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log del error
            self.json_logger.log_error(
                function="calculate_statistics",
                error=e,
                context="statistics_calculation",
                execution_time=execution_time,
                additional_data={"data_shape": df.shape}
            )
            
            print(f"❌ Error calculando estadísticas: {e}")
            raise
    
    def run_pipeline(self, file_path: str) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo con logging JSON.
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            Resultados del pipeline
        """
        print(f"\n🚀 INICIANDO PIPELINE CON LOGGING JSON")
        print(f"Session ID: {self.session_id}")
        print("="*60)
        
        results = {}
        
        try:
            # Paso 1: Cargar datos
            df = self.load_data(file_path)
            results['data'] = df
            
            # Paso 2: Validar datos
            validation_results = self.validate_data(df)
            results['validation'] = validation_results
            
            # Paso 3: Validar reglas de negocio
            business_rules_results = self.validate_business_rules(df)
            results['business_rules'] = business_rules_results
            
            # Paso 4: Extraer características
            df_features = self.extract_features(df)
            results['features'] = df_features
            
            # Paso 5: Calcular estadísticas
            stats = self.calculate_statistics(df_features)
            results['statistics'] = stats
            
            # Log de finalización exitosa
            self.json_logger.log_system_event(
                level=LogLevel.INFO,
                message="Pipeline completed successfully",
                metadata={
                    "total_steps": 5,
                    "final_data_shape": df_features.shape,
                    "statistics_count": len(stats)
                }
            )
            
            print(f"\n✅ PIPELINE COMPLETADO EXITOSAMENTE")
            
        except Exception as e:
            # Log de error en el pipeline
            self.json_logger.log_error(
                function="run_pipeline",
                error=e,
                context="pipeline_execution",
                execution_time=0.0,
                additional_data={"file_path": file_path}
            )
            
            print(f"\n❌ ERROR EN EL PIPELINE: {e}")
            raise
        
        return results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del pipeline con todos los logs.
        
        Returns:
            Resumen del pipeline
        """
        summary = self.json_logger.get_session_summary()
        
        # Agregar información específica del pipeline
        pipeline_logs = self.json_logger.get_logs_by_category(LogCategory.ANALYSIS)
        validation_logs = self.json_logger.get_logs_by_category(LogCategory.VALIDATION)
        error_logs = self.json_logger.get_error_logs()
        
        summary.update({
            "pipeline_analysis_steps": len(pipeline_logs),
            "pipeline_validation_steps": len(validation_logs),
            "pipeline_errors": len(error_logs),
            "session_id": self.session_id
        })
        
        return summary
    
    def export_pipeline_logs(self, format: str = "json") -> str:
        """
        Exporta los logs del pipeline.
        
        Args:
            format: Formato de exportación
            
        Returns:
            Ruta del archivo exportado
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pipeline_logs_{self.session_id}_{timestamp}.{format}"
        
        return self.json_logger.export_session_logs(format, filename)


def ejemplo_pipeline_completo():
    """
    Ejemplo de ejecución del pipeline completo con logging JSON.
    """
    print("🚀 EJEMPLO DE PIPELINE COMPLETO CON LOGGING JSON")
    print("="*80)
    
    # Crear pipeline
    pipeline = PipelineWithJSONLogging()
    
    try:
        # Ejecutar pipeline
        results = pipeline.run_pipeline("datos_ejemplo.csv")
        
        # Mostrar resultados
        print(f"\n📊 RESULTADOS DEL PIPELINE:")
        print(f"   - Datos cargados: {results['data'].shape}")
        print(f"   - Características extraídas: {results['features'].shape}")
        print(f"   - Estadísticas calculadas: {len(results['statistics'])}")
        
        # Obtener resumen
        summary = pipeline.get_pipeline_summary()
        print(f"\n📈 RESUMEN DEL PIPELINE:")
        print(f"   - Session ID: {summary['session_id']}")
        print(f"   - Total de logs: {summary['total_logs']}")
        print(f"   - Pasos de análisis: {summary['pipeline_analysis_steps']}")
        print(f"   - Pasos de validación: {summary['pipeline_validation_steps']}")
        print(f"   - Errores: {summary['pipeline_errors']}")
        print(f"   - Tiempo total: {summary['total_execution_time']:.3f}s")
        
        # Exportar logs
        export_path = pipeline.export_pipeline_logs()
        print(f"\n💾 Logs exportados a: {export_path}")
        
        return pipeline, results
        
    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        return pipeline, None


def ejemplo_multiple_pipelines():
    """
    Ejemplo de múltiples pipelines ejecutándose en paralelo.
    """
    print("\n🔄 EJEMPLO DE MÚLTIPLES PIPELINES")
    print("="*60)
    
    from processing.json_logging import get_json_logger_manager
    
    # Crear gestor de logging
    manager = get_json_logger_manager()
    
    # Crear múltiples pipelines
    pipelines = []
    for i in range(3):
        session_id = f"pipeline_{i+1}"
        logger = manager.create_session(session_id)
        pipeline = PipelineWithJSONLogging()
        pipeline.json_logger = logger
        pipeline.session_id = session_id
        pipelines.append(pipeline)
    
    print(f"📝 Pipelines creados: {len(pipelines)}")
    
    # Ejecutar pipelines (simulado)
    for i, pipeline in enumerate(pipelines):
        print(f"\n🔄 Ejecutando Pipeline {i+1}...")
        
        try:
            # Simular ejecución
            pipeline.json_logger.log_system_event(
                level=LogLevel.INFO,
                message=f"Pipeline {i+1} started",
                metadata={"pipeline_number": i+1}
            )
            
            # Simular pasos del pipeline
            for step in ["load", "validate", "analyze", "export"]:
                pipeline.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Step {step} completed",
                    module="pipeline",
                    function=f"step_{step}",
                    step=step,
                    category=LogCategory.PROCESSING,
                    parameters={"step": step, "pipeline": i+1},
                    before_metrics={"step": step},
                    after_metrics={"status": "completed"},
                    execution_time=0.1,
                    tags=["pipeline", f"pipeline_{i+1}", step]
                )
            
            pipeline.json_logger.log_system_event(
                level=LogLevel.INFO,
                message=f"Pipeline {i+1} completed",
                metadata={"pipeline_number": i+1, "status": "success"}
            )
            
            print(f"✅ Pipeline {i+1} completado")
            
        except Exception as e:
            pipeline.json_logger.log_error(
                function=f"pipeline_{i+1}",
                error=e,
                context="pipeline_execution",
                execution_time=0.0
            )
            print(f"❌ Error en Pipeline {i+1}: {e}")
    
    # Obtener resumen de todas las sesiones
    all_summaries = manager.get_all_sessions_summary()
    print(f"\n📊 RESUMEN DE TODAS LAS SESIONES:")
    print(f"   - Total de sesiones: {all_summaries['total_sessions']}")
    
    for session_id, summary in all_summaries['session_summaries'].items():
        print(f"   - {session_id}: {summary['total_logs']} logs, {summary['error_count']} errores")
    
    # Exportar todas las sesiones
    export_path = manager.export_all_sessions()
    print(f"\n💾 Todas las sesiones exportadas a: {export_path}")
    
    return manager


def main():
    """
    Función principal que ejecuta los ejemplos de integración.
    """
    print("🚀 EJEMPLO DE INTEGRACIÓN DE LOGGING JSON")
    print("="*80)
    
    # Ejemplo 1: Pipeline completo
    pipeline, results = ejemplo_pipeline_completo()
    
    # Ejemplo 2: Múltiples pipelines
    manager = ejemplo_multiple_pipelines()
    
    print("\n" + "="*80)
    print("✅ INTEGRACIÓN COMPLETADA")
    print("="*80)
    
    print("\n💡 BENEFICIOS DEL LOGGING JSON:")
    print("✅ Trazabilidad completa con session_id")
    print("✅ Formato estructurado para sistemas de monitoreo")
    print("✅ Métricas detalladas de rendimiento")
    print("✅ Integración con ELK, Datadog, Prometheus")
    print("✅ Análisis de errores y debugging")
    print("✅ Auditoría completa del pipeline")
    
    print("\n🔧 PRÓXIMOS PASOS:")
    print("1. Integrar en app_front.py y wizard_streamlit.py")
    print("2. Configurar sistemas de monitoreo")
    print("3. Implementar alertas automáticas")
    print("4. Crear dashboards de métricas")
    print("5. Optimizar rendimiento basado en logs")


if __name__ == "__main__":
    main() 