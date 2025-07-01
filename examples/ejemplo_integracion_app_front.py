"""
Ejemplo de Integración del Sistema de Manejo de Errores en app_front.py
======================================================================

Este ejemplo muestra cómo integrar el sistema avanzado de manejo de errores
de negocio en la aplicación Streamlit existente.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
from processing.business_error_handler import (
    BusinessErrorHandler,
    display_business_errors_in_streamlit,
    get_business_errors,
    BusinessError
)
from processing.business_rules import (
    validate_business_rules,
    ValidationResult,
    BusinessRuleError
)
from processing.logging import get_logger
from orchestrator.pipeline_orchestrator import PipelineOrchestrator, SessionData


def integrar_manejo_errores_en_carga_datos():
    """
    Ejemplo de cómo integrar el manejo de errores en la sección de carga de datos.
    """
    
    st.header("📁 Carga de Datos con Validación de Negocio")
    
    # Widget de carga de archivo
    uploaded_file = st.file_uploader(
        "Selecciona tu archivo de datos",
        type=['csv', 'xlsx', 'xls'],
        help="Soporta archivos CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        try:
            # Cargar datos
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Archivo cargado: {uploaded_file.name}")
            st.info(f"📊 Datos: {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Mostrar vista previa
            with st.expander("👀 Vista previa de los datos"):
                st.dataframe(df.head(), use_container_width=True)
            
            # ============================================================================
            # VALIDACIÓN DE REGLAS DE NEGOCIO
            # ============================================================================
            
            st.subheader("🔍 Validación de Reglas de Negocio")
            
            # Crear metadata básica
            metadata = {
                'dataset_type': 'social_sciences',
                'dataset_name': uploaded_file.name,
                'columns': list(df.columns),
                'file_size': uploaded_file.size
            }
            
            # Ejecutar validaciones de negocio
            with st.spinner("Validando reglas de negocio..."):
                try:
                    validation_results = validate_business_rules(df, metadata)
                    
                    # Contar resultados
                    total_errors = sum(1 for r in validation_results if not r.is_valid)
                    total_warnings = len([r for r in validation_results if r.details.get('alertas_generadas', 0) > 0])
                    
                    # Mostrar resumen de validación
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Reglas Ejecutadas", len(validation_results))
                    with col2:
                        st.metric("Errores Críticos", total_errors, delta_color="inverse")
                    with col3:
                        st.metric("Advertencias", total_warnings)
                    with col4:
                        success_rate = ((len(validation_results) - total_errors) / len(validation_results)) * 100
                        st.metric("Tasa de Éxito", f"{success_rate:.1f}%")
                    
                    # Si hay errores, mostrar el sistema avanzado de manejo
                    if total_errors > 0 or total_warnings > 0:
                        st.warning("⚠️ Se detectaron inconsistencias en los datos de negocio.")
                        
                        # Usar el sistema avanzado de manejo de errores
                        display_business_errors_in_streamlit(
                            logs=[],  # Los logs se pueden obtener del logger si es necesario
                            validation_results=validation_results
                        )
                        
                        # Preguntar al usuario si quiere continuar
                        st.subheader("🤔 ¿Qué quieres hacer?")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("🔄 Corregir Datos", type="primary"):
                                st.info("Funcionalidad de corrección automática en desarrollo...")
                        
                        with col2:
                            if st.button("⚠️ Continuar con Advertencias"):
                                st.success("Continuando con el análisis...")
                                # Aquí se continuaría con el análisis normal
                        
                        with col3:
                            if st.button("❌ Cancelar"):
                                st.stop()
                    
                    else:
                        st.success("✅ Todas las validaciones de negocio pasaron exitosamente!")
                        
                except Exception as e:
                    st.error(f"❌ Error en la validación: {str(e)}")
                    st.stop()
            
            # Guardar datos en session state
            st.session_state["df"] = df
            st.session_state["metadata"] = metadata
            st.session_state["validation_results"] = validation_results
            
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")
            st.stop()


def integrar_manejo_errores_en_analisis():
    """
    Ejemplo de cómo integrar el manejo de errores en la sección de análisis.
    """
    
    st.header("📊 Análisis con Monitoreo de Errores")
    
    # Verificar que hay datos cargados
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ Primero debes cargar datos en la sección anterior.")
        return
    
    df = st.session_state["df"]
    metadata = st.session_state.get("metadata", {})
    
    # Selector de tipo de análisis
    tipo_analisis = st.selectbox(
        "Selecciona el tipo de análisis:",
        [
            "📈 Análisis Descriptivo",
            "🔗 Análisis de Correlación", 
            "📊 Análisis de Contingencia",
            "🎯 Análisis de Regresión",
            "👥 Análisis de Clusters"
        ]
    )
    
    # Ejecutar análisis con validación continua
    if st.button("🚀 Ejecutar Análisis", type="primary"):
        
        # Crear handler para monitorear errores durante el análisis
        error_handler = BusinessErrorHandler()
        
        with st.spinner("Ejecutando análisis..."):
            try:
                # Ejecutar análisis según el tipo seleccionado
                if tipo_analisis == "📈 Análisis Descriptivo":
                    resultado = ejecutar_analisis_descriptivo(df, error_handler)
                elif tipo_analisis == "🔗 Análisis de Correlación":
                    resultado = ejecutar_analisis_correlacion(df, error_handler)
                elif tipo_analisis == "📊 Análisis de Contingencia":
                    resultado = ejecutar_analisis_contingencia(df, error_handler)
                elif tipo_analisis == "🎯 Análisis de Regresión":
                    resultado = ejecutar_analisis_regresion(df, error_handler)
                elif tipo_analisis == "👥 Análisis de Clusters":
                    resultado = ejecutar_analisis_clusters(df, error_handler)
                
                # Mostrar resultados del análisis
                if resultado.get('success', False):
                    st.success("✅ Análisis completado exitosamente!")
                    
                    # Mostrar resultados específicos
                    mostrar_resultados_analisis(resultado)
                    
                    # Verificar si se generaron nuevos errores durante el análisis
                    if error_handler.errors:
                        st.warning("⚠️ Se detectaron nuevos errores durante el análisis:")
                        error_handler.display_errors_in_streamlit(show_details=True, show_charts=False)
                    
                else:
                    st.error(f"❌ Error en el análisis: {resultado.get('error', 'Error desconocido')}")
                    
            except Exception as e:
                st.error(f"❌ Error inesperado: {str(e)}")


def ejecutar_analisis_descriptivo(df: pd.DataFrame, error_handler: BusinessErrorHandler) -> Dict[str, Any]:
    """Ejecuta análisis descriptivo con monitoreo de errores."""
    
    try:
        # Análisis básico
        descripcion = df.describe()
        
        # Detectar posibles problemas
        problemas = []
        
        # Verificar valores faltantes
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            problemas.append(f"Valores faltantes detectados: {missing_data.sum()} total")
        
        # Verificar outliers en columnas numéricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        for col in columnas_numericas:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                problemas.append(f"Outliers en {col}: {len(outliers)} valores")
        
        # Crear errores de negocio si se detectan problemas
        for problema in problemas:
            error = BusinessError(
                rule_name="analisis_descriptivo",
                error_type="data_quality_issue",
                message=problema,
                severity="warning",
                context={'analysis_type': 'descriptive'}
            )
            error_handler.errors.append(error)
        
        return {
            'success': True,
            'description': descripcion,
            'missing_data': missing_data,
            'problems': problemas
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def ejecutar_analisis_correlacion(df: pd.DataFrame, error_handler: BusinessErrorHandler) -> Dict[str, Any]:
    """Ejecuta análisis de correlación con monitoreo de errores."""
    
    try:
        # Seleccionar columnas numéricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        
        if len(columnas_numericas) < 2:
            error = BusinessError(
                rule_name="analisis_correlacion",
                error_type="insufficient_data",
                message="Se requieren al menos 2 columnas numéricas para análisis de correlación",
                severity="error"
            )
            error_handler.errors.append(error)
            return {
                'success': False,
                'error': 'Datos insuficientes para análisis de correlación'
            }
        
        # Calcular correlación
        correlacion = df[columnas_numericas].corr()
        
        # Detectar correlaciones muy altas (posible multicolinealidad)
        correlaciones_altas = []
        for i in range(len(correlacion.columns)):
            for j in range(i+1, len(correlacion.columns)):
                valor = correlacion.iloc[i, j]
                if abs(valor) > 0.9:
                    correlaciones_altas.append({
                        'var1': correlacion.columns[i],
                        'var2': correlacion.columns[j],
                        'correlation': valor
                    })
        
        # Crear advertencias para correlaciones altas
        for corr in correlaciones_altas:
            error = BusinessError(
                rule_name="analisis_correlacion",
                error_type="high_correlation",
                message=f"Correlación muy alta entre {corr['var1']} y {corr['var2']}: {corr['correlation']:.3f}",
                severity="warning",
                context={'correlation_value': corr['correlation']}
            )
            error_handler.errors.append(error)
        
        return {
            'success': True,
            'correlation_matrix': correlacion,
            'high_correlations': correlaciones_altas
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def ejecutar_analisis_contingencia(df: pd.DataFrame, error_handler: BusinessErrorHandler) -> Dict[str, Any]:
    """Ejecuta análisis de contingencia con monitoreo de errores."""
    
    try:
        # Seleccionar columnas categóricas
        columnas_categoricas = df.select_dtypes(include=['object']).columns
        
        if len(columnas_categoricas) < 2:
            error = BusinessError(
                rule_name="analisis_contingencia",
                error_type="insufficient_data",
                message="Se requieren al menos 2 columnas categóricas para análisis de contingencia",
                severity="error"
            )
            error_handler.errors.append(error)
            return {
                'success': False,
                'error': 'Datos insuficientes para análisis de contingencia'
            }
        
        # Crear tabla de contingencia
        tabla_contingencia = pd.crosstab(df[columnas_categoricas[0]], df[columnas_categoricas[1]])
        
        # Verificar tamaño de la tabla
        if tabla_contingencia.size > 100:
            error = BusinessError(
                rule_name="analisis_contingencia",
                error_type="large_contingency_table",
                message=f"Tabla de contingencia muy grande ({tabla_contingencia.size} celdas)",
                severity="warning",
                context={'table_size': tabla_contingencia.size}
            )
            error_handler.errors.append(error)
        
        return {
            'success': True,
            'contingency_table': tabla_contingencia,
            'variables': [columnas_categoricas[0], columnas_categoricas[1]]
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def ejecutar_analisis_regresion(df: pd.DataFrame, error_handler: BusinessErrorHandler) -> Dict[str, Any]:
    """Ejecuta análisis de regresión con monitoreo de errores."""
    
    try:
        # Seleccionar variables
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        
        if len(columnas_numericas) < 2:
            error = BusinessError(
                rule_name="analisis_regresion",
                error_type="insufficient_data",
                message="Se requieren al menos 2 columnas numéricas para análisis de regresión",
                severity="error"
            )
            error_handler.errors.append(error)
            return {
                'success': False,
                'error': 'Datos insuficientes para análisis de regresión'
            }
        
        # Verificar valores faltantes
        missing_data = df[columnas_numericas].isnull().sum()
        if missing_data.sum() > 0:
            error = BusinessError(
                rule_name="analisis_regresion",
                error_type="missing_data",
                message=f"Valores faltantes detectados en variables numéricas: {missing_data.sum()} total",
                severity="warning",
                context={'missing_counts': missing_data.to_dict()}
            )
            error_handler.errors.append(error)
        
        return {
            'success': True,
            'variables_available': list(columnas_numericas),
            'missing_data': missing_data.to_dict()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def ejecutar_analisis_clusters(df: pd.DataFrame, error_handler: BusinessErrorHandler) -> Dict[str, Any]:
    """Ejecuta análisis de clusters con monitoreo de errores."""
    
    try:
        # Seleccionar variables numéricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        
        if len(columnas_numericas) < 2:
            error = BusinessError(
                rule_name="analisis_clusters",
                error_type="insufficient_data",
                message="Se requieren al menos 2 columnas numéricas para análisis de clusters",
                severity="error"
            )
            error_handler.errors.append(error)
            return {
                'success': False,
                'error': 'Datos insuficientes para análisis de clusters'
            }
        
        # Verificar escala de las variables
        for col in columnas_numericas:
            std_dev = df[col].std()
            if std_dev == 0:
                error = BusinessError(
                    rule_name="analisis_clusters",
                    error_type="zero_variance",
                    message=f"Variable {col} tiene varianza cero (no útil para clustering)",
                    severity="warning",
                    context={'variable': col, 'std_dev': std_dev}
                )
                error_handler.errors.append(error)
        
        return {
            'success': True,
            'variables_available': list(columnas_numericas),
            'n_variables': len(columnas_numericas)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def mostrar_resultados_analisis(resultado: Dict[str, Any]):
    """Muestra los resultados del análisis."""
    
    st.subheader("📊 Resultados del Análisis")
    
    if 'description' in resultado:
        st.write("**Estadísticas Descriptivas:**")
        st.dataframe(resultado['description'])
    
    if 'correlation_matrix' in resultado:
        st.write("**Matriz de Correlación:**")
        st.dataframe(resultado['correlation_matrix'])
    
    if 'contingency_table' in resultado:
        st.write("**Tabla de Contingencia:**")
        st.dataframe(resultado['contingency_table'])
    
    if 'problems' in resultado and resultado['problems']:
        st.write("**Problemas Detectados:**")
        for problema in resultado['problems']:
            st.warning(f"⚠️ {problema}")


def integrar_manejo_errores_en_exportacion():
    """
    Ejemplo de cómo integrar el manejo de errores en la sección de exportación.
    """
    
    st.header("📤 Exportación con Validación de Calidad")
    
    # Verificar que hay datos cargados
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ Primero debes cargar datos.")
        return
    
    df = st.session_state["df"]
    validation_results = st.session_state.get("validation_results", [])
    
    # Crear handler para errores de exportación
    error_handler = BusinessErrorHandler()
    
    # Opciones de exportación
    st.subheader("🎯 Opciones de Exportación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_csv = st.checkbox("📊 Exportar CSV", value=True)
        export_excel = st.checkbox("📈 Exportar Excel")
        export_report = st.checkbox("📄 Generar Reporte")
    
    with col2:
        include_validation = st.checkbox("✅ Incluir resultados de validación", value=True)
        include_errors = st.checkbox("⚠️ Incluir errores detectados", value=True)
        include_recommendations = st.checkbox("💡 Incluir recomendaciones", value=True)
    
    if st.button("🚀 Generar Exportación", type="primary"):
        
        with st.spinner("Generando exportación..."):
            try:
                # Verificar calidad de datos antes de exportar
                problemas_exportacion = []
                
                # Verificar valores faltantes
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    problemas_exportacion.append(f"Valores faltantes: {missing_data.sum()} total")
                
                # Verificar duplicados
                duplicados = df.duplicated().sum()
                if duplicados > 0:
                    problemas_exportacion.append(f"Filas duplicadas: {duplicados}")
                
                # Crear errores de exportación si hay problemas
                for problema in problemas_exportacion:
                    error = BusinessError(
                        rule_name="exportacion",
                        error_type="data_quality_issue",
                        message=problema,
                        severity="warning",
                        context={'export_type': 'general'}
                    )
                    error_handler.errors.append(error)
                
                # Generar archivos
                archivos_generados = []
                
                if export_csv:
                    csv_data = df.to_csv(index=False)
                    archivos_generados.append(("datos_exportados.csv", csv_data, "text/csv"))
                
                if export_excel:
                    # Aquí se generaría el Excel con openpyxl o xlsxwriter
                    archivos_generados.append(("datos_exportados.xlsx", "Excel data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
                
                if export_report:
                    # Generar reporte con errores y recomendaciones
                    reporte = generar_reporte_completo(df, validation_results, error_handler)
                    archivos_generados.append(("reporte_analisis.txt", reporte, "text/plain"))
                
                # Mostrar resultados
                st.success(f"✅ Exportación completada! {len(archivos_generados)} archivos generados.")
                
                # Mostrar archivos para descarga
                for nombre, datos, tipo in archivos_generados:
                    st.download_button(
                        label=f"📥 Descargar {nombre}",
                        data=datos,
                        file_name=nombre,
                        mime=tipo
                    )
                
                # Mostrar errores de exportación si los hay
                if error_handler.errors:
                    st.warning("⚠️ Advertencias durante la exportación:")
                    error_handler.display_errors_in_streamlit(show_details=False, show_charts=False)
                
            except Exception as e:
                st.error(f"❌ Error en la exportación: {str(e)}")


def generar_reporte_completo(df: pd.DataFrame, validation_results: List[ValidationResult], error_handler: BusinessErrorHandler) -> str:
    """Genera un reporte completo con errores y recomendaciones."""
    
    reporte = f"""REPORTE DE ANÁLISIS DE DATOS
Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESUMEN DE DATOS:
- Filas: {len(df)}
- Columnas: {len(df.columns)}
- Valores faltantes: {df.isnull().sum().sum()}
- Duplicados: {df.duplicated().sum()}

VALIDACIONES DE NEGOCIO:
"""
    
    for result in validation_results:
        status = "✅" if result.is_valid else "❌"
        reporte += f"{status} {result.rule_name}: {result.message}\n"
    
    if error_handler.errors:
        reporte += "\nERRORES DETECTADOS:\n"
        for error in error_handler.errors:
            reporte += f"- {error.rule_name}: {error.message}\n"
    
    # Agregar recomendaciones
    recommendations = error_handler.get_recommendations()
    if recommendations:
        reporte += "\nRECOMENDACIONES:\n"
        for rec in recommendations:
            reporte += f"- {rec}\n"
    
    return reporte


def main():
    """
    Función principal que demuestra la integración completa.
    """
    st.set_page_config(page_title="Integración Manejo de Errores", layout="wide")
    
    st.title("🔧 Integración del Sistema de Manejo de Errores")
    
    st.markdown("""
    Este ejemplo muestra cómo integrar el sistema avanzado de manejo de errores
    de negocio en una aplicación Streamlit existente.
    """)
    
    # Tabs para diferentes secciones
    tab1, tab2, tab3 = st.tabs(["📁 Carga de Datos", "📊 Análisis", "📤 Exportación"])
    
    with tab1:
        integrar_manejo_errores_en_carga_datos()
    
    with tab2:
        integrar_manejo_errores_en_analisis()
    
    with tab3:
        integrar_manejo_errores_en_exportacion()


if __name__ == "__main__":
    main() 