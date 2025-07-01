"""
Sistema Avanzado de Manejo de Errores de Negocio
===============================================

Este módulo proporciona:
1. Extracción detallada de errores desde logs/metadata
2. Visualización avanzada en Streamlit
3. Exportación y reporte de errores
4. Integración con el sistema de logging existente
5. Funciones de utilidad para análisis de errores
"""

import pandas as pd
import streamlit as st
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Importar módulos del sistema
from .logging import UnifiedLogger, get_logger
from .error_reporter import ErrorReporter, ValidationError
from .business_rules import ValidationResult, BusinessRuleError


@dataclass
class BusinessError:
    """Estructura detallada para errores de negocio"""
    rule_name: str
    error_type: str
    message: str
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    invalid_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    severity: str = "error"  # error, warning, info
    timestamp: str = ""
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class BusinessErrorHandler:
    """
    Manejador avanzado de errores de negocio con capacidades de:
    - Extracción de errores desde logs
    - Análisis estadístico de errores
    - Visualización interactiva
    - Exportación de reportes
    """
    
    def __init__(self, logger: Optional[UnifiedLogger] = None):
        self.logger = logger or get_logger()
        self.error_reporter = ErrorReporter()
        self.errors: List[BusinessError] = []
        self.error_summary: Dict[str, Any] = {}
    
    def extract_business_errors_from_logs(self, logs: List[Dict[str, Any]]) -> List[BusinessError]:
        """
        Extrae errores de negocio desde los logs del sistema.
        
        Args:
            logs: Lista de entradas de log
            
        Returns:
            Lista de BusinessError estructurados
        """
        business_errors = []
        
        for entry in logs:
            if entry.get('step') == 'business_rules' and entry.get('status') in ['error', 'warning']:
                # Extraer errores de la metadata
                after_metrics = entry.get('after_metrics', {})
                error_details = self._parse_error_details(after_metrics, entry)
                
                for error_detail in error_details:
                    business_error = BusinessError(
                        rule_name=entry.get('function', 'unknown'),
                        error_type=entry.get('status', 'error'),
                        message=error_detail.get('message', entry.get('message', '')),
                        row_index=error_detail.get('row_index'),
                        column_name=error_detail.get('column_name'),
                        invalid_value=error_detail.get('invalid_value'),
                        expected_value=error_detail.get('expected_value'),
                        severity=entry.get('status', 'error'),
                        timestamp=entry.get('timestamp', ''),
                        context={
                            'function': entry.get('function'),
                            'execution_time': entry.get('execution_time'),
                            'parameters': entry.get('parameters', {})
                        }
                    )
                    business_errors.append(business_error)
        
        self.errors = business_errors
        return business_errors
    
    def extract_errors_from_validation_results(self, results: List[ValidationResult]) -> List[BusinessError]:
        """
        Extrae errores desde ValidationResult del sistema de business_rules.
        
        Args:
            results: Lista de ValidationResult
            
        Returns:
            Lista de BusinessError estructurados
        """
        business_errors = []
        
        for result in results:
            if not result.is_valid or result.details.get('filas_con_errores', 0) > 0:
                # Crear error principal
                main_error = BusinessError(
                    rule_name=result.rule_name,
                    error_type="validation_error",
                    message=result.message,
                    severity="error" if not result.is_valid else "warning",
                    timestamp=result.timestamp.isoformat() if hasattr(result.timestamp, 'isoformat') else str(result.timestamp),
                    context={
                        'details': result.details,
                        'is_valid': result.is_valid
                    }
                )
                business_errors.append(main_error)
                
                # Extraer errores específicos de filas si están disponibles
                if 'filas_con_errores' in result.details and result.details['filas_con_errores'] > 0:
                    # Aquí podrías agregar lógica para extraer errores específicos por fila
                    # Por ahora, creamos un error genérico
                    row_error = BusinessError(
                        rule_name=f"{result.rule_name}_rows",
                        error_type="row_validation_error",
                        message=f"{result.details['filas_con_errores']} filas con errores en {result.rule_name}",
                        severity="error",
                        timestamp=result.timestamp.isoformat() if hasattr(result.timestamp, 'isoformat') else str(result.timestamp),
                        context={'affected_rows': result.details['filas_con_errores']}
                    )
                    business_errors.append(row_error)
        
        self.errors = business_errors
        return business_errors
    
    def _parse_error_details(self, after_metrics: Dict[str, Any], log_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parsea los detalles de error desde la metadata del log.
        
        Args:
            after_metrics: Métricas después de la ejecución
            log_entry: Entrada completa del log
            
        Returns:
            Lista de detalles de error estructurados
        """
        error_details = []
        
        # Extraer errores de diferentes tipos de validación
        if 'filas_con_errores' in after_metrics:
            error_details.append({
                'message': f"{after_metrics['filas_con_errores']} filas con errores de validación",
                'row_index': None,
                'column_name': None,
                'invalid_value': None,
                'expected_value': None
            })
        
        if 'regiones_invalidas' in after_metrics and after_metrics['regiones_invalidas']:
            for region in after_metrics['regiones_invalidas']:
                error_details.append({
                    'message': f"Región inválida: {region}",
                    'column_name': 'region',
                    'invalid_value': region,
                    'expected_value': 'Región válida de Chile'
                })
        
        if 'comunas_invalidas' in after_metrics and after_metrics['comunas_invalidas']:
            for comuna in after_metrics['comunas_invalidas']:
                error_details.append({
                    'message': f"Comuna inválida: {comuna}",
                    'column_name': 'comuna',
                    'invalid_value': comuna,
                    'expected_value': 'Comuna válida de Chile'
                })
        
        if 'valores_invalidos' in after_metrics and after_metrics['valores_invalidos']:
            for valor in after_metrics['valores_invalidos']:
                error_details.append({
                    'message': f"Valor inválido detectado: {valor}",
                    'invalid_value': valor,
                    'expected_value': 'Valor dentro del rango esperado'
                })
        
        return error_details
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Genera un resumen estadístico de los errores.
        
        Returns:
            Diccionario con estadísticas de errores
        """
        if not self.errors:
            return {
                'total_errors': 0,
                'errors_by_rule': {},
                'errors_by_severity': {},
                'errors_by_column': {},
                'timeline': []
            }
        
        # Estadísticas básicas
        total_errors = len(self.errors)
        
        # Errores por regla
        errors_by_rule = {}
        for error in self.errors:
            errors_by_rule[error.rule_name] = errors_by_rule.get(error.rule_name, 0) + 1
        
        # Errores por severidad
        errors_by_severity = {}
        for error in self.errors:
            errors_by_severity[error.severity] = errors_by_severity.get(error.severity, 0) + 1
        
        # Errores por columna
        errors_by_column = {}
        for error in self.errors:
            if error.column_name:
                errors_by_column[error.column_name] = errors_by_column.get(error.column_name, 0) + 1
        
        # Timeline de errores
        timeline = []
        for error in self.errors:
            try:
                timestamp = datetime.fromisoformat(error.timestamp.replace('Z', '+00:00'))
                timeline.append({
                    'timestamp': timestamp,
                    'rule_name': error.rule_name,
                    'severity': error.severity,
                    'message': error.message
                })
            except:
                continue
        
        timeline.sort(key=lambda x: x['timestamp'])
        
        self.error_summary = {
            'total_errors': total_errors,
            'errors_by_rule': errors_by_rule,
            'errors_by_severity': errors_by_severity,
            'errors_by_column': errors_by_column,
            'timeline': timeline
        }
        
        return self.error_summary
    
    def display_errors_in_streamlit(self, show_details: bool = True, show_charts: bool = True) -> None:
        """
        Muestra los errores de negocio en Streamlit con visualizaciones avanzadas.
        
        Args:
            show_details: Si mostrar detalles expandibles
            show_charts: Si mostrar gráficos de análisis
        """
        if not self.errors:
            st.success("✅ No se detectaron errores de negocio en los datos.")
            return
        
        # Resumen general
        summary = self.get_error_summary()
        
        # Header con resumen
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Errores", summary['total_errors'])
        with col2:
            st.metric("Reglas Afectadas", len(summary['errors_by_rule']))
        with col3:
            st.metric("Columnas Afectadas", len(summary['errors_by_column']))
        with col4:
            critical_errors = summary['errors_by_severity'].get('error', 0)
            st.metric("Errores Críticos", critical_errors, delta_color="inverse")
        
        # Gráficos de análisis
        if show_charts:
            self._display_error_charts(summary)
        
        # Tabla de errores detallada
        if show_details:
            self._display_error_table()
        
        # Botones de acción
        self._display_action_buttons()
    
    def _display_error_charts(self, summary: Dict[str, Any]) -> None:
        """Muestra gráficos de análisis de errores"""
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Errores por Regla', 'Errores por Severidad', 
                          'Errores por Columna', 'Timeline de Errores'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Gráfico 1: Errores por regla
        if summary['errors_by_rule']:
            fig.add_trace(
                go.Bar(
                    x=list(summary['errors_by_rule'].keys()),
                    y=list(summary['errors_by_rule'].values()),
                    name="Errores por Regla",
                    marker_color='lightcoral'
                ),
                row=1, col=1
            )
        
        # Gráfico 2: Errores por severidad
        if summary['errors_by_severity']:
            fig.add_trace(
                go.Pie(
                    labels=list(summary['errors_by_severity'].keys()),
                    values=list(summary['errors_by_severity'].values()),
                    name="Errores por Severidad"
                ),
                row=1, col=2
            )
        
        # Gráfico 3: Errores por columna
        if summary['errors_by_column']:
            fig.add_trace(
                go.Bar(
                    x=list(summary['errors_by_column'].keys()),
                    y=list(summary['errors_by_column'].values()),
                    name="Errores por Columna",
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        # Gráfico 4: Timeline de errores
        if summary['timeline']:
            timestamps = [entry['timestamp'] for entry in summary['timeline']]
            rule_names = [entry['rule_name'] for entry in summary['timeline']]
            severities = [entry['severity'] for entry in summary['timeline']]
            
            colors = ['red' if s == 'error' else 'orange' if s == 'warning' else 'blue' 
                     for s in severities]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=rule_names,
                    mode='markers',
                    marker=dict(size=10, color=colors),
                    name="Timeline de Errores"
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="Análisis de Errores de Negocio")
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_error_table(self) -> None:
        """Muestra tabla detallada de errores"""
        st.subheader("📋 Detalle de Errores")
        
        # Convertir errores a DataFrame
        error_data = []
        for i, error in enumerate(self.errors):
            error_data.append({
                'ID': i + 1,
                'Regla': error.rule_name,
                'Tipo': error.error_type,
                'Severidad': error.severity,
                'Mensaje': error.message,
                'Columna': error.column_name or 'N/A',
                'Valor Inválido': str(error.invalid_value) if error.invalid_value is not None else 'N/A',
                'Fila': error.row_index if error.row_index is not None else 'N/A',
                'Timestamp': error.timestamp[:19] if error.timestamp else 'N/A'
            })
        
        df_errors = pd.DataFrame(error_data)
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            severity_filter = st.selectbox(
                "Filtrar por severidad",
                ['Todos'] + list(set(df_errors['Severidad'].unique()))
            )
        with col2:
            rule_filter = st.selectbox(
                "Filtrar por regla",
                ['Todas'] + list(set(df_errors['Regla'].unique()))
            )
        with col3:
            column_filter = st.selectbox(
                "Filtrar por columna",
                ['Todas'] + [col for col in df_errors['Columna'].unique() if col != 'N/A']
            )
        
        # Aplicar filtros
        filtered_df = df_errors.copy()
        if severity_filter != 'Todos':
            filtered_df = filtered_df[filtered_df['Severidad'] == severity_filter]
        if rule_filter != 'Todas':
            filtered_df = filtered_df[filtered_df['Regla'] == rule_filter]
        if column_filter != 'Todas':
            filtered_df = filtered_df[filtered_df['Columna'] == column_filter]
        
        # Mostrar tabla
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Estadísticas de filtros
        st.info(f"Mostrando {len(filtered_df)} de {len(df_errors)} errores")
    
    def _display_action_buttons(self) -> None:
        """Muestra botones de acción para manejo de errores"""
        st.subheader("🔧 Acciones")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Copiar Reporte", key="copy_report"):
                self._copy_error_report_to_clipboard()
                st.success("✅ Reporte copiado al portapapeles")
        
        with col2:
            if st.button("📊 Exportar CSV", key="export_csv"):
                self._export_errors_to_csv()
                st.success("✅ Errores exportados a CSV")
        
        with col3:
            if st.button("📄 Generar Reporte", key="generate_report"):
                self._generate_detailed_report()
                st.success("✅ Reporte detallado generado")
    
    def _copy_error_report_to_clipboard(self) -> None:
        """Copia el reporte de errores al portapapeles"""
        summary = self.get_error_summary()
        
        report_text = f"""REPORTE DE ERRORES DE NEGOCIO
Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESUMEN:
- Total de errores: {summary['total_errors']}
- Reglas afectadas: {len(summary['errors_by_rule'])}
- Columnas afectadas: {len(summary['errors_by_column'])}

ERRORES POR REGLA:
{chr(10).join([f"- {rule}: {count}" for rule, count in summary['errors_by_rule'].items()])}

ERRORES POR SEVERIDAD:
{chr(10).join([f"- {severity}: {count}" for severity, count in summary['errors_by_severity'].items()])}

DETALLE DE ERRORES:
"""
        
        for i, error in enumerate(self.errors, 1):
            report_text += f"""
{i}. {error.rule_name} ({error.severity})
   Mensaje: {error.message}
   Columna: {error.column_name or 'N/A'}
   Valor inválido: {error.invalid_value or 'N/A'}
   Fila: {error.row_index or 'N/A'}
   Timestamp: {error.timestamp}
"""
        
        # Usar JavaScript para copiar al portapapeles
        js_code = f"""
        navigator.clipboard.writeText(`{report_text}`).then(function() {{
            console.log('Reporte copiado al portapapeles');
        }});
        """
        st.components.v1.html(
            f"<script>{js_code}</script>",
            height=0
        )
    
    def _export_errors_to_csv(self) -> None:
        """Exporta los errores a un archivo CSV"""
        if not self.errors:
            return
        
        # Crear DataFrame para exportación
        export_data = []
        for error in self.errors:
            export_data.append({
                'rule_name': error.rule_name,
                'error_type': error.error_type,
                'severity': error.severity,
                'message': error.message,
                'column_name': error.column_name,
                'invalid_value': error.invalid_value,
                'expected_value': error.expected_value,
                'row_index': error.row_index,
                'timestamp': error.timestamp
            })
        
        df_export = pd.DataFrame(export_data)
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"business_errors_{timestamp}.csv"
        
        # Convertir a CSV
        csv_data = df_export.to_csv(index=False, encoding='utf-8')
        
        # Descargar archivo
        st.download_button(
            label="📥 Descargar CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
    
    def _generate_detailed_report(self) -> None:
        """Genera un reporte detallado en formato expandible"""
        summary = self.get_error_summary()
        
        with st.expander("📄 Reporte Detallado de Errores", expanded=True):
            st.markdown(f"""
            # Reporte de Errores de Negocio
            
            **Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            ## Resumen Ejecutivo
            
            - **Total de errores detectados:** {summary['total_errors']}
            - **Reglas de validación afectadas:** {len(summary['errors_by_rule'])}
            - **Columnas de datos afectadas:** {len(summary['errors_by_column'])}
            - **Errores críticos:** {summary['errors_by_severity'].get('error', 0)}
            - **Advertencias:** {summary['errors_by_severity'].get('warning', 0)}
            
            ## Análisis por Categoría
            
            ### Errores por Regla de Validación
            """)
            
            for rule, count in summary['errors_by_rule'].items():
                st.markdown(f"- **{rule}:** {count} errores")
            
            st.markdown("### Errores por Severidad")
            for severity, count in summary['errors_by_severity'].items():
                color = "🔴" if severity == "error" else "🟡" if severity == "warning" else "🔵"
                st.markdown(f"- {color} **{severity}:** {count}")
            
            st.markdown("### Errores por Columna")
            for column, count in summary['errors_by_column'].items():
                st.markdown(f"- **{column}:** {count} errores")
            
            st.markdown("## Recomendaciones")
            
            if summary['errors_by_severity'].get('error', 0) > 0:
                st.warning("⚠️ **Se detectaron errores críticos que requieren atención inmediata.**")
            
            if len(summary['errors_by_column']) > 0:
                st.info("💡 **Recomendaciones específicas:**")
                for column in summary['errors_by_column'].keys():
                    st.markdown(f"- Revisar y corregir datos en la columna **{column}**")
            
            st.markdown("## Detalle Completo de Errores")
            
            for i, error in enumerate(self.errors, 1):
                st.markdown(f"""
                ### Error {i}: {error.rule_name}
                
                - **Tipo:** {error.error_type}
                - **Severidad:** {error.severity}
                - **Mensaje:** {error.message}
                - **Columna:** {error.column_name or 'N/A'}
                - **Valor inválido:** {error.invalid_value or 'N/A'}
                - **Fila:** {error.row_index or 'N/A'}
                - **Timestamp:** {error.timestamp}
                """)
    
    def get_recommendations(self) -> List[str]:
        """
        Genera recomendaciones automáticas basadas en los errores detectados.
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        summary = self.get_error_summary()
        
        # Recomendaciones basadas en severidad
        if summary['errors_by_severity'].get('error', 0) > 0:
            recommendations.append("🔴 Revisar y corregir errores críticos antes de continuar con el análisis")
        
        if summary['errors_by_severity'].get('warning', 0) > 5:
            recommendations.append("🟡 Considerar revisar las advertencias para mejorar la calidad de los datos")
        
        # Recomendaciones basadas en columnas
        for column, count in summary['errors_by_column'].items():
            if count > 10:
                recommendations.append(f"📊 La columna '{column}' tiene muchos errores. Considerar limpieza de datos")
        
        # Recomendaciones específicas por tipo de regla
        for rule, count in summary['errors_by_rule'].items():
            if rule == 'demographics' and count > 0:
                recommendations.append("👥 Revisar datos demográficos: verificar rangos de edad, valores de género válidos")
            elif rule == 'geography' and count > 0:
                recommendations.append("🗺️ Verificar datos geográficos: regiones y comunas deben ser válidas para Chile")
            elif rule == 'likert' and count > 0:
                recommendations.append("📊 Revisar escalas Likert: verificar que los valores estén en el rango esperado")
        
        return recommendations


# Funciones de utilidad para uso directo
def get_business_errors(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Función de utilidad para extraer errores de negocio desde logs.
    
    Args:
        logs: Lista de entradas de log
        
    Returns:
        Lista de errores estructurados
    """
    handler = BusinessErrorHandler()
    business_errors = handler.extract_business_errors_from_logs(logs)
    
    # Convertir a formato de diccionario para compatibilidad
    error_dicts = []
    for error in business_errors:
        error_dicts.append({
            'regla': error.rule_name,
            'fila': error.row_index,
            'detalle': error.message,
            'timestamp': error.timestamp,
            'severidad': error.severity,
            'columna': error.column_name,
            'valor_invalido': error.invalid_value
        })
    
    return error_dicts


def display_business_errors_in_streamlit(
    logs: List[Dict[str, Any]], 
    validation_results: Optional[List[ValidationResult]] = None
) -> None:
    """
    Función de utilidad para mostrar errores de negocio en Streamlit.
    
    Args:
        logs: Lista de entradas de log
        validation_results: Resultados de validación opcionales
    """
    handler = BusinessErrorHandler()
    
    # Extraer errores desde logs
    if logs:
        handler.extract_business_errors_from_logs(logs)
    
    # Extraer errores desde resultados de validación
    if validation_results:
        handler.extract_errors_from_validation_results(validation_results)
    
    # Mostrar errores
    handler.display_errors_in_streamlit()
    
    # Mostrar recomendaciones
    recommendations = handler.get_recommendations()
    if recommendations:
        st.subheader("💡 Recomendaciones")
        for rec in recommendations:
            st.info(rec)


# Instancia global para uso directo
business_error_handler = BusinessErrorHandler() 