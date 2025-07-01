"""
Sistema de reporte y manejo de errores híbrido.

Este módulo implementa un sistema de manejo de errores que:
1. Detiene la aplicación con mensajes claros para errores de validación
2. Proporciona detalles técnicos en expanders
3. Permite copiar detalles al portapapeles
4. Envía notificaciones externas solo cuando está habilitado
"""

import streamlit as st
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@dataclass
class ValidationError:
    """Estructura para errores de validación"""
    error_type: str
    message: str
    context: str
    timestamp: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None


class ErrorReporter:
    """Sistema de reporte de errores híbrido"""
    
    def __init__(self, notifications_enabled: bool = False):
        self.notifications_enabled = notifications_enabled
        self.errors_log: List[ValidationError] = []
    
    def report_validation_error(
        self, 
        error_type: str, 
        message: str, 
        context: str, 
        details: Optional[Dict[str, Any]] = None,
        stop_app: bool = True
    ) -> None:
        """
        Reporta un error de validación y opcionalmente detiene la aplicación.
        
        Args:
            error_type: Tipo de error (ej: "DataValidationError", "ConfigError")
            message: Mensaje breve del error
            context: Contexto donde ocurrió el error
            details: Detalles adicionales del error
            stop_app: Si True, detiene la aplicación después del reporte
        """
        error = ValidationError(
            error_type=error_type,
            message=message,
            context=context,
            timestamp=datetime.now().isoformat(),
            details=details or {},
            stack_trace=traceback.format_exc() if details else None
        )
        
        self.errors_log.append(error)
        
        # Mostrar error en Streamlit
        self._display_error_in_ui(error)
        
        # Enviar notificación externa si está habilitado
        if self.notifications_enabled:
            self._send_external_notification(error)
        
        # Detener aplicación si se requiere
        if stop_app:
            st.stop()
    
    def _display_error_in_ui(self, error: ValidationError) -> None:
        """Muestra el error en la interfaz de Streamlit"""
        
        # Mensaje principal de error
        st.error(f"⚠️ Ha ocurrido un error de validación: {error.message}")
        
        # Expander con detalles técnicos
        with st.expander("🔍 Ver detalles del error"):
            st.code(
                f"""ErrorType: {error.error_type}
Mensaje: {error.message}
Contexto: {error.context}
Timestamp: {error.timestamp}
Detalles: {json.dumps(error.details, indent=2, ensure_ascii=False)}""",
                language="text"
            )
            
            # Botón para copiar detalles
            if st.button("📋 Copiar detalles", key=f"copy_{error.timestamp}"):
                self._copy_to_clipboard(error)
                st.success("✅ Detalles copiados al portapapeles")
        
        # Mensaje de soporte
        st.info(
            "💡 **No te preocupes**: copia los detalles y compártelos con el equipo "
            "para que podamos solucionarlo lo antes posible."
        )
    
    def _copy_to_clipboard(self, error: ValidationError) -> None:
        """Copia los detalles del error al portapapeles usando JavaScript"""
        error_text = f"""ErrorType: {error.error_type}
Mensaje: {error.message}
Contexto: {error.context}
Timestamp: {error.timestamp}
Detalles: {json.dumps(error.details, indent=2, ensure_ascii=False)}"""
        
        # Usar JavaScript para copiar al portapapeles
        js_code = f"""
        navigator.clipboard.writeText(`{error_text}`).then(function() {{
            console.log('Detalles copiados al portapapeles');
        }});
        """
        st.components.v1.html(
            f"<script>{js_code}</script>",
            height=0
        )
    
    def _send_external_notification(self, error: ValidationError) -> None:
        """Envía notificación externa (Slack/Email) si está configurado"""
        try:
            # Aquí se implementaría la lógica de envío a Slack/Email
            # Por ahora solo registramos en logs
            print(f"🔔 Notificación externa enviada para error: {error.error_type}")
        except Exception as e:
            print(f"❌ Error al enviar notificación externa: {e}")
    
    def get_errors_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de errores para reportes"""
        return {
            "total_errors": len(self.errors_log),
            "errors_by_type": self._group_errors_by_type(),
            "recent_errors": [
                asdict(error) for error in self.errors_log[-5:]  # Últimos 5 errores
            ]
        }
    
    def _group_errors_by_type(self) -> Dict[str, int]:
        """Agrupa errores por tipo"""
        error_counts = {}
        for error in self.errors_log:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
        return error_counts


# Instancia global del reporter
error_reporter = ErrorReporter()


def report_config_error(message: str, context: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Helper para reportar errores de configuración"""
    error_reporter.report_validation_error(
        error_type="ConfigValidationError",
        message=message,
        context=context,
        details=details
    )


def report_dataframe_error(message: str, context: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Helper para reportar errores de DataFrame"""
    error_reporter.report_validation_error(
        error_type="DataFrameValidationError",
        message=message,
        context=context,
        details=details
    )


def report_parameter_error(message: str, context: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Helper para reportar errores de parámetros"""
    error_reporter.report_validation_error(
        error_type="ParameterValidationError",
        message=message,
        context=context,
        details=details
    )


def report_business_rule_error(message: str, context: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Helper para reportar errores de reglas de negocio"""
    error_reporter.report_validation_error(
        error_type="BusinessRuleValidationError",
        message=message,
        context=context,
        details=details
    )


def set_notifications_enabled(enabled: bool) -> None:
    """Habilita o deshabilita las notificaciones externas"""
    error_reporter.notifications_enabled = enabled


def get_error_summary() -> Dict[str, Any]:
    """Obtiene resumen de errores para debugging"""
    return error_reporter.get_errors_summary() 