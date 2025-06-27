"""
Sistema de Logging Unificado
===========================

Módulo centralizado para logging de todas las acciones del pipeline.
Proporciona trazabilidad completa y auditoría de cada paso del proceso.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import yaml

@dataclass
class LogEntry:
    """Estructura de una entrada de log."""
    timestamp: str
    function: str
    step: str
    parameters: Dict[str, Any]
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    status: str  # 'success', 'error', 'warning'
    message: str
    execution_time: float
    error_details: Optional[str] = None

class UnifiedLogger:
    """
    Logger unificado para todo el sistema de análisis de datos.
    
    Responsabilidades:
    - Registrar cada acción del pipeline
    - Mantener historial de sesión
    - Proporcionar métricas antes/después
    - Exportar logs para auditoría
    """
    
    def __init__(self, config_path: str = "config/config.yml"):
        """
        Inicializa el logger unificado.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)
        self.session_logs: List[LogEntry] = []
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga la configuración del sistema."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"⚠️ Archivo de configuración no encontrado: {config_path}")
            return {}
    
    def _setup_logging(self):
        """Configura el sistema de logging."""
        log_config = self.config.get('logging', {})
        
        # Crear directorio de logs si no existe
        log_file = log_config.get('file', 'logs/pipeline.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('pipeline')
    
    def log_action(
        self,
        function: str,
        step: str,
        parameters: Dict[str, Any],
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any],
        status: str = 'success',
        message: str = '',
        execution_time: float = 0.0,
        error_details: Optional[str] = None
    ) -> LogEntry:
        """
        Registra una acción del pipeline.
        
        Args:
            function: Nombre de la función ejecutada
            step: Paso del pipeline
            parameters: Parámetros de entrada
            before_metrics: Métricas antes de la ejecución
            after_metrics: Métricas después de la ejecución
            status: Estado de la ejecución ('success', 'error', 'warning')
            message: Mensaje descriptivo
            execution_time: Tiempo de ejecución en segundos
            error_details: Detalles del error si ocurrió
            
        Returns:
            LogEntry: Entrada de log creada
        """
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            function=function,
            step=step,
            parameters=parameters,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            status=status,
            message=message,
            execution_time=execution_time,
            error_details=error_details
        )
        
        # Añadir a la lista de logs de sesión
        self.session_logs.append(entry)
        
        # Registrar en el sistema de logging
        log_message = f"[{step}] {function}: {message}"
        if status == 'error':
            self.logger.error(log_message)
            if error_details:
                self.logger.error(f"Error details: {error_details}")
        elif status == 'warning':
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        return entry
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """
        Obtiene el historial completo de la sesión.
        
        Returns:
            Lista de entradas de log convertidas a diccionarios
        """
        return [asdict(entry) for entry in self.session_logs]
    
    def get_step_history(self, step: str) -> List[LogEntry]:
        """
        Obtiene el historial de un paso específico.
        
        Args:
            step: Nombre del paso
            
        Returns:
            Lista de entradas de log para ese paso
        """
        return [entry for entry in self.session_logs if entry.step == step]
    
    def get_last_action(self) -> Optional[LogEntry]:
        """
        Obtiene la última acción registrada.
        
        Returns:
            Última entrada de log o None si no hay entradas
        """
        return self.session_logs[-1] if self.session_logs else None
    
    def clear_session(self):
        """Limpia el historial de la sesión actual."""
        self.session_logs.clear()
        self.logger.info("Session logs cleared")
    
    def export_logs(self, format: str = 'json', filepath: Optional[str] = None) -> str:
        """
        Exporta los logs de la sesión.
        
        Args:
            format: Formato de exportación ('json', 'csv', 'yaml')
            filepath: Ruta del archivo de salida (opcional)
            
        Returns:
            Ruta del archivo exportado
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/session_{timestamp}.{format}"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.get_session_history(), f, indent=2, ensure_ascii=False)
        
        elif format == 'yaml':
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.get_session_history(), f, default_flow_style=False, allow_unicode=True)
        
        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(self.get_session_history())
            df.to_csv(filepath, index=False, encoding='utf-8')
        
        self.logger.info(f"Logs exported to {filepath}")
        return filepath
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas resumidas de la sesión.
        
        Returns:
            Diccionario con estadísticas de la sesión
        """
        if not self.session_logs:
            return {}
        
        total_actions = len(self.session_logs)
        successful_actions = len([e for e in self.session_logs if e.status == 'success'])
        error_actions = len([e for e in self.session_logs if e.status == 'error'])
        warning_actions = len([e for e in self.session_logs if e.status == 'warning'])
        
        total_execution_time = sum(e.execution_time for e in self.session_logs)
        avg_execution_time = total_execution_time / total_actions if total_actions > 0 else 0
        
        steps_executed = list(set(e.step for e in self.session_logs))
        
        return {
            'total_actions': total_actions,
            'successful_actions': successful_actions,
            'error_actions': error_actions,
            'warning_actions': warning_actions,
            'success_rate': successful_actions / total_actions if total_actions > 0 else 0,
            'total_execution_time': total_execution_time,
            'avg_execution_time': avg_execution_time,
            'steps_executed': steps_executed,
            'session_start': self.session_logs[0].timestamp if self.session_logs else None,
            'session_end': self.session_logs[-1].timestamp if self.session_logs else None
        }
    
    def validate_integrity(self) -> Dict[str, Any]:
        """
        Valida la integridad de los logs de la sesión.
        
        Returns:
            Diccionario con resultados de la validación
        """
        if not self.session_logs:
            return {'valid': True, 'message': 'No logs to validate'}
        
        issues = []
        
        # Verificar que no hay gaps en el tiempo
        for i in range(1, len(self.session_logs)):
            prev_time = datetime.fromisoformat(self.session_logs[i-1].timestamp)
            curr_time = datetime.fromisoformat(self.session_logs[i].timestamp)
            
            if (curr_time - prev_time).total_seconds() > 300:  # 5 minutos
                issues.append(f"Gap detected between actions {i-1} and {i}")
        
        # Verificar que los pasos están en orden lógico
        step_order = [
            'data_load', 'schema_validation', 'semantic_classification',
            'type_detection', 'autocorrection', 'missing_handling',
            'filtering', 'integrity_validation', 'statistical_analysis',
            'visualization_suggestion', 'report_generation', 'export'
        ]
        
        executed_steps = [e.step for e in self.session_logs]
        for step in step_order:
            if step in executed_steps:
                # Verificar que los pasos anteriores también se ejecutaron
                step_index = executed_steps.index(step)
                expected_previous_steps = [s for s in step_order[:step_order.index(step)] if s in step_order[:step_order.index(step)]]
                
                for prev_step in expected_previous_steps:
                    if prev_step not in executed_steps[:step_index]:
                        issues.append(f"Step {step} executed before {prev_step}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }

# Instancia global del logger
_logger_instance: Optional[UnifiedLogger] = None

def get_logger() -> UnifiedLogger:
    """
    Obtiene la instancia global del logger.
    
    Returns:
        Instancia del UnifiedLogger
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = UnifiedLogger()
    return _logger_instance

def log_action(
    function: str,
    step: str,
    parameters: Dict[str, Any],
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    status: str = 'success',
    message: str = '',
    execution_time: float = 0.0,
    error_details: Optional[str] = None
) -> LogEntry:
    """
    Función de conveniencia para registrar una acción.
    
    Args:
        function: Nombre de la función ejecutada
        step: Paso del pipeline
        parameters: Parámetros de entrada
        before_metrics: Métricas antes de la ejecución
        after_metrics: Métricas después de la ejecución
        status: Estado de la ejecución
        message: Mensaje descriptivo
        execution_time: Tiempo de ejecución en segundos
        error_details: Detalles del error si ocurrió
        
    Returns:
        LogEntry: Entrada de log creada
    """
    logger = get_logger()
    return logger.log_action(
        function=function,
        step=step,
        parameters=parameters,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        status=status,
        message=message,
        execution_time=execution_time,
        error_details=error_details
    )

def setup_logging(config_path: str = "config/config.yml") -> UnifiedLogger:
    """
    Configura e inicializa el sistema de logging.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Instancia del UnifiedLogger configurado
    """
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = UnifiedLogger(config_path)
    
    return _logger_instance 