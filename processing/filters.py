"""
Módulo de Filtros Dinámicos
==========================

Responsabilidades:
- Aplicación de filtros dinámicos con validación
- Reversión de filtros (undo/redo)
- Validación de integridad después de filtrar
- Logging de todas las operaciones de filtrado
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar logging
from .logging import log_action

class FilterManager:
    """
    Gestor de filtros que mantiene historial y permite reversión.
    """
    
    def __init__(self):
        """Inicializa el gestor de filtros."""
        self.filter_history: List[Dict[str, Any]] = []
        self.current_filter_index = -1
        self.original_df: Optional[pd.DataFrame] = None
    
    def set_original_data(self, df: pd.DataFrame):
        """
        Establece el DataFrame original como punto de referencia.
        
        Args:
            df: DataFrame original sin filtros
        """
        self.original_df = df.copy()
        self.filter_history = []
        self.current_filter_index = -1
    
    def apply_filter(
        self, 
        df: pd.DataFrame, 
        filter_conditions: Dict[str, Any],
        filter_name: str = "Filtro personalizado"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Aplica un filtro al DataFrame con validación y logging.
        
        Args:
            df: DataFrame a filtrar
            filter_conditions: Condiciones del filtro
            filter_name: Nombre descriptivo del filtro
            
        Returns:
            Tuple: (DataFrame filtrado, metadata del filtro)
        """
        start_time = datetime.now()
        
        try:
            # Validar condiciones del filtro
            validated_conditions = self._validate_filter_conditions(df, filter_conditions)
            
            # Métricas antes del filtrado
            before_metrics = {
                'rows': len(df),
                'columns': len(df.columns),
                'filter_conditions': filter_conditions
            }
            
            # Aplicar filtro
            filtered_df = self._apply_filter_conditions(df, validated_conditions)
            
            # Calcular métricas del filtro
            filter_metrics = self._calculate_filter_metrics(df, filtered_df, filter_conditions)
            
            # Crear entrada en el historial
            filter_entry = {
                'timestamp': datetime.now().isoformat(),
                'filter_name': filter_name,
                'conditions': validated_conditions,
                'before_rows': len(df),
                'after_rows': len(filtered_df),
                'rows_removed': len(df) - len(filtered_df),
                'removal_percentage': ((len(df) - len(filtered_df)) / len(df)) * 100 if len(df) > 0 else 0,
                'metrics': filter_metrics
            }
            
            # Actualizar historial
            self.filter_history.append(filter_entry)
            self.current_filter_index = len(self.filter_history) - 1
            
            # Métricas después del filtrado
            after_metrics = {
                'rows': len(filtered_df),
                'columns': len(filtered_df.columns),
                'filter_applied': True,
                'filter_name': filter_name
            }
            
            # Registrar acción
            log_action(
                function='apply_filter',
                step='filtering',
                parameters={'filter_name': filter_name, 'conditions': filter_conditions},
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                status='success',
                message=f"Filtro '{filter_name}' aplicado exitosamente",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            return filtered_df, filter_metrics
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Registrar error
            log_action(
                function='apply_filter',
                step='filtering',
                parameters={'filter_name': filter_name, 'conditions': filter_conditions},
                before_metrics=before_metrics if 'before_metrics' in locals() else {},
                after_metrics={},
                status='error',
                message=f"Error al aplicar filtro '{filter_name}': {str(e)}",
                execution_time=execution_time,
                error_details=str(e)
            )
            
            raise ValueError(f"Error al aplicar filtro: {str(e)}")
    
    def _validate_filter_conditions(
        self, 
        df: pd.DataFrame, 
        conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Valida las condiciones del filtro.
        
        Args:
            df: DataFrame a filtrar
            conditions: Condiciones del filtro
            
        Returns:
            Condiciones validadas
            
        Raises:
            ValueError: Si las condiciones no son válidas
        """
        validated_conditions = {}
        
        for column, condition in conditions.items():
            # Verificar que la columna existe
            if column not in df.columns:
                raise ValueError(f"Columna '{column}' no existe en el DataFrame")
            
            # Validar según el tipo de condición
            if isinstance(condition, dict):
                # Condición compleja: {operator: value}
                if 'operator' not in condition or 'value' not in condition:
                    raise ValueError(f"Condición inválida para columna '{column}': debe tener 'operator' y 'value'")
                
                operator = condition['operator']
                value = condition['value']
                
                # Validar operador
                valid_operators = ['==', '!=', '>', '<', '>=', '<=', 'in', 'not_in', 'contains', 'not_contains']
                if operator not in valid_operators:
                    raise ValueError(f"Operador '{operator}' no válido para columna '{column}'")
                
                # Validar valor según tipo de datos
                if operator in ['>', '<', '>=', '<=']:
                    if not pd.api.types.is_numeric_dtype(df[column]):
                        raise ValueError(f"Operador '{operator}' solo válido para columnas numéricas")
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Valor para operador '{operator}' debe ser numérico")
                
                validated_conditions[column] = condition
                
            elif isinstance(condition, (list, tuple)):
                # Condición de lista: valores permitidos
                validated_conditions[column] = {'operator': 'in', 'value': condition}
                
            else:
                # Condición simple: igualdad
                validated_conditions[column] = {'operator': '==', 'value': condition}
        
        return validated_conditions
    
    def _apply_filter_conditions(
        self, 
        df: pd.DataFrame, 
        conditions: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Aplica las condiciones de filtro al DataFrame.
        
        Args:
            df: DataFrame a filtrar
            conditions: Condiciones validadas
            
        Returns:
            DataFrame filtrado
        """
        filtered_df = df.copy()
        
        for column, condition in conditions.items():
            operator = condition['operator']
            value = condition['value']
            
            if operator == '==':
                mask = filtered_df[column] == value
            elif operator == '!=':
                mask = filtered_df[column] != value
            elif operator == '>':
                mask = filtered_df[column] > value
            elif operator == '<':
                mask = filtered_df[column] < value
            elif operator == '>=':
                mask = filtered_df[column] >= value
            elif operator == '<=':
                mask = filtered_df[column] <= value
            elif operator == 'in':
                mask = filtered_df[column].isin(value)
            elif operator == 'not_in':
                mask = ~filtered_df[column].isin(value)
            elif operator == 'contains':
                mask = filtered_df[column].astype(str).str.contains(str(value), na=False)
            elif operator == 'not_contains':
                mask = ~filtered_df[column].astype(str).str.contains(str(value), na=False)
            else:
                raise ValueError(f"Operador '{operator}' no implementado")
            
            filtered_df = filtered_df[mask]
        
        return filtered_df
    
    def _calculate_filter_metrics(
        self, 
        original_df: pd.DataFrame, 
        filtered_df: pd.DataFrame, 
        conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcula métricas del filtro aplicado.
        
        Args:
            original_df: DataFrame original
            filtered_df: DataFrame filtrado
            conditions: Condiciones aplicadas
            
        Returns:
            Diccionario con métricas del filtro
        """
        rows_removed = len(original_df) - len(filtered_df)
        removal_percentage = (rows_removed / len(original_df)) * 100 if len(original_df) > 0 else 0
        
        # Análisis por columna
        column_analysis = {}
        for column, condition in conditions.items():
            if column in original_df.columns:
                original_unique = original_df[column].nunique()
                filtered_unique = filtered_df[column].nunique()
                
                column_analysis[column] = {
                    'original_unique': original_unique,
                    'filtered_unique': filtered_unique,
                    'unique_removed': original_unique - filtered_unique,
                    'condition_applied': condition
                }
        
        return {
            'rows_removed': rows_removed,
            'removal_percentage': removal_percentage,
            'rows_remaining': len(filtered_df),
            'column_analysis': column_analysis,
            'filter_efficiency': removal_percentage  # Porcentaje de datos removidos
        }
    
    def revert_last_filter(self) -> Optional[Dict[str, Any]]:
        """
        Revierte el último filtro aplicado.
        
        Returns:
            Información del filtro revertido o None si no hay filtros
        """
        if self.current_filter_index < 0:
            return None
        
        start_time = datetime.now()
        
        try:
            # Obtener información del filtro a revertir
            reverted_filter = self.filter_history[self.current_filter_index]
            
            # Actualizar índice
            self.current_filter_index -= 1
            
            # Registrar reversión
            log_action(
                function='revert_last_filter',
                step='filtering',
                parameters={'filter_name': reverted_filter['filter_name']},
                before_metrics={'current_filter_index': self.current_filter_index + 1},
                after_metrics={'current_filter_index': self.current_filter_index},
                status='success',
                message=f"Filtro '{reverted_filter['filter_name']}' revertido",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            return reverted_filter
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            log_action(
                function='revert_last_filter',
                step='filtering',
                parameters={},
                before_metrics={},
                after_metrics={},
                status='error',
                message=f"Error al revertir filtro: {str(e)}",
                execution_time=execution_time,
                error_details=str(e)
            )
            
            raise ValueError(f"Error al revertir filtro: {str(e)}")
    
    def get_filter_history(self) -> List[Dict[str, Any]]:
        """
        Obtiene el historial completo de filtros.
        
        Returns:
            Lista con historial de filtros
        """
        return self.filter_history.copy()
    
    def get_current_filter_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual de filtros.
        
        Returns:
            Diccionario con estado actual
        """
        if self.current_filter_index < 0:
            return {
                'has_filters': False,
                'total_filters_applied': 0,
                'current_filter_index': -1
            }
        
        return {
            'has_filters': True,
            'total_filters_applied': len(self.filter_history),
            'current_filter_index': self.current_filter_index,
            'current_filter': self.filter_history[self.current_filter_index] if self.current_filter_index >= 0 else None
        }
    
    def clear_all_filters(self):
        """Limpia todo el historial de filtros."""
        self.filter_history = []
        self.current_filter_index = -1
        
        log_action(
            function='clear_all_filters',
            step='filtering',
            parameters={},
            before_metrics={'total_filters': len(self.filter_history)},
            after_metrics={'total_filters': 0},
            status='success',
            message='Todos los filtros han sido limpiados'
        )

def apply_filters(
    df: pd.DataFrame, 
    filters: Dict[str, Any],
    filter_name: str = "Filtro personalizado"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Función de conveniencia para aplicar filtros.
    
    Args:
        df: DataFrame a filtrar
        filters: Condiciones del filtro
        filter_name: Nombre del filtro
        
    Returns:
        Tuple: (DataFrame filtrado, metadata)
    """
    filter_manager = FilterManager()
    filter_manager.set_original_data(df)
    return filter_manager.apply_filter(df, filters, filter_name)

def validate_integrity_after_filtering(
    df: pd.DataFrame, 
    original_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Valida la integridad de los datos después de aplicar filtros.
    
    Args:
        df: DataFrame filtrado
        original_df: DataFrame original
        
    Returns:
        Diccionario con resultados de validación
    """
    validation_results = {
        'total_checks': 0,
        'passed_checks': 0,
        'failed_checks': 0,
        'issues': []
    }
    
    # Verificar que no se perdieron columnas
    validation_results['total_checks'] += 1
    if set(df.columns) == set(original_df.columns):
        validation_results['passed_checks'] += 1
    else:
        validation_results['failed_checks'] += 1
        validation_results['issues'].append("Columnas perdidas durante el filtrado")
    
    # Verificar que los tipos de datos se mantuvieron
    validation_results['total_checks'] += 1
    if df.dtypes.equals(original_df.dtypes):
        validation_results['passed_checks'] += 1
    else:
        validation_results['failed_checks'] += 1
        validation_results['issues'].append("Tipos de datos cambiaron durante el filtrado")
    
    # Verificar que no hay duplicados introducidos
    validation_results['total_checks'] += 1
    if not df.duplicated().any():
        validation_results['passed_checks'] += 1
    else:
        validation_results['failed_checks'] += 1
        validation_results['issues'].append("Se introdujeron duplicados durante el filtrado")
    
    # Verificar que los valores únicos son consistentes
    for column in df.columns:
        if column in original_df.columns:
            original_unique = set(original_df[column].dropna().unique())
            filtered_unique = set(df[column].dropna().unique())
            
            validation_results['total_checks'] += 1
            if filtered_unique.issubset(original_unique):
                validation_results['passed_checks'] += 1
            else:
                validation_results['failed_checks'] += 1
                validation_results['issues'].append(f"Valores únicos inconsistentes en columna '{column}'")
    
    validation_results['integrity_score'] = validation_results['passed_checks'] / validation_results['total_checks'] if validation_results['total_checks'] > 0 else 0
    
    return validation_results

class DataFilter:
    """
    Clase principal para filtrar DataFrames de ciencias sociales.
    Encapsula todas las funciones de filtrado y validación de integridad.
    """
    
    def __init__(self):
        """Inicializar el DataFilter."""
        self.filter_manager = FilterManager()
        self.original_data = None
    
    def set_data(self, df: pd.DataFrame):
        """
        Establecer los datos a filtrar.
        
        Args:
            df: DataFrame a filtrar
        """
        self.original_data = df.copy()
        self.filter_manager.set_original_data(df)
    
    def apply_filter(self, df: pd.DataFrame, filter_conditions: Dict[str, Any], filter_name: str = "Filtro personalizado") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Aplicar filtros al DataFrame.
        
        Args:
            df: DataFrame a filtrar
            filter_conditions: Condiciones del filtro
            filter_name: Nombre del filtro
            
        Returns:
            Tuple: (DataFrame filtrado, metadata del filtro)
        """
        if self.original_data is None:
            self.set_data(df)
        
        return self.filter_manager.apply_filter(df, filter_conditions, filter_name)
    
    def validate_integrity(self, filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validar la integridad después del filtrado.
        
        Args:
            filtered_df: DataFrame filtrado
            
        Returns:
            Resultados de validación
        """
        if self.original_data is None:
            return {'error': 'No hay datos originales para comparar'}
        
        return validate_integrity_after_filtering(filtered_df, self.original_data)
    
    def get_filter_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de filtros aplicados."""
        return self.filter_manager.get_filter_history()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Obtener estado actual de filtros."""
        return self.filter_manager.get_current_filter_state()
    
    def revert_last_filter(self) -> Optional[Dict[str, Any]]:
        """Revertir el último filtro aplicado."""
        return self.filter_manager.revert_last_filter()
    
    def clear_filters(self):
        """Limpiar todos los filtros."""
        self.filter_manager.clear_all_filters() 