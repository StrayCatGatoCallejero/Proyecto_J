"""
Módulo de Validación de Datos Robusto - Proyecto J
=================================================

Responsabilidades:
- Validación de tipos de datos con esquemas flexibles
- Validación de rangos y valores permitidos
- Detección de outliers y valores anómalos
- Validación de integridad referencial
- Validación de formatos específicos (email, teléfono, etc.)
- Validación de seguridad contra inyección de datos maliciosos
- Logging detallado de validaciones
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, date
import re
import warnings
import logging
from dataclasses import dataclass, field
from pathlib import Path

# Importar módulos del sistema
from .logging import log_action
from .error_reporter import report_dataframe_error

# Configurar logging
logger = logging.getLogger(__name__)

# Suprimir warnings de pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ValidationRule:
    """Regla de validación configurable"""
    name: str
    field: str
    rule_type: str  # 'type', 'range', 'format', 'custom', 'security'
    validator: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None


@dataclass
class ValidationResult:
    """Resultado de una validación"""
    is_valid: bool
    rule_name: str
    field_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "info"  # info, warning, error, critical


@dataclass
class ValidationReport:
    """Reporte completo de validación"""
    overall_valid: bool
    total_validations: int
    passed_validations: int
    failed_validations: int
    warnings: int
    errors: int
    critical_errors: int
    results: List[ValidationResult] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class DataFrameSchema:
    """Esquema de validación para DataFrames"""
    
    def __init__(self, min_rows: int = 0, max_rows: Optional[int] = None, 
                 min_columns: int = 0, max_columns: Optional[int] = None,
                 required_columns: Optional[List[str]] = None,
                 column_types: Optional[Dict[str, str]] = None):
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.min_columns = min_columns
        self.max_columns = max_columns
        self.required_columns = required_columns or []
        self.column_types = column_types or {}
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Valida un DataFrame según el esquema"""
        errors = []
        warnings = []
        
        # Validar número de filas
        if len(df) < self.min_rows:
            errors.append(f"DataFrame debe tener al menos {self.min_rows} filas")
        
        if self.max_rows and len(df) > self.max_rows:
            errors.append(f"DataFrame debe tener máximo {self.max_rows} filas")
        
        # Validar número de columnas
        if len(df.columns) < self.min_columns:
            errors.append(f"DataFrame debe tener al menos {self.min_columns} columnas")
        
        if self.max_columns and len(df.columns) > self.max_columns:
            errors.append(f"DataFrame debe tener máximo {self.max_columns} columnas")
        
        # Validar columnas requeridas
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Columnas requeridas faltantes: {missing_columns}")
        
        # Validar tipos de columna
        for col, expected_type in self.column_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    warnings.append(f"Columna {col}: tipo esperado {expected_type}, actual {actual_type}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            rule_name="DataFrameSchema",
            field_name="dataframe",
            message="Validación de esquema completada",
            details={
                "errors": errors,
                "warnings": warnings,
                "shape": df.shape,
                "columns": list(df.columns)
            },
            severity="error" if errors else "warning" if warnings else "info"
        )


def create_dataframe_schema(min_rows: int = 0, max_rows: Optional[int] = None,
                          min_columns: int = 0, max_columns: Optional[int] = None,
                          required_columns: Optional[List[str]] = None,
                          column_types: Optional[Dict[str, str]] = None):
    """Función de conveniencia para crear esquemas de DataFrame"""
    return DataFrameSchema(
        min_rows=min_rows,
        max_rows=max_rows,
        min_columns=min_columns,
        max_columns=max_columns,
        required_columns=required_columns,
        column_types=column_types
    )


class DataValidator:
    """
    Validador de datos robusto con múltiples tipos de validación.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el validador con configuración opcional.
        
        Args:
            config: Configuración de validación
        """
        self.config = config or {}
        self.rules: List[ValidationRule] = []
        self.custom_validators: Dict[str, Callable] = {}
        
        # Configurar validadores por defecto
        self._setup_default_validators()
    
    def _setup_default_validators(self) -> None:
        """Configura validadores por defecto"""
        
        # Validadores de tipo
        self.custom_validators.update({
            'email': self._validate_email,
            'phone': self._validate_phone,
            'dni': self._validate_dni,
            'age': self._validate_age,
            'percentage': self._validate_percentage,
            'currency': self._validate_currency,
            'date': self._validate_date,
            'url': self._validate_url,
            'postal_code': self._validate_postal_code,
        })
        
        # Validadores de seguridad
        self.custom_validators.update({
            'sql_injection': self._validate_sql_injection,
            'xss': self._validate_xss,
            'script_injection': self._validate_script_injection,
            'path_traversal': self._validate_path_traversal,
        })
    
    def add_rule(self, rule: ValidationRule) -> None:
        """
        Agrega una regla de validación personalizada.
        
        Args:
            rule: Regla de validación a agregar
        """
        self.rules.append(rule)
        logger.info(f"✅ Regla de validación agregada: {rule.name}")
    
    def add_type_rule(self, field: str, expected_type: str, **kwargs) -> None:
        """
        Agrega una regla de validación de tipo.
        
        Args:
            field: Nombre del campo a validar
            expected_type: Tipo esperado (int, float, str, bool, date, etc.)
            **kwargs: Parámetros adicionales
        """
        rule = ValidationRule(
            name=f"type_check_{field}",
            field=field,
            rule_type="type",
            validator=self._validate_type,
            parameters={"expected_type": expected_type, **kwargs},
            error_message=f"El campo {field} debe ser de tipo {expected_type}"
        )
        self.add_rule(rule)
    
    def add_range_rule(self, field: str, min_val: Optional[float] = None, 
                      max_val: Optional[float] = None, **kwargs) -> None:
        """
        Agrega una regla de validación de rango.
        
        Args:
            field: Nombre del campo a validar
            min_val: Valor mínimo permitido
            max_val: Valor máximo permitido
            **kwargs: Parámetros adicionales
        """
        rule = ValidationRule(
            name=f"range_check_{field}",
            field=field,
            rule_type="range",
            validator=self._validate_range,
            parameters={"min_val": min_val, "max_val": max_val, **kwargs},
            error_message=f"El campo {field} debe estar entre {min_val} y {max_val}"
        )
        self.add_rule(rule)
    
    def add_format_rule(self, field: str, format_type: str, **kwargs) -> None:
        """
        Agrega una regla de validación de formato.
        
        Args:
            field: Nombre del campo a validar
            format_type: Tipo de formato (email, phone, dni, etc.)
            **kwargs: Parámetros adicionales
        """
        if format_type not in self.custom_validators:
            raise ValueError(f"Formato no soportado: {format_type}")
        
        rule = ValidationRule(
            name=f"format_check_{field}",
            field=field,
            rule_type="format",
            validator=self.custom_validators[format_type],
            parameters=kwargs,
            error_message=f"El campo {field} no tiene el formato {format_type} válido"
        )
        self.add_rule(rule)
    
    def add_security_rule(self, field: str, security_type: str, **kwargs) -> None:
        """
        Agrega una regla de validación de seguridad.
        
        Args:
            field: Nombre del campo a validar
            security_type: Tipo de validación de seguridad
            **kwargs: Parámetros adicionales
        """
        security_validators = {
            'sql_injection': self._validate_sql_injection,
            'xss': self._validate_xss,
            'script_injection': self._validate_script_injection,
            'path_traversal': self._validate_path_traversal,
        }
        
        if security_type not in security_validators:
            raise ValueError(f"Tipo de seguridad no soportado: {security_type}")
        
        rule = ValidationRule(
            name=f"security_check_{field}",
            field=field,
            rule_type="security",
            validator=security_validators[security_type],
            parameters=kwargs,
            error_message=f"El campo {field} contiene contenido potencialmente peligroso",
            severity="critical"
        )
        self.add_rule(rule)
    
    def validate_dataframe(self, df: pd.DataFrame, context: str = "") -> ValidationReport:
        """
        Valida un DataFrame completo aplicando todas las reglas configuradas.
        
        Args:
            df: DataFrame a validar
            context: Contexto de validación para logging
            
        Returns:
            Reporte completo de validación
        """
        start_time = datetime.now()
        
        if df is None or df.empty:
            return ValidationReport(
                overall_valid=False,
                total_validations=0,
                passed_validations=0,
                failed_validations=1,
                warnings=0,
                errors=1,
                critical_errors=0,
                results=[
                    ValidationResult(
                        is_valid=False,
                        rule_name="dataframe_empty",
                        field_name="dataframe",
                        message="El DataFrame está vacío o es None",
                        severity="error"
                    )
                ]
            )
        
        results = []
        passed = 0
        failed = 0
        warnings_count = 0
        errors_count = 0
        critical_count = 0
        
        # Aplicar reglas de validación
        for rule in self.rules:
            if rule.field in df.columns:
                try:
                    result = rule.validator(df[rule.field], rule.parameters)
                    
                    if isinstance(result, bool):
                        # Validador simple que retorna True/False
                        is_valid = result
                        details = {}
                    elif isinstance(result, dict):
                        # Validador que retorna diccionario con detalles
                        is_valid = result.get('is_valid', False)
                        details = result.get('details', {})
                    else:
                        # Validador que retorna ValidationResult
                        results.append(result)
                        if result.is_valid:
                            passed += 1
                        else:
                            failed += 1
                            if result.severity == "critical":
                                critical_count += 1
                            elif result.severity == "error":
                                errors_count += 1
                            elif result.severity == "warning":
                                warnings_count += 1
                        continue
                    
                    # Crear ValidationResult
                    validation_result = ValidationResult(
                        is_valid=is_valid,
                        rule_name=rule.name,
                        field_name=rule.field,
                        message=rule.error_message if not is_valid else "Validación exitosa",
                        details=details,
                        severity=rule.severity or ("error" if not is_valid else "info")
                    )
                    
                    results.append(validation_result)
                    
                    if is_valid:
                        passed += 1
                    else:
                        failed += 1
                        if validation_result.severity == "critical":
                            critical_count += 1
                        elif validation_result.severity == "error":
                            errors_count += 1
                        elif validation_result.severity == "warning":
                            warnings_count += 1
                
                except Exception as e:
                    # Error en la validación
                    error_result = ValidationResult(
                        is_valid=False,
                        rule_name=rule.name,
                        field_name=rule.field,
                        message=f"Error en validación: {str(e)}",
                        details={"error": str(e)},
                        severity="error"
                    )
                    results.append(error_result)
                    failed += 1
                    errors_count += 1
        
        # Validaciones adicionales automáticas
        auto_results = self._run_automatic_validations(df)
        results.extend(auto_results)
        
        for result in auto_results:
            if result.is_valid:
                passed += 1
            else:
                failed += 1
                if result.severity == "critical":
                    critical_count += 1
                elif result.severity == "error":
                    errors_count += 1
                elif result.severity == "warning":
                    warnings_count += 1
        
        execution_time = (datetime.now() - start_time).total_seconds()
        overall_valid = critical_count == 0 and errors_count == 0
        
        report = ValidationReport(
            overall_valid=overall_valid,
            total_validations=len(results),
            passed_validations=passed,
            failed_validations=failed,
            warnings=warnings_count,
            errors=errors_count,
            critical_errors=critical_count,
            results=results,
            execution_time=execution_time
        )
        
        # Log del reporte
        logger.info(f"📊 Validación completada: {passed} pasadas, {failed} fallidas")
        if critical_count > 0:
            logger.critical(f"🚨 {critical_count} errores críticos detectados")
        if errors_count > 0:
            logger.error(f"❌ {errors_count} errores detectados")
        if warnings_count > 0:
            logger.warning(f"⚠️ {warnings_count} advertencias detectadas")
        
        return report
    
    def _run_automatic_validations(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Ejecuta validaciones automáticas adicionales"""
        results = []
        
        # Validar duplicados
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="duplicate_check",
                field_name="dataframe",
                message=f"Se encontraron {duplicate_count} filas duplicadas",
                details={"duplicate_count": duplicate_count},
                severity="warning"
            ))
        
        # Validar columnas completamente vacías
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="empty_columns_check",
                field_name="dataframe",
                message=f"Columnas completamente vacías: {empty_columns}",
                details={"empty_columns": empty_columns},
                severity="warning"
            ))
        
        # Validar tipos de datos mixtos
        mixed_type_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Verificar si hay mezcla de tipos
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    type_counts = non_null_values.apply(type).value_counts()
                    if len(type_counts) > 1:
                        mixed_type_columns.append(col)
        
        if mixed_type_columns:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="mixed_types_check",
                field_name="dataframe",
                message=f"Columnas con tipos mixtos: {mixed_type_columns}",
                details={"mixed_type_columns": mixed_type_columns},
                severity="warning"
            ))
        
        return results
    
    # Validadores de tipo
    def _validate_type(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida el tipo de datos de una serie"""
        expected_type = params.get('expected_type', 'object')
        
        if expected_type == 'int':
            return pd.api.types.is_integer_dtype(series) or series.apply(
                lambda x: pd.isna(x) or (isinstance(x, (int, np.integer)) and not isinstance(x, bool))
            ).all()
        elif expected_type == 'float':
            return pd.api.types.is_float_dtype(series) or series.apply(
                lambda x: pd.isna(x) or isinstance(x, (float, np.floating))
            ).all()
        elif expected_type == 'str':
            return pd.api.types.is_string_dtype(series) or series.apply(
                lambda x: pd.isna(x) or isinstance(x, str)
            ).all()
        elif expected_type == 'bool':
            return pd.api.types.is_bool_dtype(series) or series.apply(
                lambda x: pd.isna(x) or isinstance(x, bool)
            ).all()
        elif expected_type == 'date':
            return pd.api.types.is_datetime64_any_dtype(series) or series.apply(
                lambda x: pd.isna(x) or isinstance(x, (datetime, date))
            ).all()
        
        return True
    
    def _validate_range(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida el rango de valores de una serie numérica"""
        min_val = params.get('min_val')
        max_val = params.get('max_val')
        
        if min_val is None and max_val is None:
            return True
        
        # Convertir a numérico
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        if min_val is not None:
            if (numeric_series < min_val).any():
                return False
        
        if max_val is not None:
            if (numeric_series > max_val).any():
                return False
        
        return True
    
    # Validadores de formato
    def _validate_email(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida formato de email"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return series.apply(lambda x: pd.isna(x) or bool(re.match(email_pattern, str(x)))).all()
    
    def _validate_phone(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida formato de teléfono"""
        phone_pattern = r'^[+]?[0-9]{8,15}$'
        return series.apply(lambda x: pd.isna(x) or bool(re.match(phone_pattern, str(x)))).all()
    
    def _validate_dni(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida formato de DNI"""
        dni_pattern = r'^[0-9]{7,8}[A-Z]$'
        return series.apply(lambda x: pd.isna(x) or bool(re.match(dni_pattern, str(x)))).all()
    
    def _validate_age(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida rango de edad"""
        min_age = params.get('min_age', 0)
        max_age = params.get('max_age', 120)
        return self._validate_range(series, {'min_val': min_age, 'max_val': max_age})
    
    def _validate_percentage(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida porcentajes (0-100)"""
        return self._validate_range(series, {'min_val': 0, 'max_val': 100})
    
    def _validate_currency(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida valores monetarios"""
        min_val = params.get('min_val', 0)
        return self._validate_range(series, {'min_val': min_val})
    
    def _validate_date(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida fechas"""
        try:
            pd.to_datetime(series, errors='raise')
            return True
        except:
            return False
    
    def _validate_url(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida URLs"""
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return series.apply(lambda x: pd.isna(x) or bool(re.match(url_pattern, str(x)))).all()
    
    def _validate_postal_code(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Valida códigos postales"""
        postal_pattern = r'^[0-9]{5,7}$'
        return series.apply(lambda x: pd.isna(x) or bool(re.match(postal_pattern, str(x)))).all()
    
    # Validadores de seguridad
    def _validate_sql_injection(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Detecta posibles intentos de inyección SQL"""
        sql_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
            r'(\b(OR|AND)\b\s+\d+\s*=\s*\d+)',
            r'(\b(OR|AND)\b\s+\'[^\']*\'\s*=\s*\'[^\']*\')',
            r'(\b(OR|AND)\b\s+\d+\s*=\s*\d+\s*--)',
            r'(\b(OR|AND)\b\s+\'[^\']*\'\s*=\s*\'[^\']*\'--)',
        ]
        
        for pattern in sql_patterns:
            if series.apply(lambda x: pd.notna(x) and bool(re.search(pattern, str(x), re.IGNORECASE))).any():
                return False
        return True
    
    def _validate_xss(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Detecta posibles ataques XSS"""
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
        ]
        
        for pattern in xss_patterns:
            if series.apply(lambda x: pd.notna(x) and bool(re.search(pattern, str(x), re.IGNORECASE))).any():
                return False
        return True
    
    def _validate_script_injection(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Detecta posibles inyecciones de script"""
        script_patterns = [
            r'<script',
            r'javascript:',
            r'vbscript:',
            r'data:text/html',
            r'data:application/x-javascript',
        ]
        
        for pattern in script_patterns:
            if series.apply(lambda x: pd.notna(x) and bool(re.search(pattern, str(x), re.IGNORECASE))).any():
                return False
        return True
    
    def _validate_path_traversal(self, series: pd.Series, params: Dict[str, Any]) -> bool:
        """Detecta posibles ataques de path traversal"""
        path_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'%2e%2e%2f',
            r'%2e%2e%5c',
            r'\.\.%2f',
            r'\.\.%5c',
        ]
        
        for pattern in path_patterns:
            if series.apply(lambda x: pd.notna(x) and bool(re.search(pattern, str(x), re.IGNORECASE))).any():
                return False
        return True


# Funciones de conveniencia
def create_validator(config: Optional[Dict[str, Any]] = None) -> DataValidator:
    """
    Crea un validador de datos con configuración predefinida.
    
    Args:
        config: Configuración del validador
        
    Returns:
        Instancia de DataValidator configurada
    """
    validator = DataValidator(config)
    
    # Agregar reglas comunes si no se especifican
    if not config or not config.get('skip_default_rules', False):
        # Reglas básicas de validación
        validator.add_type_rule('edad', 'int')
        validator.add_range_rule('edad', min_val=0, max_val=120)
        
        validator.add_type_rule('ingresos', 'float')
        validator.add_range_rule('ingresos', min_val=0)
        
        validator.add_format_rule('email', 'email')
        validator.add_format_rule('telefono', 'phone')
        validator.add_format_rule('dni', 'dni')
        
        # Reglas de seguridad para campos de texto
        text_fields = ['comentarios', 'descripcion', 'notas', 'observaciones']
        for field in text_fields:
            validator.add_security_rule(field, 'sql_injection')
            validator.add_security_rule(field, 'xss')
            validator.add_security_rule(field, 'script_injection')
    
    return validator


def validate_dataframe_simple(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> ValidationReport:
    """
    Valida un DataFrame con reglas simples especificadas como diccionarios.
    
    Args:
        df: DataFrame a validar
        rules: Lista de reglas de validación
        
    Returns:
        Reporte de validación
    """
    validator = create_validator()
    
    for rule in rules:
        rule_type = rule.get('type', 'type')
        
        if rule_type == 'type':
            validator.add_type_rule(rule['field'], rule['expected_type'])
        elif rule_type == 'range':
            validator.add_range_rule(rule['field'], rule.get('min'), rule.get('max'))
        elif rule_type == 'format':
            validator.add_format_rule(rule['field'], rule['format_type'])
        elif rule_type == 'security':
            validator.add_security_rule(rule['field'], rule['security_type'])
    
    return validator.validate_dataframe(df)


def validate_dataframe(df: pd.DataFrame, schema: DataFrameSchema, context: str = "") -> ValidationResult:
    """
    Función independiente para validar un DataFrame contra un esquema.
    
    Args:
        df: DataFrame a validar
        schema: Esquema de validación
        context: Contexto de la validación
        
    Returns:
        Resultado de la validación
    """
    return schema.validate(df)


class FeatureSelectionParams:
    """
    Parámetros para selección de características.
    """
    def __init__(self, 
                 method: str = "correlation",
                 threshold: float = 0.8,
                 max_features: int = None,
                 target_column: str = None):
        self.method = method
        self.threshold = threshold
        self.max_features = max_features
        self.target_column = target_column 