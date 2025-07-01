"""
Módulo de Validación de Esquemas de Datos

Proporciona funcionalidades para validar que los datos cumplan con un esquema
específico, incluyendo tipos de datos, rangos, formatos y reglas personalizadas.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Representa un error de validación específico."""

    row: int
    column: str
    value: Any
    expected_type: str
    message: str
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class ValidationResult:
    """Resultado de la validación de esquema."""

    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    summary: Dict[str, Any]
    recommendations: List[str]


class SchemaValidationError(Exception):
    """Excepción personalizada para errores de validación de esquema."""

    def __init__(self, message: str, validation_result: ValidationResult):
        super().__init__(message)
        self.validation_result = validation_result


def validate_schema(
    df: pd.DataFrame,
    schema: Dict[str, Dict[str, Any]],
    strict: bool = True,
    auto_correct: bool = False,
) -> ValidationResult:
    """
    Valida que un DataFrame cumpla con un esquema específico.

    Args:
        df: DataFrame a validar
        schema: Diccionario con el esquema esperado
                {
                    'columna': {
                        'type': 'numeric|text|categorical|date|boolean',
                        'required': True/False,
                        'min_value': float/int,
                        'max_value': float/int,
                        'allowed_values': [valores],
                        'pattern': 'regex_pattern',
                        'date_format': 'YYYY-MM-DD',
                        'nullable': True/False
                    }
                }
        strict: Si es True, falla en el primer error crítico
        auto_correct: Si es True, intenta corregir errores automáticamente

    Returns:
        ValidationResult con el resultado de la validación
    """
    errors = []
    warnings = []
    recommendations = []

    # 1. Validar que todas las columnas requeridas estén presentes
    missing_columns = []
    for col, rules in schema.items():
        if rules.get("required", True) and col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        error_msg = f"Columnas faltantes: {missing_columns}"
        if strict:
            raise SchemaValidationError(
                error_msg, ValidationResult(False, [], [], {}, [])
            )
        else:
            errors.append(
                ValidationError(
                    row=-1,
                    column="",
                    value="",
                    expected_type="",
                    message=error_msg,
                    severity="error",
                )
            )

    # 2. Validar cada columna según su esquema
    for col, rules in schema.items():
        if col not in df.columns:
            continue

        col_errors, col_warnings, col_recommendations = _validate_column(
            df, col, rules, auto_correct
        )
        errors.extend(col_errors)
        warnings.extend(col_warnings)
        recommendations.extend(col_recommendations)

    # 3. Crear resumen
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns_validated": len([col for col in schema.keys() if col in df.columns]),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "missing_columns": missing_columns,
    }

    is_valid = len(errors) == 0 or not strict

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        summary=summary,
        recommendations=recommendations,
    )


def _validate_column(
    df: pd.DataFrame, column: str, rules: Dict[str, Any], auto_correct: bool
) -> Tuple[List[ValidationError], List[ValidationError], List[str]]:
    """Valida una columna específica según las reglas definidas."""
    errors = []
    warnings = []
    recommendations = []

    expected_type = rules.get("type", "text")
    nullable = rules.get("nullable", True)

    # Validar valores nulos
    if not nullable and df[column].isnull().any():
        null_rows = df[df[column].isnull()].index.tolist()
        for row in null_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=None,
                    expected_type=expected_type,
                    message="Valor nulo no permitido",
                    severity="error",
                )
            )

    # Filtrar valores no nulos para validaciones adicionales
    non_null_mask = df[column].notna()
    non_null_data = df.loc[non_null_mask, column]

    if len(non_null_data) == 0:
        return errors, warnings, recommendations

    # Validar según el tipo
    if expected_type == "numeric":
        col_errors, col_warnings, col_recs = _validate_numeric(
            non_null_data, column, rules, auto_correct
        )
    elif expected_type == "text":
        col_errors, col_warnings, col_recs = _validate_text(
            non_null_data, column, rules, auto_correct
        )
    elif expected_type == "categorical":
        col_errors, col_warnings, col_recs = _validate_categorical(
            non_null_data, column, rules, auto_correct
        )
    elif expected_type == "date":
        col_errors, col_warnings, col_recs = _validate_date(
            non_null_data, column, rules, auto_correct
        )
    elif expected_type == "boolean":
        col_errors, col_warnings, col_recs = _validate_boolean(
            non_null_data, column, rules, auto_correct
        )
    else:
        warnings.append(
            ValidationError(
                row=-1,
                column=column,
                value="",
                expected_type=expected_type,
                message=f"Tipo '{expected_type}' no reconocido",
                severity="warning",
            )
        )
        col_errors, col_warnings, col_recs = [], [], []

    errors.extend(col_errors)
    warnings.extend(col_warnings)
    recommendations.extend(col_recs)

    return errors, warnings, recommendations


def _validate_numeric(
    data: pd.Series, column: str, rules: Dict[str, Any], auto_correct: bool
) -> Tuple[List[ValidationError], List[ValidationError], List[str]]:
    """Valida datos numéricos."""
    errors = []
    warnings = []
    recommendations = []

    # Verificar si los datos son numéricos
    if not pd.api.types.is_numeric_dtype(data):
        # Intentar convertir si auto_correct está habilitado
        if auto_correct:
            try:
                data = pd.to_numeric(data, errors="coerce")
                recommendations.append(f"Columna '{column}' convertida a numérico")
            except:
                pass

        # Marcar errores para valores no convertibles
        non_numeric_mask = pd.to_numeric(data, errors="coerce").isna()
        non_numeric_rows = data[non_numeric_mask].index.tolist()

        for row in non_numeric_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=data.loc[row],
                    expected_type="numeric",
                    message="Valor no es numérico",
                    severity="error",
                )
            )

    # Validar rangos
    min_value = rules.get("min_value")
    max_value = rules.get("max_value")

    if min_value is not None:
        below_min = data < min_value
        below_min_rows = data[below_min].index.tolist()
        for row in below_min_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=data.loc[row],
                    expected_type="numeric",
                    message=f"Valor {data.loc[row]} está por debajo del mínimo {min_value}",
                    severity="error",
                )
            )

    if max_value is not None:
        above_max = data > max_value
        above_max_rows = data[above_max].index.tolist()
        for row in above_max_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=data.loc[row],
                    expected_type="numeric",
                    message=f"Valor {data.loc[row]} está por encima del máximo {max_value}",
                    severity="error",
                )
            )

    return errors, warnings, recommendations


def _validate_text(
    data: pd.Series, column: str, rules: Dict[str, Any], auto_correct: bool
) -> Tuple[List[ValidationError], List[ValidationError], List[str]]:
    """Valida datos de texto."""
    errors = []
    warnings = []
    recommendations = []

    # Validar patrón regex
    pattern = rules.get("pattern")
    if pattern:
        try:
            regex = re.compile(pattern)
            non_matching_mask = ~data.astype(str).str.match(regex, na=False)
            non_matching_rows = data[non_matching_mask].index.tolist()

            for row in non_matching_rows:
                errors.append(
                    ValidationError(
                        row=row,
                        column=column,
                        value=data.loc[row],
                        expected_type="text",
                        message=f"Valor no coincide con el patrón '{pattern}'",
                        severity="error",
                    )
                )
        except re.error:
            warnings.append(
                ValidationError(
                    row=-1,
                    column=column,
                    value="",
                    expected_type="text",
                    message=f"Patrón regex inválido: '{pattern}'",
                    severity="warning",
                )
            )

    # Validar longitud mínima/máxima
    min_length = rules.get("min_length")
    max_length = rules.get("max_length")

    if min_length is not None:
        too_short = data.astype(str).str.len() < min_length
        too_short_rows = data[too_short].index.tolist()
        for row in too_short_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=data.loc[row],
                    expected_type="text",
                    message=f"Texto demasiado corto (mínimo {min_length} caracteres)",
                    severity="error",
                )
            )

    if max_length is not None:
        too_long = data.astype(str).str.len() > max_length
        too_long_rows = data[too_long].index.tolist()
        for row in too_long_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=data.loc[row],
                    expected_type="text",
                    message=f"Texto demasiado largo (máximo {max_length} caracteres)",
                    severity="error",
                )
            )

    return errors, warnings, recommendations


def _validate_categorical(
    data: pd.Series, column: str, rules: Dict[str, Any], auto_correct: bool
) -> Tuple[List[ValidationError], List[ValidationError], List[str]]:
    """Valida datos categóricos."""
    errors = []
    warnings = []
    recommendations = []

    allowed_values = rules.get("allowed_values")
    if allowed_values:
        invalid_values = ~data.isin(allowed_values)
        invalid_rows = data[invalid_values].index.tolist()

        for row in invalid_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=data.loc[row],
                    expected_type="categorical",
                    message=f"Valor '{data.loc[row]}' no está en la lista permitida: {allowed_values}",
                    severity="error",
                )
            )

    # Verificar cardinalidad
    max_categories = rules.get("max_categories")
    if max_categories and data.nunique() > max_categories:
        warnings.append(
            ValidationError(
                row=-1,
                column=column,
                value="",
                expected_type="categorical",
                message=f"Demasiadas categorías ({data.nunique()} > {max_categories})",
                severity="warning",
            )
        )

    return errors, warnings, recommendations


def _validate_date(
    data: pd.Series, column: str, rules: Dict[str, Any], auto_correct: bool
) -> Tuple[List[ValidationError], List[ValidationError], List[str]]:
    """Valida datos de fecha."""
    errors = []
    warnings = []
    recommendations = []

    date_format = rules.get("date_format", "%Y-%m-%d")

    # Intentar convertir a datetime
    try:
        if auto_correct:
            data = pd.to_datetime(data, errors="coerce", format=date_format)
            recommendations.append(f"Columna '{column}' convertida a datetime")
        else:
            # Solo validar sin convertir
            pd.to_datetime(data, errors="raise", format=date_format)
    except:
        # Marcar errores para fechas inválidas
        invalid_dates = pd.to_datetime(data, errors="coerce").isna()
        invalid_rows = data[invalid_dates].index.tolist()

        for row in invalid_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=data.loc[row],
                    expected_type="date",
                    message=f"Fecha inválida o formato incorrecto (esperado: {date_format})",
                    severity="error",
                )
            )

    # Validar rangos de fecha
    min_date = rules.get("min_date")
    max_date = rules.get("max_date")

    if min_date is not None:
        if isinstance(min_date, str):
            min_date = pd.to_datetime(min_date)
        before_min = data < min_date
        before_min_rows = data[before_min].index.tolist()
        for row in before_min_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=data.loc[row],
                    expected_type="date",
                    message=f"Fecha anterior al mínimo permitido: {min_date}",
                    severity="error",
                )
            )

    if max_date is not None:
        if isinstance(max_date, str):
            max_date = pd.to_datetime(max_date)
        after_max = data > max_date
        after_max_rows = data[after_max].index.tolist()
        for row in after_max_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=data.loc[row],
                    expected_type="date",
                    message=f"Fecha posterior al máximo permitido: {max_date}",
                    severity="error",
                )
            )

    return errors, warnings, recommendations


def _validate_boolean(
    data: pd.Series, column: str, rules: Dict[str, Any], auto_correct: bool
) -> Tuple[List[ValidationError], List[ValidationError], List[str]]:
    """Valida datos booleanos."""
    errors = []
    warnings = []
    recommendations = []

    # Verificar si los datos son booleanos
    if not pd.api.types.is_bool_dtype(data):
        # Intentar convertir si auto_correct está habilitado
        if auto_correct:
            try:
                # Mapear valores comunes a booleanos
                bool_map = {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "yes": True,
                    "no": False,
                    "si": True,
                    "no": False,
                    "sí": True,
                    "no": False,
                }
                data = data.astype(str).str.lower().map(bool_map)
                recommendations.append(f"Columna '{column}' convertida a booleano")
            except:
                pass

        # Marcar errores para valores no convertibles
        non_bool_mask = ~data.isin([True, False])
        non_bool_rows = data[non_bool_mask].index.tolist()

        for row in non_bool_rows:
            errors.append(
                ValidationError(
                    row=row,
                    column=column,
                    value=data.loc[row],
                    expected_type="boolean",
                    message="Valor no es booleano válido",
                    severity="error",
                )
            )

    return errors, warnings, recommendations


def create_validation_report(validation_result: ValidationResult) -> str:
    """Crea un reporte legible de la validación."""
    report = []
    report.append("=" * 60)
    report.append("REPORTE DE VALIDACIÓN DE ESQUEMA")
    report.append("=" * 60)

    # Resumen
    summary = validation_result.summary
    report.append(f"📊 Resumen:")
    report.append(f"   • Filas totales: {summary['total_rows']:,}")
    report.append(f"   • Columnas validadas: {summary['columns_validated']}")
    report.append(f"   • Errores encontrados: {summary['error_count']}")
    report.append(f"   • Advertencias: {summary['warning_count']}")

    if summary["missing_columns"]:
        report.append(f"   • Columnas faltantes: {summary['missing_columns']}")

    report.append("")

    # Errores
    if validation_result.errors:
        report.append("❌ ERRORES:")
        for error in validation_result.errors[:10]:  # Mostrar solo los primeros 10
            report.append(
                f"   • Fila {error.row}, Columna '{error.column}': {error.message}"
            )
            report.append(
                f"     Valor: {error.value} | Esperado: {error.expected_type}"
            )

        if len(validation_result.errors) > 10:
            report.append(f"   ... y {len(validation_result.errors) - 10} errores más")
        report.append("")

    # Advertencias
    if validation_result.warnings:
        report.append("⚠️ ADVERTENCIAS:")
        for warning in validation_result.warnings[:5]:  # Mostrar solo las primeras 5
            report.append(f"   • {warning.message}")
        report.append("")

    # Recomendaciones
    if validation_result.recommendations:
        report.append("💡 RECOMENDACIONES:")
        for rec in validation_result.recommendations:
            report.append(f"   • {rec}")
        report.append("")

    # Estado final
    status = "✅ VÁLIDO" if validation_result.is_valid else "❌ INVÁLIDO"
    report.append(f"Estado final: {status}")
    report.append("=" * 60)

    return "\n".join(report)
