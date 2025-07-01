"""
Módulo de Validación de Reglas de Negocio - Ciencias Sociales
============================================================

Responsabilidades:
- Validación de datos demográficos (edad, género, educación, ingresos)
- Validación de escalas Likert y encuestas
- Validación de datos geográficos (regiones, comunas)
- Validaciones de consistencia cruzada
- Integración con el sistema híbrido de errores
- Logging detallado de validaciones
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import warnings
from scipy import stats
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# Importar módulos del sistema
from .logging import log_action
from .error_reporter import report_business_rule_error
from .validation_decorators import validate_io, create_dataframe_schema


class BusinessRuleError(Exception):
    """Excepción específica para errores de reglas de negocio"""
    def __init__(self, message: str, rule_name: str, details: dict[str, Any]):
        self.message = message
        self.rule_name = rule_name
        self.details = details
        super().__init__(self.message)


class ValidationResult:
    """Resultado de una validación de regla de negocio"""
    def __init__(self, is_valid: bool, rule_name: str, message: str, details: dict[str, Any]):
        self.is_valid = is_valid
        self.rule_name = rule_name
        self.message = message
        self.details = details
        self.timestamp = datetime.now()


# Configuraciones predefinidas para ciencias sociales
DEMOGRAPHIC_CONFIG = {
    'edad': {
        'min': 0,
        'max': 120,
        'outlier_threshold': 3.0  # Z-score para outliers
    },
    'genero': {
        'valores_validos': ['Masculino', 'Femenino', 'Otro', 'M', 'F', 'O', 'Hombre', 'Mujer'],
        'mapeo': {
            'M': 'Masculino', 'F': 'Femenino', 'O': 'Otro',
            'Hombre': 'Masculino', 'Mujer': 'Femenino'
        }
    },
    'nivel_educacion': {
        'categorias': [
            'Sin educación', 'Primaria', 'Secundaria', 'Técnica', 
            'Universitaria', 'Postgrado', 'Doctorado'
        ],
        'mapeo': {
            'Básica': 'Primaria', 'Media': 'Secundaria', 'Superior': 'Universitaria',
            'Magíster': 'Postgrado', 'PhD': 'Doctorado'
        }
    },
    'ingresos': {
        'min': 0,
        'outlier_multiplier': 5.0  # 5x mediana nacional
    }
}

LIKERT_SCALES = {
    '5_puntos': {
        'valores': [1, 2, 3, 4, 5],
        'etiquetas': ['Muy en desacuerdo', 'En desacuerdo', 'Neutral', 'De acuerdo', 'Muy de acuerdo']
    },
    '7_puntos': {
        'valores': [1, 2, 3, 4, 5, 6, 7],
        'etiquetas': ['Muy en desacuerdo', 'En desacuerdo', 'Ligeramente en desacuerdo', 
                     'Neutral', 'Ligeramente de acuerdo', 'De acuerdo', 'Muy de acuerdo']
    }
}

# Datos geográficos de Chile (ejemplo)
CHILE_REGIONS = {
    'Arica y Parinacota': ['Arica', 'Camarones', 'Putre', 'General Lagos'],
    'Tarapacá': ['Iquique', 'Alto Hospicio', 'Pozo Almonte', 'Camiña', 'Colchane', 'Huara', 'Pica'],
    'Antofagasta': ['Antofagasta', 'Mejillones', 'Sierra Gorda', 'Taltal', 'Calama', 'Ollagüe', 'San Pedro de Atacama', 'Tocopilla', 'María Elena'],
    # ... más regiones
}

CHILE_COMMUNAS = {
    'Arica': 'Arica y Parinacota',
    'Camarones': 'Arica y Parinacota',
    'Putre': 'Arica y Parinacota',
    'General Lagos': 'Arica y Parinacota',
    'Iquique': 'Tarapacá',
    'Alto Hospicio': 'Tarapacá',
    # ... más comunas
}


def validate_demographics(df: pd.DataFrame, metadata: dict[str, Any]) -> ValidationResult:
    """
    Valida datos demográficos según reglas de negocio.
    
    Args:
        df: DataFrame con datos demográficos
        metadata: Metadatos del dataset
        
    Returns:
        ValidationResult con resultados de validación
    """
    start_time = datetime.now()
    errors = []
    warnings = []
    details = {
        'columnas_validadas': [],
        'filas_con_errores': 0,
        'outliers_detectados': 0
    }
    
    # Validar edad
    if 'edad' in df.columns:
        details['columnas_validadas'].append('edad')
        edad_series = pd.to_numeric(df['edad'], errors='coerce')
        
        # Rango válido
        invalid_edad = edad_series[(edad_series < DEMOGRAPHIC_CONFIG['edad']['min']) | 
                                  (edad_series > DEMOGRAPHIC_CONFIG['edad']['max'])]
        if len(invalid_edad) > 0:
            errors.append(f"Edad fuera de rango válido [{DEMOGRAPHIC_CONFIG['edad']['min']}, {DEMOGRAPHIC_CONFIG['edad']['max']}]: {len(invalid_edad)} filas")
            details['filas_con_errores'] += len(invalid_edad)
        
        # Detectar outliers
        edad_clean = edad_series.dropna()
        if len(edad_clean) > 0:
            z_scores = np.abs(stats.zscore(edad_clean))
            outliers = edad_clean[z_scores > DEMOGRAPHIC_CONFIG['edad']['outlier_threshold']]
            if len(outliers) > 0:
                warnings.append(f"Posibles outliers en edad detectados: {len(outliers)} valores")
                details['outliers_detectados'] += len(outliers)
    
    # Validar género
    if 'genero' in df.columns:
        details['columnas_validadas'].append('genero')
        valores_validos = DEMOGRAPHIC_CONFIG['genero']['valores_validos']
        mapeo = DEMOGRAPHIC_CONFIG['genero']['mapeo']
        
        # Normalizar valores
        genero_normalizado = df['genero'].astype(str).str.strip()
        for old_val, new_val in mapeo.items():
            genero_normalizado = genero_normalizado.replace(old_val, new_val)
        
        # Verificar valores válidos
        invalid_genero = genero_normalizado[~genero_normalizado.isin(valores_validos)]
        if len(invalid_genero) > 0:
            errors.append(f"Valores de género inválidos: {invalid_genero.unique().tolist()}")
            details['filas_con_errores'] += len(invalid_genero)
    
    # Validar nivel de educación
    if 'nivel_educacion' in df.columns:
        details['columnas_validadas'].append('nivel_educacion')
        categorias_validas = DEMOGRAPHIC_CONFIG['nivel_educacion']['categorias']
        mapeo = DEMOGRAPHIC_CONFIG['nivel_educacion']['mapeo']
        
        # Normalizar valores
        educacion_normalizada = df['nivel_educacion'].astype(str).str.strip()
        for old_val, new_val in mapeo.items():
            educacion_normalizada = educacion_normalizada.replace(old_val, new_val)
        
        # Verificar categorías válidas
        invalid_educacion = educacion_normalizada[~educacion_normalizada.isin(categorias_validas)]
        if len(invalid_educacion) > 0:
            errors.append(f"Categorías de educación inválidas: {invalid_educacion.unique().tolist()}")
            details['filas_con_errores'] += len(invalid_educacion)
    
    # Validar ingresos
    if 'ingresos' in df.columns:
        details['columnas_validadas'].append('ingresos')
        ingresos_series = pd.to_numeric(df['ingresos'], errors='coerce')
        
        # Valor mínimo
        invalid_ingresos = ingresos_series[ingresos_series < DEMOGRAPHIC_CONFIG['ingresos']['min']]
        if len(invalid_ingresos) > 0:
            errors.append(f"Ingresos negativos detectados: {len(invalid_ingresos)} filas")
            details['filas_con_errores'] += len(invalid_ingresos)
        
        # Detectar outliers (5x mediana)
        ingresos_clean = ingresos_series.dropna()
        if len(ingresos_clean) > 0:
            mediana = ingresos_clean.median()
            threshold = mediana * DEMOGRAPHIC_CONFIG['ingresos']['outlier_multiplier']
            outliers = ingresos_clean[ingresos_clean > threshold]
            if len(outliers) > 0:
                warnings.append(f"Posibles outliers en ingresos (> {DEMOGRAPHIC_CONFIG['ingresos']['outlier_multiplier']}x mediana): {len(outliers)} valores")
                details['outliers_detectados'] += len(outliers)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Logging
    log_action(
        function="validate_demographics",
        step="business_rules",
        parameters={"metadata_keys": list(metadata.keys())},
        before_metrics={"n_rows": len(df)},
        after_metrics=details,
        status="warning" if warnings else "success",
        message=f"Validación demográfica: {len(errors)} errores, {len(warnings)} advertencias",
        execution_time=execution_time
    )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        rule_name="demographics",
        message=f"Validación demográfica completada: {len(errors)} errores, {len(warnings)} advertencias",
        details=details
    )


def validate_likert(df: pd.DataFrame, col: str, scale: str = '5_puntos') -> ValidationResult:
    """
    Valida escalas Likert según reglas de negocio.
    
    Args:
        df: DataFrame con datos
        col: Columna a validar
        scale: Tipo de escala ('5_puntos', '7_puntos')
        
    Returns:
        ValidationResult con resultados de validación
    """
    start_time = datetime.now()
    errors = []
    warnings = []
    details = {
        'columna': col,
        'escala': scale,
        'valores_validos': [],
        'valores_invalidos': [],
        'consistencia_interna': None
    }
    
    if col not in df.columns:
        return ValidationResult(
            is_valid=False,
            rule_name="likert",
            message=f"Columna {col} no encontrada",
            details=details
        )
    
    scale_config = LIKERT_SCALES.get(scale)
    if not scale_config:
        return ValidationResult(
            is_valid=False,
            rule_name="likert",
            message=f"Escala {scale} no soportada",
            details=details
        )
    
    valores_esperados = scale_config['valores']
    etiquetas_esperadas = scale_config['etiquetas']
    details['valores_validos'] = valores_esperados + etiquetas_esperadas
    
    # Verificar valores
    col_series = df[col].astype(str).str.strip()
    valores_unicos = col_series.unique()
    
    # Intentar convertir a numérico
    col_numeric = pd.to_numeric(col_series, errors='coerce')
    
    # Verificar valores numéricos
    if not col_numeric.isna().all():
        valores_numericos_validos = col_numeric[col_numeric.isin(valores_esperados)]
        valores_numericos_invalidos = col_numeric[~col_numeric.isin(valores_esperados) & ~col_numeric.isna()]
        
        if len(valores_numericos_invalidos) > 0:
            errors.append(f"Valores numéricos inválidos en escala {scale}: {valores_numericos_invalidos.unique().tolist()}")
            details['valores_invalidos'].extend(valores_numericos_invalidos.unique().tolist())
    else:
        # Verificar etiquetas
        valores_etiquetas_validos = col_series[col_series.isin(etiquetas_esperadas)]
        valores_etiquetas_invalidos = col_series[~col_series.isin(etiquetas_esperadas)]
        
        if len(valores_etiquetas_invalidos) > 0:
            errors.append(f"Etiquetas inválidas en escala {scale}: {valores_etiquetas_invalidos.unique().tolist()}")
            details['valores_invalidos'].extend(valores_etiquetas_invalidos.unique().tolist())
    
    # Calcular consistencia interna si hay múltiples ítems relacionados
    likert_columns = [c for c in df.columns if any(keyword in c.lower() for keyword in ['likert', 'escala', 'pregunta', 'item'])]
    if len(likert_columns) > 1:
        try:
            # Convertir a numérico para cálculo
            likert_data = df[likert_columns].apply(pd.to_numeric, errors='coerce')
            likert_clean = likert_data.dropna()
            
            if len(likert_clean) > 0 and likert_clean.shape[1] > 1:
                # Calcular correlación promedio como proxy de consistencia
                correlations = []
                for i in range(len(likert_columns)):
                    for j in range(i+1, len(likert_columns)):
                        corr, _ = pearsonr(likert_clean.iloc[:, i], likert_clean.iloc[:, j])
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                if correlations:
                    consistencia = np.mean(correlations)
                    details['consistencia_interna'] = consistencia
                    
                    if consistencia < 0.3:
                        warnings.append(f"Baja consistencia interna en escala Likert: {consistencia:.3f}")
                    elif consistencia > 0.9:
                        warnings.append(f"Alta consistencia interna (posible redundancia): {consistencia:.3f}")
        except Exception as e:
            warnings.append(f"No se pudo calcular consistencia interna: {str(e)}")
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Logging
    log_action(
        function="validate_likert",
        step="business_rules",
        parameters={"columna": col, "escala": scale},
        before_metrics={"n_rows": len(df)},
        after_metrics=details,
        status="warning" if warnings else "success",
        message=f"Validación Likert {col}: {len(errors)} errores, {len(warnings)} advertencias",
        execution_time=execution_time
    )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        rule_name="likert",
        message=f"Validación Likert completada: {len(errors)} errores, {len(warnings)} advertencias",
        details=details
    )


def validate_geography(df: pd.DataFrame, region_col: str, comuna_col: str) -> ValidationResult:
    """
    Valida datos geográficos según reglas de negocio.
    
    Args:
        df: DataFrame con datos geográficos
        region_col: Columna con regiones
        comuna_col: Columna con comunas
        
    Returns:
        ValidationResult con resultados de validación
    """
    start_time = datetime.now()
    errors = []
    warnings = []
    details = {
        'columnas_validadas': [],
        'regiones_invalidas': [],
        'comunas_invalidas': [],
        'inconsistencias_region_comuna': 0
    }
    
    # Verificar que las columnas existen
    if region_col not in df.columns:
        return ValidationResult(
            is_valid=False,
            rule_name="geography",
            message=f"Columna de región {region_col} no encontrada",
            details=details
        )
    
    if comuna_col not in df.columns:
        return ValidationResult(
            is_valid=False,
            rule_name="geography",
            message=f"Columna de comuna {comuna_col} no encontrada",
            details=details
        )
    
    details['columnas_validadas'] = [region_col, comuna_col]
    
    # Validar regiones
    regiones_unicas = df[region_col].astype(str).str.strip().unique()
    regiones_invalidas = [r for r in regiones_unicas if r not in CHILE_REGIONS]
    if regiones_invalidas:
        errors.append(f"Regiones inválidas: {regiones_invalidas}")
        details['regiones_invalidas'] = regiones_invalidas
    
    # Validar comunas
    comunas_unicas = df[comuna_col].astype(str).str.strip().unique()
    comunas_invalidas = [c for c in comunas_unicas if c not in CHILE_COMMUNAS]
    if comunas_invalidas:
        errors.append(f"Comunas inválidas: {comunas_invalidas}")
        details['comunas_invalidas'] = comunas_invalidas
    
    # Validar consistencia región-comuna
    if not regiones_invalidas and not comunas_invalidas:
        inconsistencias = 0
        for _, row in df.iterrows():
            region = str(row[region_col]).strip()
            comuna = str(row[comuna_col]).strip()
            
            if region in CHILE_REGIONS and comuna in CHILE_COMMUNAS:
                region_esperada = CHILE_COMMUNAS[comuna]
                if region != region_esperada:
                    inconsistencias += 1
        
        if inconsistencias > 0:
            errors.append(f"Inconsistencias región-comuna: {inconsistencias} registros")
            details['inconsistencias_region_comuna'] = inconsistencias
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Logging
    log_action(
        function="validate_geography",
        step="business_rules",
        parameters={"region_col": region_col, "comuna_col": comuna_col},
        before_metrics={"n_rows": len(df)},
        after_metrics=details,
        status="warning" if warnings else "success",
        message=f"Validación geográfica: {len(errors)} errores, {len(warnings)} advertencias",
        execution_time=execution_time
    )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        rule_name="geography",
        message=f"Validación geográfica completada: {len(errors)} errores, {len(warnings)} advertencias",
        details=details
    )


def validate_cross_consistency(df: pd.DataFrame, metadata: dict[str, Any]) -> ValidationResult:
    """
    Valida consistencia cruzada entre variables.
    
    Args:
        df: DataFrame con datos
        metadata: Metadatos del dataset
        
    Returns:
        ValidationResult con resultados de validación
    """
    start_time = datetime.now()
    errors = []
    warnings = []
    details = {
        'reglas_ejecutadas': [],
        'inconsistencias_detectadas': 0,
        'alertas_generadas': 0
    }
    
    # Regla 1: Edad vs Educación
    if 'edad' in df.columns and 'nivel_educacion' in df.columns:
        details['reglas_ejecutadas'].append('edad_educacion')
        
        edad_series = pd.to_numeric(df['edad'], errors='coerce')
        educacion_series = df['nivel_educacion'].astype(str).str.strip()
        
        # Menores de 18 con educación universitaria
        menores_universitarios = df[(edad_series < 18) & 
                                   (educacion_series.isin(['Universitaria', 'Postgrado', 'Doctorado']))]
        if len(menores_universitarios) > 0:
            warnings.append(f"Menores de 18 con educación universitaria: {len(menores_universitarios)} casos")
            details['alertas_generadas'] += len(menores_universitarios)
        
        # Mayores de 25 sin educación
        mayores_sin_educacion = df[(edad_series > 25) & 
                                  (educacion_series == 'Sin educación')]
        if len(mayores_sin_educacion) > 0:
            warnings.append(f"Mayores de 25 sin educación: {len(mayores_sin_educacion)} casos")
            details['alertas_generadas'] += len(mayores_sin_educacion)
    
    # Regla 2: Edad vs Empleo
    if 'edad' in df.columns and 'empleo' in df.columns:
        details['reglas_ejecutadas'].append('edad_empleo')
        
        edad_series = pd.to_numeric(df['edad'], errors='coerce')
        empleo_series = df['empleo'].astype(str).str.strip()
        
        # Jubilados menores de 60
        jubilados_jovenes = df[(edad_series < 60) & 
                              (empleo_series.str.contains('Jubilado', case=False, na=False))]
        if len(jubilados_jovenes) > 0:
            warnings.append(f"Jubilados menores de 60: {len(jubilados_jovenes)} casos")
            details['alertas_generadas'] += len(jubilados_jovenes)
        
        # Menores de 16 con empleo formal
        menores_empleados = df[(edad_series < 16) & 
                              (empleo_series.str.contains('Empleado|Trabajador', case=False, na=False))]
        if len(menores_empleados) > 0:
            errors.append(f"Menores de 16 con empleo formal: {len(menores_empleados)} casos")
            details['inconsistencias_detectadas'] += len(menores_empleados)
    
    # Regla 3: Ingresos vs Ocupación
    if 'ingresos' in df.columns and 'ocupacion' in df.columns:
        details['reglas_ejecutadas'].append('ingresos_ocupacion')
        
        ingresos_series = pd.to_numeric(df['ingresos'], errors='coerce')
        ocupacion_series = df['ocupacion'].astype(str).str.strip()
        
        # Desempleados con ingresos altos
        ingresos_mediana = ingresos_series.median()
        desempleados_ricos = df[(ocupacion_series.str.contains('Desempleado|Cesante', case=False, na=False)) & 
                               (ingresos_series > ingresos_mediana * 3)]
        if len(desempleados_ricos) > 0:
            warnings.append(f"Desempleados con ingresos altos: {len(desempleados_ricos)} casos")
            details['alertas_generadas'] += len(desempleados_ricos)
    
    # Regla 4: Género vs Ocupaciones típicas
    if 'genero' in df.columns and 'ocupacion' in df.columns:
        details['reglas_ejecutadas'].append('genero_ocupacion')
        
        genero_series = df['genero'].astype(str).str.strip()
        ocupacion_series = df['ocupacion'].astype(str).str.strip()
        
        # Verificar patrones de género en ocupaciones (solo como advertencia)
        # Esta regla es más informativa que restrictiva
        pass
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Logging
    log_action(
        function="validate_cross_consistency",
        step="business_rules",
        parameters={"metadata_keys": list(metadata.keys())},
        before_metrics={"n_rows": len(df)},
        after_metrics=details,
        status="warning" if warnings else "success",
        message=f"Validación de consistencia cruzada: {len(errors)} errores, {len(warnings)} advertencias",
        execution_time=execution_time
    )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        rule_name="cross_consistency",
        message=f"Validación de consistencia cruzada completada: {len(errors)} errores, {len(warnings)} advertencias",
        details=details
    )


def validate_business_rules(df: pd.DataFrame, metadata: dict[str, Any]) -> List[ValidationResult]:
    """
    Aplica todas las reglas de negocio relevantes según los metadatos.
    
    Args:
        df: DataFrame a validar
        metadata: Metadatos del dataset
        
    Returns:
        Lista de ValidationResult con resultados de todas las validaciones
    """
    start_time = datetime.now()
    results = []
    
    # Determinar qué validaciones aplicar basado en metadatos
    dataset_type = metadata.get('dataset_type', 'general')
    columns = list(df.columns)
    
    # Validación demográfica (siempre que haya columnas demográficas)
    demographic_cols = ['edad', 'genero', 'nivel_educacion', 'ingresos']
    if any(col in columns for col in demographic_cols):
        result = validate_demographics(df, metadata)
        results.append(result)
    
    # Validación Likert (si hay columnas de encuesta)
    likert_keywords = ['likert', 'escala', 'pregunta', 'item', 'acuerdo']
    likert_cols = [col for col in columns if any(keyword in col.lower() for keyword in likert_keywords)]
    for col in likert_cols:
        result = validate_likert(df, col)
        results.append(result)
    
    # Validación geográfica (si hay columnas geográficas)
    geo_cols = ['region', 'comuna', 'provincia', 'ciudad']
    if any(col in columns for col in geo_cols):
        region_col = next((col for col in geo_cols if col in columns), None)
        comuna_col = next((col for col in ['comuna', 'municipio'] if col in columns), None)
        if region_col and comuna_col:
            result = validate_geography(df, region_col, comuna_col)
            results.append(result)
    
    # Validación de consistencia cruzada
    result = validate_cross_consistency(df, metadata)
    results.append(result)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Logging general
    total_errors = sum(1 for r in results if not r.is_valid)
    total_warnings = len([r for r in results if r.details.get('alertas_generadas', 0) > 0])
    
    log_action(
        function="validate_business_rules",
        step="business_rules",
        parameters={"dataset_type": dataset_type, "n_rules": len(results)},
        before_metrics={"n_rows": len(df), "n_columns": len(columns)},
        after_metrics={"total_errors": total_errors, "total_warnings": total_warnings},
        status="error" if total_errors > 0 else "warning" if total_warnings > 0 else "success",
        message=f"Validación de reglas de negocio: {len(results)} reglas, {total_errors} errores, {total_warnings} advertencias",
        execution_time=execution_time
    )
    
    return results


# Decorador para aplicar validaciones de reglas de negocio
def validate_business_rules_decorator(func):
    """Decorador que aplica validaciones de reglas de negocio antes de ejecutar la función"""
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        # Extraer metadatos de kwargs o crear uno básico
        metadata = kwargs.get('metadata', {})
        
        # Aplicar validaciones de reglas de negocio
        validation_results = validate_business_rules(df, metadata)
        
        # Verificar si hay errores críticos
        critical_errors = [r for r in validation_results if not r.is_valid]
        if critical_errors:
            # Reportar errores críticos
            error_messages = [f"{r.rule_name}: {r.message}" for r in critical_errors]
            combined_message = "; ".join(error_messages)
            
            # Convertir ValidationResult a diccionario serializable
            validation_details = []
            for r in validation_results:
                result_dict = {
                    "is_valid": r.is_valid,
                    "rule_name": r.rule_name,
                    "message": r.message,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat() if hasattr(r.timestamp, 'isoformat') else str(r.timestamp)
                }
                validation_details.append(result_dict)
            
            report_business_rule_error(
                message=f"Errores críticos en reglas de negocio: {combined_message}",
                context=f"{func.__module__}.{func.__name__}",
                details={
                    "validation_results": validation_details,
                    "function_name": func.__name__,
                    "module": func.__module__
                }
            )
        
        # Continuar con la función original
        return func(df, *args, **kwargs)
    
    return wrapper 