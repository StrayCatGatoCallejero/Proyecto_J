# Sistema Avanzado de Manejo de Errores de Negocio

## 📋 Descripción General

El sistema de manejo de errores de negocio proporciona una solución completa para detectar, visualizar y gestionar errores en datos de ciencias sociales, encuestas y estudios demográficos. Integra perfectamente con el pipeline existente y ofrece visualizaciones avanzadas en Streamlit.

## 🚀 Características Principales

### ✅ Extracción Inteligente de Errores
- Extrae errores desde logs del sistema
- Procesa resultados de validación de reglas de negocio
- Identifica errores específicos por fila y columna
- Clasifica errores por severidad (error, warning, info)

### 📊 Visualización Avanzada
- Métricas en tiempo real de errores
- Gráficos interactivos (barras, pastel, timeline)
- Tablas filtrables y ordenables
- Dashboard completo de análisis de errores

### 🔧 Acciones Automatizadas
- Copiar reportes al portapapeles
- Exportar errores a CSV
- Generar reportes detallados
- Recomendaciones automáticas

### 🎯 Integración Completa
- Compatible con el pipeline existente
- Funciona con SessionData y PipelineOrchestrator
- Integración nativa con Streamlit
- Logging detallado de todas las acciones

## 📁 Estructura del Sistema

```
processing/
├── business_error_handler.py    # Sistema principal de manejo de errores
├── business_rules.py           # Reglas de negocio y validaciones
├── error_reporter.py           # Reporte de errores híbrido
└── logging.py                  # Sistema de logging unificado

orchestrator/
└── pipeline_orchestrator.py    # Pipeline principal (ya integrado)

ejemplos/
├── ejemplo_manejo_errores_avanzado.py    # Ejemplos completos
└── ejemplo_integracion_app_front.py      # Integración con app_front.py
```

## 🔧 Instalación y Configuración

### 1. Dependencias Requeridas

El sistema utiliza las dependencias ya instaladas en el proyecto:

```python
# Ya incluidas en requirements.txt
pandas
numpy
streamlit
plotly
scipy
```

### 2. Importación Básica

```python
from processing.business_error_handler import (
    BusinessErrorHandler,
    get_business_errors,
    display_business_errors_in_streamlit
)
from processing.business_rules import validate_business_rules
```

## 📖 Guía de Uso

### 1. Uso Básico - Extracción de Errores

```python
# Extraer errores desde logs
logs = session_data.logs  # o desde tu fuente de logs
errors = get_business_errors(logs)

# Mostrar errores
for error in errors:
    print(f"[{error['timestamp']}] {error['regla']} fila {error['fila']}: {error['detalle']}")
```

### 2. Uso Avanzado - BusinessErrorHandler

```python
# Crear handler
handler = BusinessErrorHandler()

# Extraer errores desde múltiples fuentes
handler.extract_business_errors_from_logs(logs)
handler.extract_errors_from_validation_results(validation_results)

# Obtener resumen
summary = handler.get_error_summary()
print(f"Total errores: {summary['total_errors']}")

# Mostrar en Streamlit
handler.display_errors_in_streamlit()
```

### 3. Integración con Pipeline

```python
# En tu pipeline existente
try:
    # Ejecutar validaciones
    validation_results = validate_business_rules(df, metadata)
    
    # Verificar si hay errores
    failed_validations = [r for r in validation_results if not r.is_valid]
    
    if failed_validations:
        # Mostrar errores usando el sistema avanzado
        display_business_errors_in_streamlit(
            logs=session_data.logs,
            validation_results=validation_results
        )
        st.stop()  # Detener pipeline si hay errores críticos
    
except BusinessRuleError as e:
    st.error(f"Error de regla de negocio: {e.message}")
```

### 4. Integración en app_front.py

```python
# Después de cargar datos
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Validar reglas de negocio
    metadata = {'dataset_type': 'social_sciences'}
    validation_results = validate_business_rules(df, metadata)
    
    # Si hay errores, mostrarlos
    total_errors = sum(1 for r in validation_results if not r.is_valid)
    if total_errors > 0:
        st.warning("⚠️ Se detectaron inconsistencias en los datos")
        
        # Usar el sistema avanzado
        display_business_errors_in_streamlit(
            logs=[],
            validation_results=validation_results
        )
        
        # Preguntar al usuario qué hacer
        if st.button("⚠️ Continuar con Advertencias"):
            st.success("Continuando con el análisis...")
        else:
            st.stop()
```

## 📊 Tipos de Errores Soportados

### 1. Errores Demográficos
- Edad fuera de rango válido
- Género no reconocido
- Nivel de educación inválido
- Ingresos negativos o outliers

### 2. Errores Geográficos
- Regiones no válidas
- Comunas no reconocidas
- Inconsistencias región-comuna

### 3. Errores en Escalas Likert
- Valores fuera de escala
- Mezcla de tipos de datos
- Baja consistencia interna

### 4. Errores de Consistencia Cruzada
- Edad vs educación inconsistente
- Edad vs empleo problemático
- Ingresos vs ocupación sospechosa

### 5. Errores de Calidad de Datos
- Valores faltantes excesivos
- Duplicados
- Outliers estadísticos

## 🎨 Personalización

### 1. Configurar Reglas de Negocio

```python
# En business_rules.py, modificar las configuraciones
DEMOGRAPHIC_CONFIG = {
    'edad': {
        'min': 0,
        'max': 120,
        'outlier_threshold': 3.0
    },
    'genero': {
        'valores_validos': ['Masculino', 'Femenino', 'Otro', 'No binario'],
        # Agregar más valores según tu dominio
    }
}
```

### 2. Agregar Nuevas Reglas

```python
def validate_custom_rule(df: pd.DataFrame, metadata: dict) -> ValidationResult:
    """Nueva regla de validación personalizada"""
    errors = []
    warnings = []
    details = {}
    
    # Tu lógica de validación aquí
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        rule_name="custom_rule",
        message=f"Validación personalizada: {len(errors)} errores",
        details=details
    )
```

### 3. Personalizar Visualizaciones

```python
# Modificar el método display_errors_in_streamlit
handler.display_errors_in_streamlit(
    show_details=True,    # Mostrar tabla detallada
    show_charts=True      # Mostrar gráficos
)
```

## 📈 Métricas y Reportes

### 1. Métricas Disponibles

```python
summary = handler.get_error_summary()

# Métricas básicas
total_errors = summary['total_errors']
errors_by_rule = summary['errors_by_rule']
errors_by_severity = summary['errors_by_severity']
errors_by_column = summary['errors_by_column']

# Timeline de errores
timeline = summary['timeline']
```

### 2. Generar Reportes

```python
# Reporte automático
handler._generate_detailed_report()

# Exportar a CSV
handler._export_errors_to_csv()

# Copiar al portapapeles
handler._copy_error_report_to_clipboard()
```

### 3. Recomendaciones Automáticas

```python
recommendations = handler.get_recommendations()

# Ejemplos de recomendaciones generadas:
# - "Revisar y corregir errores críticos antes de continuar"
# - "La columna 'edad' tiene muchos errores. Considerar limpieza"
# - "Verificar datos geográficos: regiones y comunas deben ser válidas"
```

## 🔍 Debugging y Troubleshooting

### 1. Verificar Logs

```python
# Obtener logs completos
logger = get_logger()
logs = logger.get_session_history()

# Filtrar por paso
business_logs = logger.get_step_history("business_rules")
```

### 2. Validar Integridad

```python
# Verificar integridad de logs
integrity_check = logger.validate_integrity()
if not integrity_check['valid']:
    print(f"Problemas detectados: {integrity_check['issues']}")
```

### 3. Modo Debug

```python
# Activar logging detallado
import logging
logging.getLogger("pipeline").setLevel(logging.DEBUG)
```

## 🚀 Ejemplos Prácticos

### Ejemplo 1: Validación en Carga de Datos

```python
def cargar_y_validar_datos(uploaded_file):
    """Carga datos y valida reglas de negocio"""
    
    # Cargar datos
    df = pd.read_csv(uploaded_file)
    
    # Validar
    metadata = {'dataset_type': 'social_sciences'}
    validation_results = validate_business_rules(df, metadata)
    
    # Manejar errores
    if any(not r.is_valid for r in validation_results):
        st.error("⚠️ Datos con inconsistencias detectadas")
        
        # Mostrar errores detallados
        display_business_errors_in_streamlit(
            logs=[],
            validation_results=validation_results
        )
        
        # Opciones para el usuario
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Corregir Automáticamente"):
                # Lógica de corrección
                pass
        with col2:
            if st.button("⚠️ Continuar con Errores"):
                return df, validation_results
    
    return df, validation_results
```

### Ejemplo 2: Monitoreo Continuo

```python
def analizar_con_monitoreo(df, tipo_analisis):
    """Ejecuta análisis con monitoreo de errores"""
    
    # Crear handler
    handler = BusinessErrorHandler()
    
    # Ejecutar análisis
    if tipo_analisis == "correlacion":
        resultado = ejecutar_correlacion(df, handler)
    elif tipo_analisis == "regresion":
        resultado = ejecutar_regresion(df, handler)
    
    # Verificar errores generados durante el análisis
    if handler.errors:
        st.warning("⚠️ Nuevos errores detectados durante el análisis:")
        handler.display_errors_in_streamlit()
    
    return resultado
```

### Ejemplo 3: Exportación con Validación

```python
def exportar_con_validacion(df, validation_results):
    """Exporta datos con validación de calidad"""
    
    handler = BusinessErrorHandler()
    handler.extract_errors_from_validation_results(validation_results)
    
    # Verificar calidad antes de exportar
    problemas = []
    if df.isnull().sum().sum() > 0:
        problemas.append("Valores faltantes detectados")
    
    if df.duplicated().sum() > 0:
        problemas.append("Duplicados detectados")
    
    # Crear errores de exportación
    for problema in problemas:
        error = BusinessError(
            rule_name="exportacion",
            error_type="data_quality",
            message=problema,
            severity="warning"
        )
        handler.errors.append(error)
    
    # Generar reporte
    if handler.errors:
        reporte = generar_reporte_con_errores(df, handler.errors)
        st.download_button(
            "📥 Descargar Reporte",
            reporte,
            "reporte_errores.txt"
        )
    
    return handler.errors
```

## 📚 Referencias y Enlaces

### Archivos Principales
- `processing/business_error_handler.py` - Sistema principal
- `processing/business_rules.py` - Reglas de validación
- `ejemplo_manejo_errores_avanzado.py` - Ejemplos completos
- `ejemplo_integracion_app_front.py` - Integración práctica

### Funciones Clave
- `BusinessErrorHandler` - Clase principal
- `get_business_errors()` - Función de utilidad
- `display_business_errors_in_streamlit()` - Visualización
- `validate_business_rules()` - Validación

### Configuración
- `config/config.yml` - Configuración del sistema
- `logs/pipeline.log` - Logs del sistema

## 🤝 Contribución

Para contribuir al sistema de manejo de errores:

1. **Agregar nuevas reglas**: Modificar `business_rules.py`
2. **Mejorar visualizaciones**: Actualizar `business_error_handler.py`
3. **Nuevas funcionalidades**: Crear nuevos métodos en las clases
4. **Documentación**: Actualizar este README

## 📞 Soporte

Para soporte técnico o preguntas:

1. Revisar los ejemplos en `ejemplo_manejo_errores_avanzado.py`
2. Consultar la documentación de las funciones
3. Verificar los logs en `logs/pipeline.log`
4. Crear un issue en el repositorio

---

**¡El sistema de manejo de errores está listo para usar! 🚀**

Integra fácilmente con tu aplicación existente y proporciona una experiencia de usuario excepcional para la gestión de errores de negocio. 