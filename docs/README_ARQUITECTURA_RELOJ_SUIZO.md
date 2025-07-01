# Arquitectura "Reloj Suizo" - Refactorización del Sistema de Estadísticas

## 🎯 Objetivo

Esta refactorización implementa el patrón de arquitectura "Reloj Suizo" para mejorar la calidad, cohesión y mantenibilidad del código estadístico. Cada módulo tiene una responsabilidad única y bien definida, con interfaces claras y logging sistemático.

## 🏗️ Arquitectura Modular

### 📁 Estructura de Módulos

```
processing/
├── io.py           # Carga y validación de archivos
├── stats.py        # Análisis estadístico
├── visualization.py # Visualizaciones (matplotlib + plotly)
├── filters.py      # Filtrado y limpieza de datos
├── features.py     # Creación y transformación de features
├── logging.py      # Sistema de logging unificado
└── types.py        # Tipos y validaciones
```

### 🔧 Principios de Diseño

1. **Separación de Responsabilidades**: Cada módulo tiene una función específica
2. **Interfaces Consistentes**: Todas las funciones siguen el mismo patrón
3. **Logging Sistemático**: Cada operación se registra automáticamente
4. **Validación de Entrada**: Verificación robusta de parámetros
5. **Manejo de Errores**: Excepciones controladas y informativas

## 📊 Módulos Principales

### 1. I/O (`io.py`)

**Responsabilidades:**
- Carga de archivos (.sav, .dta, .csv, .xlsx)
- Validación de integridad de datos
- Metadatos estructurados

**Funciones Principales:**
```python
# Cargar archivo con metadatos
df, metadata = cargar_archivo("datos.sav")

# Validar DataFrame
validation = validar_dataframe(df, metadata)

# Obtener información del archivo
info = obtener_info_archivo("datos.csv")
```

### 2. Estadísticas (`stats.py`)

**Responsabilidades:**
- Estadísticas descriptivas avanzadas
- Pruebas estadísticas (normalidad, t-test, chi²)
- Análisis de correlaciones y contingencia
- Regresión lineal y logística

**Funciones Principales:**
```python
# Estadísticas descriptivas
stats = summary_statistics(df, ['edad', 'ingresos'])

# Correlaciones
corr = compute_correlations(df, ['edad', 'ingresos'], method='pearson')

# Análisis de contingencia
table, stats = contingency_analysis(df, 'educacion', 'genero')

# Pruebas de normalidad
normality = normality_test(df, 'edad')

# T-test independiente
t_result = t_test_independent(df, 'ingresos', 'genero')

# Regresión lineal
regression = linear_regression(df, 'satisfaccion', ['edad', 'ingresos'])
```

### 3. Visualización (`visualization.py`)

**Responsabilidades:**
- Visualizaciones estáticas con matplotlib
- Visualizaciones interactivas con plotly
- Generación de gráficos según tipo de datos

**Funciones Principales:**
```python
# Visualizaciones estáticas (matplotlib)
fig = plot_histogram(df, 'edad', bins=20)
fig = plot_boxplot(df, 'ingresos', group_column='educacion')
fig = plot_scatter(df, 'edad', 'ingresos', color_column='genero')
fig = plot_heatmap(corr_matrix)
fig = plot_bar_chart(df, 'educacion', top_n=5)

# Visualizaciones interactivas (plotly)
fig = plotly_histogram(df, 'edad')
fig = plotly_heatmap(corr_matrix)
fig = plotly_scatter(df, 'edad', 'ingresos')
```

### 4. Filtros (`filters.py`)

**Responsabilidades:**
- Filtrado por condiciones simples y complejas
- Limpieza de datos (outliers, valores faltantes)
- Muestreo y selección de datos

**Funciones Principales:**
```python
# Filtrado por condición
df_filtered = filter_by_condition(df, "edad > 25 and genero == 'M'")

# Filtrado por rango
df_filtered = filter_by_range(df, 'ingresos', min_val=1000, max_val=5000)

# Filtrado por valores
df_filtered = filter_by_values(df, 'region', ['Norte', 'Centro'])

# Eliminar outliers
df_clean = remove_outliers(df, 'ingresos', method='iqr')

# Manejar valores faltantes
df_clean = handle_missing_values(df, method='fill_median')

# Muestreo
df_sample = sample_data(df, fraction=0.5, random_state=42)
```

### 5. Features (`features.py`)

**Responsabilidades:**
- Creación de features derivadas
- Codificación de variables categóricas
- Escalado y normalización
- Selección de features

**Funciones Principales:**
```python
# Features numéricas derivadas
df_features = create_numeric_features(df, ['edad', 'ingresos'])

# Codificación categórica
df_features = encode_categorical(df, ['educacion'], method='label')

# Escalado
df_scaled = scale_features(df, ['edad', 'ingresos'], method='standard')

# Features de interacción
df_features = create_interaction_features(df, ['edad', 'ingresos'])

# Selección de features
selected = select_features(df, 'target', method='correlation', n_features=10)

# Features temporales
df_features = create_time_features(df, 'fecha')

# Binning
df_features = create_binning_features(df, ['edad'], n_bins=5)
```

## 🔄 Patrón de Uso

### Flujo Típico de Análisis

```python
# 1. Cargar datos
df, metadata = cargar_archivo("datos.csv")
validation = validar_dataframe(df, metadata)

# 2. Limpiar datos
df_clean = handle_missing_values(df, method='fill_median')
df_clean = remove_outliers(df_clean, 'ingresos', method='iqr')
df_clean = drop_duplicates(df_clean)

# 3. Análisis estadístico
stats = summary_statistics(df_clean, ['edad', 'ingresos'])
corr = compute_correlations(df_clean, ['edad', 'ingresos'])
normality = normality_test(df_clean, 'edad')

# 4. Crear features
df_features = create_numeric_features(df_clean, ['edad', 'ingresos'])
df_features = encode_categorical(df_features, ['educacion'], method='label')

# 5. Visualizar
fig = plot_histogram(df_clean, 'edad')
fig = plot_scatter(df_clean, 'edad', 'ingresos')
fig = plot_heatmap(corr)

# 6. Análisis avanzado
t_result = t_test_independent(df_clean, 'ingresos', 'genero')
regression = linear_regression(df_clean, 'satisfaccion', ['edad', 'ingresos'])
```

## 📈 Logging y Trazabilidad

### Sistema de Logging Unificado

Cada función registra automáticamente:
- Parámetros de entrada
- Métricas antes y después
- Tiempo de ejecución
- Estado de éxito/error
- Mensajes descriptivos

```python
# Ejemplo de log automático
log_action(
    function="summary_statistics",
    step="stats",
    parameters={"columns": ["edad", "ingresos"]},
    before_metrics={"n_rows": 1000},
    after_metrics={"n_variables": 2},
    status="success",
    message="Estadísticas descriptivas calculadas",
    execution_time=0.15
)
```

### Configuración de Logging

```python
# Configurar logging
from processing.logging import setup_logging
logger = setup_logging("config/config.yml")

# Obtener historial
history = logger.get_session_history()
summary = logger.get_summary_stats()
```

## 🧪 Testing y Validación

### Validaciones Automáticas

- **Entrada**: Verificación de tipos, rangos y existencia de columnas
- **Procesamiento**: Control de errores y excepciones
- **Salida**: Validación de formatos y consistencia

### Ejemplos de Validación

```python
# Validación de entrada
if column not in df.columns:
    raise ValueError(f"La columna {column} no existe en el DataFrame")

if method not in valid_methods:
    raise ValueError(f"Método debe ser uno de: {valid_methods}")

# Validación de datos
if len(series) == 0:
    raise ValueError("No hay datos válidos para procesar")
```

## 🚀 Ventajas de la Nueva Arquitectura

### 1. **Modularidad**
- Cada módulo tiene responsabilidades claras
- Fácil mantenimiento y actualización
- Reutilización de componentes

### 2. **Consistencia**
- Interfaces uniformes en todos los módulos
- Patrones de uso similares
- Documentación estandarizada

### 3. **Trazabilidad**
- Logging automático de todas las operaciones
- Auditoría completa del pipeline
- Debugging facilitado

### 4. **Robustez**
- Validación exhaustiva de entrada
- Manejo controlado de errores
- Recuperación graceful de fallos

### 5. **Escalabilidad**
- Fácil adición de nuevas funcionalidades
- Integración con otros sistemas
- Paralelización de operaciones

## 📋 Migración desde Código Anterior

### Cambios Principales

1. **Funciones Renombradas**: Nombres más descriptivos y consistentes
2. **Parámetros Estandarizados**: Interfaces uniformes
3. **Logging Automático**: No requiere configuración manual
4. **Validación Robusta**: Verificación automática de entrada
5. **Separación de Responsabilidades**: Funciones más específicas

### Ejemplo de Migración

**Antes:**
```python
# Código anterior (mezclado)
def analyze_data(df):
    # I/O, estadísticas, visualización mezcladas
    pass
```

**Después:**
```python
# Código refactorizado (modular)
# 1. I/O
df, metadata = cargar_archivo("datos.csv")

# 2. Estadísticas
stats = summary_statistics(df, ['edad', 'ingresos'])

# 3. Visualización
fig = plot_histogram(df, 'edad')
```

## 🎯 Próximos Pasos

1. **Documentación**: Completar docstrings y ejemplos
2. **Testing**: Aumentar cobertura de pruebas
3. **Performance**: Optimización de operaciones críticas
4. **Integración**: Conectar con interfaces de usuario
5. **Extensión**: Añadir nuevas funcionalidades estadísticas

## 📚 Recursos Adicionales

- `ejemplo_arquitectura_reloj_suizo.py`: Ejemplo completo de uso
- `processing/logging.py`: Documentación del sistema de logging
- `tests/`: Suite de pruebas unitarias
- `config/config.yml`: Configuración del sistema

---

**Nota**: Esta arquitectura sigue los principios SOLID y las mejores prácticas de ingeniería de software para crear un sistema robusto, mantenible y escalable. 