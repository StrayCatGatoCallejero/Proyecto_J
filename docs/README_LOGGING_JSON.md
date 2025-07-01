# Sistema de Logging JSON Avanzado

## Descripción

El **Sistema de Logging JSON Avanzado** es una capa de registro estructurado que extiende el sistema de logging unificado existente, proporcionando capacidades avanzadas para monitoreo, trazabilidad y análisis de pipelines de datos.

## Características Principales

### 🔍 **Trazabilidad Completa**
- **Session ID único** para cada ejecución del pipeline
- **Correlación de eventos** a través de request_id y correlation_id
- **Historial completo** de cada paso del proceso

### 📊 **Logging Estructurado**
- **Formato JSON** estándar para fácil ingestión
- **Campos estandarizados** (timestamp, level, module, function, etc.)
- **Métricas detalladas** antes y después de cada operación
- **Contexto completo** con parámetros y metadatos

### 🔧 **Integración con Sistemas de Monitoreo**
- **Compatibilidad con ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Integración con Datadog** para métricas y alertas
- **Soporte para Prometheus** para métricas de rendimiento
- **Configuración flexible** para diferentes entornos

### 📈 **Métricas y Análisis**
- **Tiempo de ejecución** de cada paso
- **Estadísticas de rendimiento** del sistema
- **Análisis de errores** y debugging
- **Reportes automáticos** de sesiones

## Arquitectura

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Pipeline      │───▶│   JSONLogger     │───▶│   Log Files     │
│   Components    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Monitoring      │
                       │  Systems         │
                       │  (ELK, Datadog)  │
                       └──────────────────┘
```

## Instalación y Configuración

### 1. Dependencias

El sistema utiliza las dependencias existentes del proyecto:

```bash
pip install -r requirements.txt
```

### 2. Configuración

Editar `config/config.yml` para habilitar el logging JSON:

```yaml
logging:
  json_logging:
    enabled: true
    json_file: "logs/pipeline.json"
    console_json: false
    include_system_info: true
    include_stack_traces: true
    
    monitoring:
      elk:
        enabled: false
        host: "localhost"
        port: 9200
        index_prefix: "pipeline-logs"
      
      datadog:
        enabled: false
        api_key: ""
        host: "localhost"
        port: 8126
```

### 3. Uso Básico

```python
from processing.json_logging import create_json_logger, LogLevel, LogCategory

# Crear logger JSON
logger = create_json_logger()

# Registrar evento
logger.log_event(
    level=LogLevel.INFO,
    message="Datos cargados exitosamente",
    module="data_loader",
    function="load_csv",
    step="data_loading",
    category=LogCategory.DATA_LOAD,
    parameters={"file_path": "data.csv"},
    before_metrics={"file_size": 1024},
    after_metrics={"rows": 1000, "columns": 10},
    execution_time=1.5
)
```

## Componentes Principales

### JSONLogger

Clase principal para logging JSON estructurado:

```python
class JSONLogger:
    def __init__(self, config: Dict[str, Any], session_id: Optional[str] = None)
    
    def log_event(self, level, message, module, function, step, category, ...)
    def log_data_load(self, function, file_path, file_size, rows, columns, ...)
    def log_validation(self, function, validation_type, total_checks, ...)
    def log_business_rules(self, function, rules_executed, rules_failed, ...)
    def log_analysis(self, function, analysis_type, input_size, output_size, ...)
    def log_error(self, function, error, context, execution_time, ...)
```

### JSONLoggingManager

Gestor para múltiples sesiones simultáneas:

```python
class JSONLoggingManager:
    def __init__(self, config: Dict[str, Any])
    
    def create_session(self, session_id: Optional[str] = None) -> JSONLogger
    def get_session(self, session_id: str) -> Optional[JSONLogger]
    def close_session(self, session_id: str) -> bool
    def get_all_sessions_summary(self) -> Dict[str, Any]
    def export_all_sessions(self, format: str = "json") -> str
```

### Enums y Estructuras

```python
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    DATA_LOAD = "data_load"
    VALIDATION = "validation"
    PROCESSING = "processing"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    EXPORT = "export"
    SYSTEM = "system"
    BUSINESS_RULES = "business_rules"
    ERROR_HANDLING = "error_handling"
```

## Formato JSON Generado

```json
{
  "timestamp": "2024-01-15T10:30:45.123456+00:00",
  "level": "INFO",
  "message": "Datos cargados exitosamente",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "module": "data_loader",
  "function": "load_csv",
  "step": "data_loading",
  "category": "data_load",
  "execution_time": 1.5,
  "before_metrics": {
    "file_size": 1024
  },
  "after_metrics": {
    "rows": 1000,
    "columns": 10
  },
  "parameters": {
    "file_path": "data.csv"
  },
  "error_details": null,
  "stack_trace": null,
  "error_type": null,
  "user_id": null,
  "request_id": null,
  "correlation_id": null,
  "system_info": {
    "platform": "Windows-10-10.0.19041-SP0",
    "python_version": "3.11.7",
    "cpu_count": 8,
    "memory_total": 17179869184,
    "process_id": 1234
  },
  "tags": ["data_load", "file_operation"],
  "metadata": {
    "version": "1.0",
    "environment": "development"
  }
}
```

## Integración con el Pipeline Existente

### 1. Reemplazar Logger Actual

```python
# Antes
from processing.logging import get_logger
logger = get_logger()

# Después
from processing.json_logging import create_json_logger
logger = create_json_logger()
```

### 2. Pipeline con Logging JSON

```python
class PipelineWithJSONLogging:
    def __init__(self, config_path: str = "config/config.yml"):
        self.json_logger = create_json_logger(config_path)
        self.session_id = self.json_logger.session_id
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        start_time = time.time()
        
        try:
            df = pd.read_csv(file_path)
            execution_time = time.time() - start_time
            
            self.json_logger.log_data_load(
                function="load_data",
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                rows=len(df),
                columns=len(df.columns),
                execution_time=execution_time,
                success=True
            )
            
            return df
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.json_logger.log_error(
                function="load_data",
                error=e,
                context="data_loading",
                execution_time=execution_time
            )
            raise
```

## Configuración para Sistemas de Monitoreo

### ELK Stack (Elasticsearch, Logstash, Kibana)

1. **Configurar Filebeat**:

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /path/to/project/logs/pipeline.json
  json.keys_under_root: true
  json.add_error_key: true
  json.message_key: message

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "pipeline-logs-%{+yyyy.MM.dd}"
```

2. **Configurar Logstash** (opcional):

```ruby
input {
  beats {
    port => 5044
  }
}

filter {
  if [category] == "data_load" {
    mutate {
      add_tag => ["data_operations"]
    }
  }
  
  if [level] == "ERROR" {
    mutate {
      add_tag => ["errors"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "pipeline-logs-%{+yyyy.MM.dd}"
  }
}
```

### Datadog

1. **Configurar Datadog Agent**:

```yaml
logs:
  - type: file
    path: /path/to/project/logs/pipeline.json
    service: pipeline
    source: python
    sourcecategory: sourcecode
```

2. **Configurar métricas personalizadas**:

```python
# En el código del pipeline
from datadog import initialize, statsd

initialize(api_key='your_api_key', app_key='your_app_key')

# Enviar métricas
statsd.gauge('pipeline.execution_time', execution_time, tags=['step:data_load'])
statsd.increment('pipeline.errors', tags=['function:load_data'])
```

### Prometheus

1. **Configurar métricas**:

```python
from prometheus_client import Counter, Histogram, start_http_server

# Métricas
pipeline_executions = Counter('pipeline_executions_total', 'Total pipeline executions', ['status'])
pipeline_duration = Histogram('pipeline_duration_seconds', 'Pipeline execution time', ['step'])

# En el código
pipeline_executions.labels(status='success').inc()
pipeline_duration.labels(step='data_load').observe(execution_time)
```

## Ejemplos de Uso

### Ejemplo 1: Logging Básico

```python
from processing.json_logging import create_json_logger, LogLevel, LogCategory

logger = create_json_logger()

# Log de carga de datos
logger.log_data_load(
    function="load_csv_file",
    file_path="datos.csv",
    file_size=1024 * 50,
    rows=1000,
    columns=10,
    execution_time=1.2,
    success=True
)

# Log de validación
logger.log_validation(
    function="validate_schema",
    validation_type="schema_validation",
    total_checks=15,
    passed_checks=14,
    failed_checks=1,
    execution_time=0.5,
    details={"failed_column": "edad"}
)
```

### Ejemplo 2: Múltiples Sesiones

```python
from processing.json_logging import get_json_logger_manager

manager = get_json_logger_manager()

# Crear múltiples sesiones
session1 = manager.create_session("analisis_1")
session2 = manager.create_session("analisis_2")

# Usar sesiones independientes
session1.log_system_event(LogLevel.INFO, "Análisis 1 iniciado")
session2.log_system_event(LogLevel.INFO, "Análisis 2 iniciado")

# Obtener resumen de todas las sesiones
summary = manager.get_all_sessions_summary()
```

### Ejemplo 3: Manejo de Errores

```python
try:
    # Código que puede fallar
    df = pd.read_csv("archivo_inexistente.csv")
except Exception as e:
    logger.log_error(
        function="load_data",
        error=e,
        context="data_loading",
        execution_time=0.1,
        additional_data={"file_path": "archivo_inexistente.csv"}
    )
```

## Análisis y Reportes

### 1. Resumen de Sesión

```python
summary = logger.get_session_summary()
print(f"Total logs: {summary['total_logs']}")
print(f"Errores: {summary['error_count']}")
print(f"Tiempo total: {summary['total_execution_time']}s")
```

### 2. Logs por Categoría

```python
validation_logs = logger.get_logs_by_category(LogCategory.VALIDATION)
error_logs = logger.get_error_logs()
analysis_logs = logger.get_logs_by_category(LogCategory.ANALYSIS)
```

### 3. Exportación de Logs

```python
# Exportar como JSON
json_path = logger.export_session_logs("json")

# Exportar como CSV
csv_path = logger.export_session_logs("csv")

# Exportar todas las sesiones
all_path = manager.export_all_sessions("json")
```

## Dashboards y Visualizaciones

### Kibana Dashboard

Crear dashboards en Kibana con:

- **Métricas de rendimiento**: Tiempo de ejecución por paso
- **Distribución de errores**: Errores por función y categoría
- **Análisis de sesiones**: Logs por session_id
- **Tendencias temporales**: Logs por timestamp

### Grafana Dashboard

Configurar dashboards en Grafana con:

- **Métricas de Prometheus**: Pipeline execution time, error rates
- **Logs de Loki**: Análisis de logs estructurados
- **Alertas**: Notificaciones automáticas

## Mejores Prácticas

### 1. **Niveles de Log Apropiados**

```python
# DEBUG: Información detallada para debugging
logger.log_event(level=LogLevel.DEBUG, message="Valor de variable X: 42")

# INFO: Información general del flujo
logger.log_event(level=LogLevel.INFO, message="Paso completado exitosamente")

# WARNING: Situaciones que requieren atención
logger.log_event(level=LogLevel.WARNING, message="Datos faltantes detectados")

# ERROR: Errores que afectan la funcionalidad
logger.log_event(level=LogLevel.ERROR, message="Error crítico en procesamiento")
```

### 2. **Categorización Correcta**

```python
# Usar categorías específicas
logger.log_event(category=LogCategory.DATA_LOAD, ...)
logger.log_event(category=LogCategory.VALIDATION, ...)
logger.log_event(category=LogCategory.ANALYSIS, ...)
logger.log_event(category=LogCategory.BUSINESS_RULES, ...)
```

### 3. **Métricas Útiles**

```python
# Incluir métricas relevantes
logger.log_event(
    before_metrics={"input_size": len(df)},
    after_metrics={"output_size": len(result), "processing_time": execution_time}
)
```

### 4. **Tags para Filtrado**

```python
# Usar tags para facilitar el filtrado
logger.log_event(
    tags=["data_load", "csv", "large_file"],
    ...
)
```

## Troubleshooting

### Problemas Comunes

1. **Logs no se generan**:
   - Verificar que `json_logging.enabled: true` en config.yml
   - Comprobar permisos de escritura en directorio logs/

2. **Formato JSON inválido**:
   - Verificar que todos los campos sean serializables
   - Usar `ensure_ascii=False` para caracteres especiales

3. **Rendimiento lento**:
   - Configurar `console_json: false` en producción
   - Usar compresión si es necesario

4. **Sesiones no se cierran**:
   - Llamar `manager.close_session(session_id)` explícitamente
   - Configurar `session_timeout` apropiado

### Debugging

```python
# Habilitar logging detallado
import logging
logging.basicConfig(level=logging.DEBUG)

# Verificar configuración
logger = create_json_logger()
print(f"Session ID: {logger.session_id}")
print(f"Config: {logger.config}")
```

## Contribución

### Agregar Nuevas Categorías

```python
class LogCategory(Enum):
    # Categorías existentes...
    NEW_CATEGORY = "new_category"
```

### Agregar Nuevos Métodos de Logging

```python
def log_custom_event(self, custom_param: str, ...):
    """Registra un evento personalizado"""
    return self.log_event(
        level=LogLevel.INFO,
        message=f"Custom event: {custom_param}",
        category=LogCategory.SYSTEM,
        # ... otros parámetros
    )
```

### Testing

```python
def test_json_logging():
    logger = create_json_logger()
    
    # Registrar evento de prueba
    log_entry = logger.log_event(
        level=LogLevel.INFO,
        message="Test event",
        module="test",
        function="test_function",
        step="test_step",
        category=LogCategory.SYSTEM,
        parameters={},
        before_metrics={},
        after_metrics={},
        execution_time=0.0
    )
    
    # Verificar que se generó correctamente
    assert log_entry.session_id is not None
    assert log_entry.timestamp is not None
    assert log_entry.level == "INFO"
```

## Roadmap

### Versión 1.1
- [ ] Integración con Apache Kafka para streaming de logs
- [ ] Compresión automática de logs antiguos
- [ ] Rotación de archivos JSON

### Versión 1.2
- [ ] Dashboard web integrado para visualización de logs
- [ ] Alertas automáticas basadas en patrones de logs
- [ ] Análisis de rendimiento automático

### Versión 1.3
- [ ] Integración con sistemas de CI/CD
- [ ] Métricas de negocio automáticas
- [ ] Machine Learning para detección de anomalías

## Licencia

Este sistema de logging JSON es parte del proyecto Proyecto_J y sigue la misma licencia del proyecto principal.

## Contacto

Para preguntas, sugerencias o reportes de bugs, crear un issue en el repositorio del proyecto. 