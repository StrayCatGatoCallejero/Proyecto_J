# Proyecto J - Sistema de Análisis de Datos para Ciencias Sociales

Un sistema completo y robusto para análisis de datos en ciencias sociales, encuestas y demografía, con capacidades avanzadas de procesamiento, visualización y logging estructurado.

## 🏗️ Estructura del Proyecto

```
Proyecto_J/
├── README.md                    # Documentación principal
├── pyproject.toml              # Configuración del proyecto
├── requirements.txt            # Dependencias principales
├── runtime.txt                 # Versión de Python
├── .python-version             # Versión específica de Python
├── .flake8                     # Configuración de linting
├── mypy.ini                    # Configuración de type checking
├── __init__.py                 # Inicialización del paquete
│
├── src_folder/                 # Código fuente principal
│   ├── streamlit_app.py        # Aplicación Streamlit principal
│   └── streamlit_app_backup.py # Backup de la aplicación
│
├── processing/                 # Módulos de procesamiento de datos
│   ├── __init__.py
│   ├── io.py                   # Operaciones de entrada/salida
│   ├── stats.py                # Análisis estadístico
│   ├── filters.py              # Filtros de datos
│   ├── business_rules.py       # Reglas de negocio
│   ├── business_error_handler.py # Manejo de errores de negocio
│   ├── json_logging.py         # Sistema de logging JSON
│   ├── config_manager.py       # Gestión de configuración
│   ├── data_validators.py      # Validación de datos
│   ├── features.py             # Generación de features
│   ├── visualization.py        # Generación de visualizaciones
│   └── ...
│
├── orchestrator/               # Orquestadores de pipeline
│   ├── __init__.py
│   └── pipeline_orchestrator.py # Orquestador principal
│
├── config/                     # Configuraciones
│   └── config.yml              # Configuración principal
│
├── examples/                   # Ejemplos y demostraciones
│   ├── ejemplo_*.py            # Scripts de ejemplo
│   ├── test_*.py               # Tests de ejemplo
│   ├── app_*.py                # Aplicaciones de ejemplo
│   ├── social_sciences_*.py    # Análisis de ciencias sociales
│   ├── wizard_*.py             # Asistentes interactivos
│   └── asistente_*.py          # Asistentes de análisis
│
├── scripts/                    # Scripts utilitarios
│   ├── install.sh              # Instalación en Linux/macOS
│   ├── install.bat             # Instalación en Windows
│   ├── check_python_version.py # Verificación de versión
│   ├── tasks.py                # Tareas automatizadas
│   └── requirements_*.txt      # Dependencias específicas
│
├── docs/                       # Documentación extendida
│   ├── README_*.md             # Documentación específica
│   ├── RESUMEN_*.md            # Resúmenes técnicos
│   └── REPORTE_*.md            # Reportes de análisis
│
├── data_folder/                # Datos de ejemplo y archivos
│   ├── *.csv                   # Datos en formato CSV
│   ├── *.xlsx                  # Datos en formato Excel
│   ├── *.png                   # Imágenes generadas
│   └── ...
│
├── notebooks/                  # Jupyter notebooks
│   └── *.ipynb                 # Notebooks de análisis
│
├── tests/                      # Tests automáticos
│   ├── test_*.py               # Tests unitarios
│   ├── e2e/                    # Tests end-to-end
│   └── ...
│
├── logs/                       # Logs y reportes
│   ├── *.json                  # Logs JSON estructurados
│   ├── *.log                   # Logs de texto
│   └── *.xml                   # Reportes de cobertura
│
├── proyecto_j/                 # Código fuente adicional
│   ├── src/                    # Módulos adicionales
│   ├── tests/                  # Tests adicionales
│   ├── data/                   # Datos adicionales
│   └── ...
│
└── temp/                       # Archivos temporales
    └── ...
```

## 🚀 Instalación

### Requisitos Previos
- Python 3.11.7 o superior
- pip o conda

### Instalación Rápida

**Windows:**
```bash
scripts\install.bat
```

**Linux/macOS:**
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

**Manual:**
```bash
pip install -r requirements.txt
```

### Verificación de Instalación
```bash
python scripts/check_python_version.py
```

## 📖 Uso

### Aplicación Principal (Streamlit)
```bash
streamlit run src_folder/streamlit_app.py
```

### Ejemplos de Uso

**Análisis de Encuestas:**
```bash
python examples/ejemplo_sistema_completo_final.py
```

**Análisis de Ciencias Sociales:**
```bash
python examples/social_sciences_analyzer.py
```

**Pipeline de Datos:**
```bash
python examples/ejemplo_pipeline_json_logging.py
```

## 🔧 Características Principales

### 1. Sistema de Logging JSON Robusto
- Logs estructurados en formato JSON
- Rotación automática de archivos
- Compatible con ELK Stack, Datadog, Prometheus
- Métricas detalladas de ejecución
- Manejo robusto de serialización

### 2. Manejo de Errores de Negocio
- Extracción automática de errores de logs
- Visualización en Streamlit
- Exportación de reportes
- Categorización por tipo de error

### 3. Pipeline de Procesamiento
- Orquestador modular y extensible
- Validación de esquemas
- Aplicación de reglas de negocio
- Análisis estadístico automático
- Generación de visualizaciones

### 4. Configuración Flexible
- Configuración centralizada en YAML
- Variables de entorno
- Configuración por ambiente

## 📊 Logging y Monitoreo

### Ubicación de Logs
- **Logs JSON:** `logs/pipeline.json`
- **Logs de texto:** `logs/pipeline.log`
- **Reportes:** `logs/coverage.xml`

### Estructura de Logs JSON
```json
{
  "level": "INFO",
  "event": "step",
  "message": "Paso completado",
  "module": "pipeline_orchestrator",
  "function": "run_pipeline",
  "step": "data_loading",
  "category": "data_load",
  "parameters": {...},
  "before_metrics": {...},
  "after_metrics": {...},
  "execution_time": 0.25,
  "tags": ["pipeline", "data"],
  "metadata": {...},
  "timestamp": "2025-06-30T04:53:18.989911+00:00",
  "session_id": "pipeline_20250630_005318",
  "system_info": {...}
}
```

### Visualización de Logs

**Kibana:**
- Importar logs desde `logs/pipeline.json`
- Crear dashboards con filtros por `step`, `level`, `session_id`

**Grafana:**
- Configurar fuente de datos JSON
- Crear paneles con métricas de `execution_time`, `success_rate`

**Python:**
```python
import json
from processing.json_logging import JsonLogger

# Leer logs
with open('logs/pipeline.json', 'r') as f:
    logs = [json.loads(line) for line in f]

# Analizar métricas
execution_times = [log['execution_time'] for log in logs if 'execution_time' in log]
```

## 🧪 Testing

### Ejecutar Tests
```bash
# Tests unitarios
pytest tests/

# Tests con cobertura
pytest --cov=processing --cov=orchestrator tests/

# Tests end-to-end
pytest tests/e2e/
```

### Verificar Calidad de Código
```bash
# Linting
flake8 .

# Type checking
mypy .

# Formateo
black .
```

## 📚 Documentación

- **README.md** - Documentación principal
- **docs/README_*.md** - Documentación específica por módulo
- **docs/RESUMEN_*.md** - Resúmenes técnicos
- **docs/REPORTE_*.md** - Reportes de análisis

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Estándares de Código
- Seguir PEP 8
- Incluir docstrings
- Escribir tests para nuevas funcionalidades
- Usar type hints

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

Para soporte técnico o preguntas:
- Revisar la documentación en `docs/`
- Ejecutar ejemplos en `examples/`
- Verificar logs en `logs/`

## 🔄 Changelog

### v2.0.0 (2025-06-30)
- ✅ Reestructuración completa del proyecto
- ✅ Sistema de logging JSON robusto
- ✅ Manejo avanzado de errores de negocio
- ✅ Pipeline orquestador modular
- ✅ Serialización robusta para todos los tipos de datos
- ✅ Documentación completa y ejemplos
- ✅ Compatibilidad con Python 3.11+
- ✅ Tests automatizados y CI/CD

---

**Proyecto J** - Transformando el análisis de datos en ciencias sociales con tecnología moderna y robusta. 