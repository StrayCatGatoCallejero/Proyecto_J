# 🤖 Proyecto J - Sistema de Análisis de Datos para Ciencias Sociales

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Descripción

**Proyecto J** es un sistema completo de análisis de datos diseñado específicamente para investigadores en ciencias sociales, demógrafos y analistas de encuestas. Combina herramientas avanzadas de visualización, análisis estadístico y procesamiento de datos en una interfaz web moderna e intuitiva.

### 🎯 Características Principales

- **🔍 Análisis Exploratorio Avanzado** - Detección automática de tipos de datos y visualizaciones sugeridas
- **📊 Visualizaciones Interactivas** - Gráficos dinámicos con Plotly y estadísticas descriptivas
- **🧠 Consultas en Lenguaje Natural** - Análisis de datos mediante preguntas en español
- **📈 Análisis de Tendencias** - Detección automática de patrones temporales
- **🏛️ Ciencias Sociales** - Herramientas especializadas para encuestas y demografía
- **📋 Validación de Datos** - Verificación automática de datos chilenos y latinoamericanos
- **🔄 Pipeline Modular** - Arquitectura flexible y extensible
- **📝 Logging JSON** - Monitoreo detallado de procesos
- **⚡ Manejo Robusto de Errores** - Sistema de recuperación y reportes

## 🏗️ Arquitectura del Sistema

```
Proyecto_J/
├── proyecto_j/                    # 🎯 Aplicación principal
│   ├── streamlit_app.py          # 🌐 App web unificada
│   ├── src/                      # 📦 Módulos core
│   │   ├── core.py              # 🔧 Pipeline principal
│   │   ├── steps.py             # 📋 Pasos del pipeline
│   │   ├── utils.py             # 🛠️ Utilidades
│   │   ├── estadistica.py       # 📊 Análisis estadístico
│   │   ├── ciencias_sociales.py # 🏛️ Ciencias sociales
│   │   ├── nl_query.py          # 🧠 Consultas naturales
│   │   ├── nl_query_trends.py   # 📈 Análisis de tendencias
│   │   ├── complex_grouping.py  # 🔀 Agrupaciones complejas
│   │   ├── validacion_chile.py  # 🇨🇱 Validación Chile
│   │   ├── analisis_demografico.py # 👥 Demografía
│   │   ├── pipeline_encuestas.py   # 📋 Encuestas
│   │   ├── variable_classifier.py  # 🏷️ Clasificador variables
│   │   └── column_inspector.py     # 🔍 Inspector columnas
│   ├── data/                    # 📁 Datos de ejemplo
│   ├── notebooks/               # 📓 Jupyter notebooks
│   └── tests/                   # 🧪 Tests unitarios
├── processing/                   # ⚙️ Procesamiento avanzado
│   ├── json_logging.py          # 📝 Logging JSON
│   ├── business_rules.py        # 📋 Reglas de negocio
│   ├── data_validators.py       # ✅ Validadores
│   ├── visualization.py         # 📊 Visualización
│   ├── stats.py                 # 📈 Estadísticas
│   └── config_manager.py        # ⚙️ Configuración
├── orchestrator/                 # 🎼 Orquestación
│   └── pipeline_orchestrator.py # 🎯 Orquestador principal
├── examples/                     # 💡 Ejemplos de uso
├── docs/                         # 📚 Documentación
├── scripts/                      # 🔧 Scripts de instalación
├── tests/                        # 🧪 Tests unificados
│   ├── unit/                    # 🧪 Tests unitarios
│   ├── integration/             # 🔗 Tests de integración
│   ├── e2e/                     # 🌐 Tests end-to-end
│   ├── fixtures/                # 📁 Datos de prueba
│   └── conftest.py              # ⚙️ Configuración pytest
└── logs/                         # 📝 Logs del sistema
```

## 🚀 Instalación

### Requisitos Previos

- **Python 3.8+** (recomendado 3.9+)
- **Git** para clonar el repositorio
- **Navegador web** moderno

### Instalación Automática

#### Windows
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/Proyecto_J.git
cd Proyecto_J

# Ejecutar instalador automático
scripts\install.bat
```

#### Linux/macOS
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/Proyecto_J.git
cd Proyecto_J

# Ejecutar instalador automático
chmod +x scripts/install.sh
./scripts/install.sh
```

### Instalación Manual

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno virtual
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements_unified.txt

# 4. Verificar instalación
python scripts/check_python_version.py
```

## 🎯 Ejecución de la Aplicación

### Aplicación Principal (Streamlit)

La aplicación principal está ubicada en `proyecto_j/streamlit_app.py`:

```bash
# Ejecutar aplicación web
streamlit run proyecto_j/streamlit_app.py --server.port 8502 --server.headless false
```

**Acceso:** http://localhost:8502

### Características de la App Web

1. **📁 Carga de Datos** - Soporte para CSV, Excel, SPSS, JSON
2. **📊 Resumen Automático** - Estadísticas descriptivas y análisis de datos faltantes
3. **🔍 Detección de Tipos** - Clasificación automática de variables
4. **💡 Sugerencias** - Visualizaciones recomendadas por tipo de dato
5. **📈 Visualizaciones** - Gráficos interactivos y configurables
6. **🧠 Análisis Avanzado** - Consultas en lenguaje natural
7. **🏛️ Ciencias Sociales** - Análisis especializados para encuestas
8. **📊 Estadísticas Avanzadas** - Correlaciones, tablas de contingencia
9. **📤 Exportación** - PDF, Excel, HTML con resultados

### CLI (Línea de Comandos)

```bash
# Ejecutar pipeline desde CLI
python proyecto_j/src/cli.py run --config config/config.yml
```

### Ejemplos de Uso

```bash
# Ejecutar ejemplos específicos
python examples/ejemplo_sistema_completo.py
python examples/ejemplo_analisis_social.py
python examples/asistente_encuestas.py
```

## 📦 Módulos Principales

### 🔧 Core (`proyecto_j/src/`)

- **`core.py`** - Pipeline principal del sistema
- **`steps.py`** - Pasos del procesamiento de datos
- **`utils.py`** - Utilidades generales
- **`cli.py`** - Interfaz de línea de comandos

### 📊 Análisis Estadístico (`proyecto_j/src/estadistica.py`)

- Estadísticas descriptivas avanzadas
- Análisis de correlaciones
- Tests de hipótesis
- Visualizaciones estadísticas
- Exportación de resultados

### 🏛️ Ciencias Sociales (`proyecto_j/src/ciencias_sociales.py`)

- Análisis de encuestas
- Índices de desigualdad (Gini, Theil)
- Análisis de regresión
- Clustering de respuestas
- Validación de escalas

### 🧠 Consultas Naturales (`proyecto_j/src/nl_query.py`)

- Procesamiento de lenguaje natural
- Análisis de tendencias temporales
- Agrupaciones complejas
- Consultas inteligentes en español

### 🇨🇱 Validación Chile (`proyecto_j/src/validacion_chile.py`)

- Validación de datos geográficos chilenos
- Verificación de códigos de región/comuna
- Análisis de consistencia demográfica
- Reportes de validación

### 📋 Encuestas (`proyecto_j/src/pipeline_encuestas.py`)

- Procesamiento de encuestas
- Análisis de respuestas múltiples
- Validación de escalas
- Reportes de encuestas

### ⚙️ Procesamiento (`processing/`)

- **`json_logging.py`** - Sistema de logging JSON
- **`business_rules.py`** - Reglas de negocio
- **`data_validators.py`** - Validación de datos
- **`visualization.py`** - Visualizaciones avanzadas
- **`stats.py`** - Estadísticas computacionales

## 📝 Logging y Monitoreo

El sistema incluye un sistema de logging JSON completo:

```python
# Ejemplo de log generado
{
    "timestamp": "2024-01-15T10:30:00",
  "level": "INFO",
    "module": "pipeline_encuestas",
    "function": "procesar_encuesta",
    "message": "Encuesta procesada exitosamente",
    "data": {
        "filas_procesadas": 1500,
        "columnas_validadas": 25,
        "tiempo_procesamiento": 2.5
    }
}
```

**Ubicación de logs:** `logs/`

## 🧪 Testing

El proyecto incluye una suite de tests completa y unificada organizada en tres niveles:

### 📁 Estructura de Tests

```
tests/
├── unit/                    # 🧪 Tests unitarios
│   ├── test_config_manager.py
│   ├── test_stats.py
│   ├── test_advanced_stats.py
│   └── test_*.py
├── integration/             # 🔗 Tests de integración
│   ├── test_pipeline.py
│   ├── test_survey_analysis.py
│   ├── test_features_complete.py
│   └── test_*.py
├── e2e/                     # 🌐 Tests end-to-end
│   ├── test_app_workflow.py
│   ├── test_streamlit_e2e.py
│   └── test_*.py
├── fixtures/                # 📁 Datos de prueba
│   ├── sample_data.csv
│   └── test_config.yml
└── conftest.py              # ⚙️ Configuración pytest
```

### 🚀 Ejecución de Tests

```bash
# Ejecutar todos los tests
pytest tests/

# Tests por categoría
pytest tests/unit/           # Tests unitarios
pytest tests/integration/    # Tests de integración
pytest tests/e2e/            # Tests end-to-end

# Tests específicos
pytest tests/unit/test_stats.py
pytest tests/integration/test_pipeline.py

# Tests con cobertura
pytest --cov=proyecto_j tests/
pytest --cov=processing tests/
pytest --cov=orchestrator tests/

# Tests con reporte HTML
pytest --cov=proyecto_j --cov-report=html tests/
```

### 📊 Estado Actual de Tests

**✅ Completado:**
- Migración completa de tests a estructura unificada
- Organización en unit/integration/e2e
- Fixtures y configuración centralizada
- Tests de configuración y logging

**⚠️ Requiere Atención:**
- Algunos tests de configuración necesitan ajustes de rutas
- Tests de validación de esquemas requieren actualización
- Tests E2E necesitan dependencias adicionales (playwright)

**🔧 Próximos Pasos:**
- Corregir imports en tests legacy
- Actualizar esquemas de validación
- Instalar dependencias para tests E2E
- Mejorar cobertura de tests

### 🛠️ Configuración de Tests

El archivo `tests/conftest.py` contiene la configuración centralizada para todos los tests, incluyendo:
- Fixtures de datos de prueba
- Configuración de logging
- Manejo de rutas de archivos
- Configuración de pytest

## 🔧 Configuración

### Archivo de Configuración (`config/config.yml`)

```yaml
# Configuración del pipeline
input:
  path: "data/datos_ejemplo.csv"
  format: "csv"
  encoding: "utf-8"

processing:
  clean_data: true
  validate_types: true
  handle_missing: true

output:
  format: "pdf"
  path: "resultados/"
  include_charts: true

logging:
  level: "INFO"
  format: "json"
  file: "logs/pipeline.log"
```

## 🚀 Características Avanzadas

### 🔄 Pipeline Modular

El sistema utiliza una arquitectura de pipeline modular que permite:

- **Flexibilidad** - Agregar/quitar pasos fácilmente
- **Reutilización** - Componentes independientes
- **Testing** - Pruebas unitarias por módulo
- **Configuración** - Parámetros por YAML/JSON

### 🧠 Análisis Inteligente

- **Detección automática** de tipos de datos
- **Sugerencias inteligentes** de visualizaciones
- **Consultas en lenguaje natural** en español
- **Análisis de tendencias** automático

### 📊 Visualizaciones Avanzadas

- **Plotly** - Gráficos interactivos
- **Seaborn** - Visualizaciones estadísticas
- **Matplotlib** - Gráficos personalizados
- **Exportación** a múltiples formatos

## 🔮 Mejoras Futuras

### 🚀 Características en Desarrollo

1. **⚡ Sistema Asíncrono Completo**
   - **Estado:** Código base implementado en `scripts/tasks.py` y `proyecto_j/utils/run_async_system.py`
   - **Requisito:** Configuración de Redis y Celery por el usuario
   - **Beneficio:** Procesamiento paralelo y monitoreo en tiempo real
   - **Implementación:** `pip install -r scripts/requirements_async.txt`

2. **🤖 IA y Machine Learning**
   - Predicción automática de tipos de datos
   - Sugerencias de análisis basadas en patrones
   - Detección automática de anomalías
   - Recomendaciones de visualizaciones

3. **🌐 API REST**
   - Endpoints para integración con otros sistemas
   - Documentación automática con Swagger
   - Autenticación y autorización
   - Rate limiting y caching

4. **📱 Aplicación Móvil**
   - Interfaz responsive para tablets
   - Notificaciones push de resultados
   - Sincronización offline
   - Captura de datos en campo

5. **🔗 Integración con Bases de Datos**
   - Conexión directa a PostgreSQL/MySQL
   - Sincronización automática
   - Consultas SQL optimizadas
   - Backup automático

6. **📊 Dashboard en Tiempo Real**
   - Métricas en vivo
   - Alertas automáticas
   - Gráficos dinámicos
   - Exportación programada

### 🛠️ Mejoras Técnicas Pendientes

1. **📦 Optimización de Módulos**
   - Refactorización de `estadistica.py` (38KB, 1130 líneas)
   - Modularización de `nl_query_trends.py` (41KB, 1139 líneas)
   - Optimización de `pipeline_encuestas.py` (38KB, 1126 líneas)

2. **🔧 Consistencia de Nombres**
   - Estandarización de nombres de archivos
   - Corrección de rutas en documentación
   - Unificación de convenciones

3. **📚 Documentación**
   - Documentación automática con Sphinx
   - Ejemplos interactivos
   - Guías de mejores prácticas
   - Tutoriales paso a paso

4. **🧪 Testing**
   - Cobertura de tests al 90%+
   - Tests de integración E2E
   - Tests de rendimiento
   - Tests de seguridad

## 🤝 Contribución

### Cómo Contribuir

1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Crea** un Pull Request

### Estándares de Código

- **Python:** PEP 8, type hints
- **Documentación:** Docstrings en español
- **Tests:** Cobertura mínima 80%
- **Commits:** Mensajes descriptivos en español

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Soporte

### Canales de Ayuda

- **📧 Email:** soporte@proyectoj.com
- **💬 Discord:** [Servidor de la comunidad](https://discord.gg/proyectoj)
- **📖 Wiki:** [Documentación completa](https://github.com/tu-usuario/Proyecto_J/wiki)
- **🐛 Issues:** [Reportar problemas](https://github.com/tu-usuario/Proyecto_J/issues)

### FAQ

**Q: ¿Puedo usar el sistema sin instalar Redis?**
A: Sí, el sistema funciona completamente sin Redis. Solo necesitas Redis para las características asíncronas avanzadas.

**Q: ¿Qué formatos de datos soporta?**
A: CSV, Excel (.xlsx, .xls), SPSS (.sav), JSON, y más.

**Q: ¿Es compatible con Python 3.7?**
A: Se requiere Python 3.8+ para todas las funcionalidades.

**Q: ¿Puedo personalizar las visualizaciones?**
A: Sí, todas las visualizaciones son completamente configurables.

---

**Desarrollado con ❤️ para la comunidad de ciencias sociales** 