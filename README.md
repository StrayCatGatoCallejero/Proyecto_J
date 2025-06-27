# Proyecto J - Sistema Integral de Análisis de Datos para Ciencias Sociales

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/tu_usuario/proyecto_j/actions/workflows/ci.yml/badge.svg)](https://github.com/tu_usuario/proyecto_j/actions)

> **Una plataforma completa de análisis de datos especializada en ciencias sociales, encuestas y estudios demográficos con interfaces modernas y funcionalidades avanzadas.**

## Características Principales

### Análisis Estadístico Completo
- **Estadísticas descriptivas** avanzadas con interpretación automática
- **Análisis de correlaciones** Pearson y Spearman con heatmaps interactivos
- **Tablas de contingencia** con análisis χ² completo
- **Regresión múltiple** con validación de supuestos
- **Análisis de valores perdidos** con estrategias de imputación

### Especializado en Ciencias Sociales
- **Clasificación semántica automática** de variables de encuesta
- **Detección de escalas Likert** y ordinales
- **Normalización inteligente** de categorías con fuzzy matching
- **Validación de consistencia** lógica entre variables
- **Análisis de texto libre** con nubes de palabras

### Visualizaciones Avanzadas
- **15+ tipos de gráficos** interactivos con Plotly
- **Sugerencias automáticas** de visualización por tipo de dato
- **Paneles completos** de análisis multivariado
- **Exportación profesional** en múltiples formatos
- **Diseño responsive** para diferentes dispositivos

### Pipeline Automatizado
- **Procesamiento modular** con orquestador inteligente
- **Soporte multi-formato** (CSV, Excel, SPSS, Stata)
- **Validaciones específicas** para datos chilenos
- **Generación automática** de reportes en PDF y HTML
- **Sistema de logging** completo

## Instalación Rápida

### Requisitos Previos
```bash
Python 3.11 o superior (recomendado Python 3.11.7)
```

### Instalación con pip
```bash
# Clonar el repositorio
git clone https://github.com/tu_usuario/proyecto_j.git
cd proyecto_j

# Crear entorno virtual con Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación principal
streamlit run app_front.py
```

### Instalación con conda
```bash
# Crear entorno conda con Python 3.11
conda env create -f environment.yml
conda activate proyecto_j

# Ejecutar aplicación
streamlit run app_front.py
```

### Verificación de Instalación
```bash
# Verificar versión de Python
python --version  # Debe mostrar Python 3.11.x

# Verificar instalación de dependencias
python -c "import streamlit, pandas, plotly, scipy; print('✅ Todas las dependencias instaladas correctamente')"
```

## Aplicaciones Disponibles

### 1. Aplicación Principal (`app_front.py`)
**Interfaz unificada con 7 secciones especializadas**
```bash
streamlit run app_front.py
```

**Características:**
- **Filtros dinámicos** por rango y categoría
- **Estadísticas básicas** completas
- **Análisis de correlaciones** avanzado
- **Tablas de contingencia** con χ²
- **Visualizaciones avanzadas** (15+ tipos)
- **Análisis de ciencias sociales** especializado
- **Exportación** en múltiples formatos

### 2. Estadísticas Avanzadas (`app_estadistica_avanzada.py`)
**Análisis estadístico completo con interfaz moderna**
```bash
streamlit run app_estadistica_avanzada.py
```

### 3. Análisis de Encuestas (`app_encuestas.py`)
**Procesamiento especializado para datos de encuestas**
```bash
streamlit run app_encuestas.py
```

### 4. Wizard Interactivo (`wizard_streamlit.py`)
**Guía paso a paso para análisis completo**
```bash
streamlit run wizard_streamlit.py
```

### 5. Ciencias Sociales (`social_sciences_streamlit.py`)
**Análisis especializado para investigación social**
```bash
streamlit run social_sciences_streamlit.py
```

## Uso Básico

### Análisis Simple desde Python
```python
from social_sciences_analyzer import analyze_survey_data
import pandas as pd

# Cargar datos
df = pd.read_csv('mi_encuesta.csv')

# Análisis automático
results = analyze_survey_data(df)

# Ver resumen
print(results['semantic_classification'])
```

### Análisis con Configuración Personalizada
```python
from social_sciences_analyzer import SocialSciencesAnalyzer

# Diccionario personalizado
custom_dict = {
    'escalas_especializadas': {
        'satisfaccion_vida': ['satisfaccion_vida', 'felicidad', 'bienestar'],
        'confianza_institucional': ['confianza_gobierno', 'confianza_congreso']
    }
}

# Crear analizador
analyzer = SocialSciencesAnalyzer(custom_dict)

# Analizar datos
results = analyzer.analyze_dataframe(df)
```

### Pipeline Completo desde CLI
```bash
python -m proyecto_j.src.cli run --config config/config.yml
```

## Casos de Uso

### Encuestas de Satisfacción
```python
# El sistema detectará automáticamente:
# - Escalas Likert de satisfacción
# - Variables demográficas
# - Texto libre de comentarios
# - Sugerirá visualizaciones apropiadas
```

### Estudios Demográficos
```python
# Funcionalidades especiales:
# - Normalización de categorías de género
# - Detección de inconsistencias edad/estado civil
# - Clasificación de variables socioeconómicas
# - Análisis de distribución geográfica
```

### Investigación Política
```python
# Características específicas:
# - Escalas de confianza institucional
# - Análisis de participación política
# - Procesamiento de respuestas abiertas
# - Detección de sentimientos en texto
```

## Configuración Avanzada

### Diccionario Semántico Personalizado
```yaml
# config/config.yml
semantic_dictionary:
  mi_categoria:
    mi_subcategoria:
      - termino1
      - termino2
      - termino3
```

### Umbrales de Configuración
```python
# En la interfaz de Streamlit:
# - Umbral de Similitud: 50-95% (default: 80%)
# - Umbral de Confianza: 50-95% (default: 70%)
```

### Validaciones para Chile
```python
# Validaciones específicas incluidas:
# - Regiones y comunas chilenas
# - Indicadores socioeconómicos
# - Validaciones demográficas
```

## Visualizaciones Disponibles

### Gráficos Automáticos
- **Barras horizontales** - Para variables categóricas
- **Gráficos de pastel** - Para distribuciones demográficas
- **Histogramas** - Para variables continuas
- **Barras apiladas** - Para escalas Likert
- **Nubes de palabras** - Para texto libre
- **Gráficos de líneas** - Para datos temporales
- **Boxplots** - Para distribución de variables
- **Scatter plots** - Para relaciones bivariadas
- **Heatmaps** - Para correlaciones múltiples
- **Violin plots** - Para distribuciones completas

### Personalización
- **Colores temáticos** por categoría
- **Tooltips informativos** con metadatos
- **Interactividad completa** con Plotly
- **Exportación** en múltiples formatos

## Arquitectura del Proyecto

```
proyecto_j/
├── Aplicaciones Principales
│   ├── app_front.py              # Aplicación principal
│   ├── app_estadistica_avanzada.py
│   ├── app_encuestas.py
│   ├── wizard_streamlit.py
│   └── social_sciences_streamlit.py
├── Módulos de Procesamiento
│   ├── processing/               # Pipeline de procesamiento
│   ├── orchestrator/             # Orquestador principal
│   └── proyecto_j/src/           # Módulos especializados
├── Datos y Configuración
│   ├── config/                   # Configuraciones
│   ├── data/                     # Datos de ejemplo
│   └── logs/                     # Registros
└── Testing y Documentación
    ├── tests/                    # Pruebas automatizadas
    ├── notebooks/                # Jupyter notebooks
    └── docs/                     # Documentación
```

## Testing

### Ejecutar Tests
```bash
# Tests completos
pytest tests/

# Con cobertura
pytest --cov=proyecto_j tests/

# Tests específicos
pytest tests/test_pipeline.py
pytest tests/test_column_inspector.py
```

### Linting
```bash
# Verificar estilo de código
flake8 proyecto_j/

# Formatear código
black proyecto_j/
```

## Dependencias Principales

```txt
# Análisis de datos
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.1.0

# Visualización
plotly>=5.0.0
matplotlib>=3.5.0
wordcloud>=1.9.0

# Interfaz web
streamlit>=1.28.0

# Procesamiento de archivos
openpyxl>=3.0.0
pyreadstat>=1.1.0

# Análisis estadístico
statsmodels
missingno

# Reportes
fpdf2==2.5.6
```

## Contribuir

### Cómo Contribuir
1. **Fork** el proyecto
2. **Crea** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

### Guías de Contribución
- **Sigue** las convenciones de código existentes
- **Añade tests** para nuevas funcionalidades
- **Actualiza** la documentación según sea necesario
- **Verifica** que todos los tests pasen

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## Agradecimientos

- **Streamlit** por la excelente plataforma de desarrollo
- **Plotly** por las visualizaciones interactivas
- **Pandas** por el procesamiento de datos
- **Comunidad de Python** por las herramientas de análisis estadístico

## Contacto

- **Proyecto:** [Proyecto J](https://github.com/tu_usuario/proyecto_j)
- **Issues:** [GitHub Issues](https://github.com/tu_usuario/proyecto_j/issues)
- **Discusiones:** [GitHub Discussions](https://github.com/tu_usuario/proyecto_j/discussions)

## Estadísticas del Proyecto

- **Stars:** [![GitHub stars](https://img.shields.io/github/stars/tu_usuario/proyecto_j.svg)](https://github.com/tu_usuario/proyecto_j/stargazers)
- **Forks:** [![GitHub forks](https://img.shields.io/github/forks/tu_usuario/proyecto_j.svg)](https://github.com/tu_usuario/proyecto_j/network)
- **Issues:** [![GitHub issues](https://img.shields.io/github/issues/tu_usuario/proyecto_j.svg)](https://github.com/tu_usuario/proyecto_j/issues)
- **Releases:** [![GitHub release](https://img.shields.io/github/release/tu_usuario/proyecto_j.svg)](https://github.com/tu_usuario/proyecto_j/releases)

---

<div align="center">

**¿Te gustó el proyecto? ¡Dale una estrella!**

*Desarrollado para la comunidad de ciencias sociales*

</div> 