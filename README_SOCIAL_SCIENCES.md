# 🔬 Analizador Avanzado de Datos de Ciencias Sociales

Un sistema completo de análisis de datos especializado para investigaciones en ciencias sociales, encuestas y estudios demográficos.

## 🌟 Características Principales

### 🏷️ Clasificación Semántica de Columnas
- **Detección automática** del tipo de pregunta basada en diccionarios de términos clave
- **Categorías especializadas**: demográfico, socioeconómico, opinión, escalas Likert, temporal, geográfico
- **Diccionario personalizable** para términos específicos de tu investigación
- **Confianza configurable** en las clasificaciones

### 📏 Detección de Escalas Ordinales y Likert
- **Identificación automática** de escalas tipo "Muy de acuerdo - Totalmente en desacuerdo"
- **Escalas numéricas** (1-5, 1-7, 1-10)
- **Sugerencias de visualización** específicas para cada tipo de escala
- **Mapeo interno** como variables ordinales

### 🔄 Normalización y Unificación de Categorías
- **Fuzzy matching** con algoritmo Levenshtein
- **Agrupamiento automático** de variantes ("Masculino", "M", "Hombre" → "Masculino")
- **Umbral configurable** de similitud
- **Reporte de agrupamientos** para revisión manual

### 📐 Reconocimiento de Unidades de Medida
- **Detección automática** de unidades en nombres de columnas y valores
- **Tipos soportados**: tiempo, distancia, moneda
- **Conversiones automáticas** a unidades estándar
- **Flexibilidad** para diferentes formatos

### 📝 Procesamiento de Texto Libre
- **Tokenización y conteo** de palabras clave
- **Análisis de frecuencia** de términos
- **Detección automática** de columnas de texto libre
- **Sugerencias de visualización** (nube de palabras, gráficos de barras)

### ✅ Validación de Consistencia Interna
- **Detección de casos contradictorios** (edad < 18 y estado civil "Casado")
- **Reglas basadas en lógica** cruzada de variables
- **Alertas configurables** para inconsistencias
- **Reporte detallado** de problemas encontrados

### 📊 Sugerencias Automáticas de Visualización
- **Basadas en tipo semántico** de columna
- **Demográficas** → barras horizontales o pastel
- **Temporales** → líneas o áreas
- **Escalas Likert** → barras apiladas
- **Texto libre** → nube de palabras

### 📋 Anotaciones y Metadatos
- **Metadatos completos** para cada columna detectada
- **Trazabilidad** en session_state de Streamlit
- **Exportación** de resultados en JSON
- **Tooltips informativos** en la interfaz

## 🚀 Instalación

### Requisitos Previos
```bash
Python 3.8 o superior
```

### Instalación de Dependencias
```bash
pip install -r requirements_social_sciences.txt
```

### Dependencias Principales
- `pandas` - Análisis de datos
- `fuzzywuzzy` - Comparación de strings
- `plotly` - Visualizaciones interactivas
- `streamlit` - Interfaz web
- `wordcloud` - Nubes de palabras
- `scikit-learn` - Análisis estadístico

## 📖 Uso Básico

### 1. Análisis Simple
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

### 2. Análisis con Configuración Personalizada
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

# Obtener resumen ejecutivo
summary = analyzer.get_analysis_summary()
print(summary)
```

### 3. Interfaz Web con Streamlit
```bash
streamlit run social_sciences_streamlit.py
```

## 🎯 Casos de Uso Específicos

### Encuestas de Satisfacción
```python
# El analizador detectará automáticamente:
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

## 🔧 Configuración Avanzada

### Diccionario Semántico Personalizado
```python
custom_dictionary = {
    'mi_categoria': {
        'mi_subcategoria': [
            'termino1',
            'termino2',
            'termino3'
        ]
    }
}
```

### Umbrales de Configuración
```python
# En la interfaz de Streamlit:
# - Umbral de Similitud: 50-95% (default: 80%)
# - Umbral de Confianza: 50-95% (default: 70%)
```

### Mapeos de Normalización
```python
# El sistema incluye mapeos predefinidos para:
# - Género (Masculino, Femenino, No binario, etc.)
# - Estado civil (Soltero, Casado, Divorciado, etc.)
# - Educación (Básica, Media, Técnica, etc.)
```

## 📊 Visualizaciones Disponibles

### Gráficos Automáticos
- **Barras horizontales** - Para variables categóricas
- **Gráficos de pastel** - Para distribuciones demográficas
- **Histogramas** - Para variables continuas
- **Barras apiladas** - Para escalas Likert
- **Nubes de palabras** - Para texto libre
- **Gráficos de líneas** - Para datos temporales

### Personalización
- **Colores temáticos** por categoría
- **Tooltips informativos** con metadatos
- **Interactividad** completa con Plotly
- **Exportación** en múltiples formatos

## 🔍 Ejemplos de Análisis

### Ejemplo 1: Encuesta de Satisfacción Laboral
```python
# Datos de entrada:
# - edad, genero, antiguedad, satisfaccion_laboral, comentarios

# Análisis automático detecta:
# - Variable demográfica: edad, genero
# - Variable temporal: antiguedad
# - Escala Likert: satisfaccion_laboral
# - Texto libre: comentarios

# Sugerencias de visualización:
# - Histograma para edad
# - Gráfico de pastel para género
# - Barras apiladas para satisfacción
# - Nube de palabras para comentarios
```

### Ejemplo 2: Estudio de Confianza Institucional
```python
# Datos de entrada:
# - region, edad, confianza_gobierno, confianza_congreso, comentarios

# Análisis automático detecta:
# - Variable geográfica: region
# - Variable demográfica: edad
# - Escalas de confianza: confianza_gobierno, confianza_congreso
# - Texto libre: comentarios

# Validaciones de consistencia:
# - Verifica coherencia entre variables demográficas
# - Detecta patrones en respuestas de confianza
```

## 📈 Métricas y Reportes

### Resumen Ejecutivo
- **Clasificación semántica** de todas las columnas
- **Escalas detectadas** con tipos y valores
- **Normalizaciones sugeridas** con mapeos
- **Problemas de consistencia** encontrados
- **Metadatos completos** del análisis

### Exportación de Resultados
- **Formato JSON** con todos los resultados
- **Metadatos estructurados** para análisis posterior
- **Configuraciones guardadas** para reutilización
- **Reportes personalizables** por tipo de análisis

## 🛠️ Desarrollo y Extensión

### Añadir Nuevas Categorías
```python
# Extender el diccionario semántico
new_category = {
    'mi_nueva_categoria': {
        'mi_subcategoria': ['termino1', 'termino2']
    }
}

analyzer = SocialSciencesAnalyzer(new_category)
```

### Personalizar Validaciones
```python
# Añadir reglas de consistencia personalizadas
def custom_consistency_check(df):
    # Tu lógica personalizada aquí
    pass
```

### Crear Visualizaciones Específicas
```python
# Integrar con Plotly para gráficos personalizados
import plotly.express as px

def custom_visualization(df, column):
    # Tu visualización personalizada aquí
    pass
```

## 🤝 Contribuciones

### Cómo Contribuir
1. **Fork** el repositorio
2. **Crea** una rama para tu feature
3. **Implementa** tus mejoras
4. **Añade** tests si es necesario
5. **Envía** un pull request

### Áreas de Mejora
- **Nuevos algoritmos** de clasificación semántica
- **Más tipos** de escalas y validaciones
- **Integración** con APIs de NLP
- **Visualizaciones** adicionales
- **Optimización** de rendimiento

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Soporte

### Documentación
- **Ejemplos completos** en `ejemplo_analisis_social.py`
- **Tutorial interactivo** en la interfaz de Streamlit
- **Documentación de API** en los docstrings

### Comunidad
- **Issues** en GitHub para reportar bugs
- **Discussions** para preguntas y sugerencias
- **Wiki** para documentación adicional

---

**¡Transforma tus datos de ciencias sociales en insights valiosos con nuestro analizador avanzado!** 🚀 