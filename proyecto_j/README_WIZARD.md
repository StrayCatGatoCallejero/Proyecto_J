# 🧙‍♂️ Wizard de Visualización de Datos

Un asistente interactivo paso a paso para crear visualizaciones efectivas de datos en Streamlit.

## 🚀 Características Actuales

### ✅ Funcionalidades Implementadas

- **📁 Carga de múltiples formatos**: CSV, Excel (.xlsx, .xls), SPSS (.sav), Stata (.dta)
- **🔍 Detección automática de tipos**: Numérico, categórico, booleano, fecha/tiempo, texto
- **📊 Resumen automático de datos**: Estadísticas descriptivas, valores faltantes, tipos de datos
- **🎨 Visualizaciones univariadas**:
  - **Numéricas**: Histograma, Boxplot, Estadísticas descriptivas
  - **Categóricas**: Gráfico de barras, Gráfico de torta, Tabla de frecuencias
  - **Booleanas**: Gráfico de barras, Tabla de frecuencias
  - **Fechas**: Serie temporal
  - **Texto**: Tabla de frecuencias, Longitud de texto
- **📈 Visualizaciones interactivas**: Usando Plotly para mejor experiencia
- **💾 Exportación de resultados**: Datos y resúmenes en CSV
- **🎯 Interfaz intuitiva**: Wizard paso a paso con navegación clara

### 🎨 Tipos de Visualizaciones Disponibles

| Tipo de Variable | Visualizaciones Disponibles |
|------------------|------------------------------|
| **Numérico** | Histograma, Boxplot, Estadísticas descriptivas |
| **Categórico** | Gráfico de barras, Gráfico de torta, Tabla de frecuencias |
| **Booleano** | Gráfico de barras, Tabla de frecuencias |
| **Fecha/Tiempo** | Serie temporal |
| **Texto** | Tabla de frecuencias, Longitud de texto |

## 🛠️ Instalación y Uso

### 1. Instalar dependencias
```bash
pip install -r requirements_wizard.txt
```

### 2. Ejecutar el wizard
```bash
streamlit run wizard_streamlit.py
```

### 3. Probar funcionalidades
```bash
python test_wizard_simple.py
```

## 📋 Pasos del Wizard

1. **📁 Cargar archivo**: Sube tu archivo de datos
2. **📊 Resumen de datos**: Revisa información general y estadísticas
3. **🔍 Detección de tipos**: Análisis automático de tipos de variables
4. **💡 Sugerencias**: Recomendaciones de visualizaciones apropiadas
5. **🎨 Selección de gráfico**: Elige la visualización que prefieras
6. **📈 Visualización**: Genera y visualiza el gráfico interactivo
7. **💾 Exportar resultados**: Descarga datos y resúmenes

## 🏗️ Estructura Preparada para el Futuro

El wizard está diseñado con una arquitectura modular que facilita la expansión a visualizaciones bivariadas:

### 🔮 Próximas Mejoras Planificadas

#### **Fase 2: Visualizaciones Bivariadas**
- **📊 Gráficos de dispersión**: Relaciones entre dos variables numéricas
- **📈 Boxplots agrupados**: Distribuciones por categorías
- **🌡️ Gráficos de correlación**: Matrices de correlación interactivas
- **📊 Gráficos de barras apiladas**: Comparaciones entre categorías
- **📈 Gráficos de líneas múltiples**: Series temporales con múltiples variables

#### **Fase 3: Personalización Avanzada**
- **🎨 Personalización de colores**: Paletas de colores personalizables
- **📏 Ajustes de tamaño**: Control de dimensiones de gráficos
- **📝 Anotaciones**: Agregar títulos, etiquetas y notas
- **🔄 Temas visuales**: Diferentes estilos de gráficos

#### **Fase 4: Análisis Estadístico**
- **📊 Tests estadísticos**: Correlaciones, chi-cuadrado, t-test
- **📈 Regresiones simples**: Análisis de relaciones lineales
- **📊 Análisis de varianza**: Comparaciones entre grupos
- **📈 Pronósticos básicos**: Tendencias y proyecciones

#### **Fase 5: Funcionalidades Avanzadas**
- **💾 Guardar configuraciones**: Reutilizar configuraciones de visualización
- **📱 Exportación avanzada**: PNG, PDF, HTML interactivo
- **🔄 Batch processing**: Procesar múltiples archivos
- **📊 Dashboards**: Múltiples visualizaciones en una sola vista

## 🧪 Testing

El archivo `test_wizard_simple.py` incluye pruebas automatizadas para verificar:

- ✅ Carga de datos de ejemplo
- ✅ Detección automática de tipos
- ✅ Generación de sugerencias de visualización
- ✅ Creación de visualizaciones básicas
- ✅ Manejo de errores

## 📁 Estructura de Archivos

```
Proyecto_J/
├── wizard_streamlit.py          # Aplicación principal del wizard
├── requirements_wizard.txt      # Dependencias específicas
├── test_wizard_simple.py       # Pruebas automatizadas
├── README_WIZARD.md            # Esta documentación
└── data/                       # Datos de ejemplo
    ├── datos_ejemplo_chile.csv
    └── datos_ejemplo.sav
```

## 🎯 Casos de Uso

### **Investigación Académica**
- Análisis exploratorio de datos de encuestas
- Visualización de resultados de investigación
- Preparación de gráficos para publicaciones

### **Análisis de Negocios**
- Exploración de datos de ventas
- Análisis de comportamiento de clientes
- Reportes de métricas empresariales

### **Educación**
- Enseñanza de estadística descriptiva
- Demostración de tipos de visualización
- Práctica de análisis de datos

## 🔧 Personalización

### Agregar Nuevas Visualizaciones

Para agregar una nueva visualización, modifica la función `crear_visualizacion()`:

```python
elif tipo_vis == "Nueva Visualización" and tipo_col == "tipo_apropiado":
    # Tu código de visualización aquí
    fig = px.nueva_visualizacion(...)
    return fig
```

### Agregar Nuevos Tipos de Variables

Para detectar nuevos tipos, modifica `detectar_tipos_columnas()`:

```python
elif nueva_condicion:
    tipo = "nuevo_tipo"
    detalles = "Descripción del nuevo tipo"
```

## 🐛 Solución de Problemas

### Error de carga de archivo
- Verifica que el formato sea soportado (.csv, .xlsx, .xls, .sav, .dta)
- Asegúrate de que el archivo no esté corrupto
- Verifica que tenga encabezados en la primera fila

### Error de visualización
- Revisa que la variable seleccionada tenga datos válidos
- Verifica que el tipo de variable sea compatible con la visualización
- Intenta con otra visualización sugerida

### Problemas de rendimiento
- Para archivos grandes (>100MB), considera muestrear los datos
- Cierra otras aplicaciones para liberar memoria
- Usa el botón "Reiniciar Wizard" si hay problemas de estado

## 🤝 Contribuciones

Para contribuir al desarrollo del wizard:

1. **Reportar bugs**: Usa el sistema de issues
2. **Sugerir mejoras**: Propon nuevas funcionalidades
3. **Contribuir código**: Envía pull requests con mejoras
4. **Mejorar documentación**: Ayuda a mantener esta documentación actualizada

## 📄 Licencia

Este proyecto es parte del Proyecto J y sigue las mismas políticas de licencia.

---

**💡 Consejo**: El wizard está diseñado para ser intuitivo. Si tienes dudas, sigue las sugerencias automáticas y experimenta con diferentes visualizaciones para encontrar la que mejor represente tus datos. 