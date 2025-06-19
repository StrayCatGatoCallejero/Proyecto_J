# 🔢 Procesamiento Estadístico + Frontend

Una aplicación web completa para análisis estadístico de archivos `.sav` y `.dta` con interfaz moderna y funcionalidades avanzadas.

## 🚀 Características Principales

### 📊 **Análisis Estadístico Completo**
- **Estadísticas Descriptivas**: Media, mediana, moda, percentiles, desviación estándar
- **Análisis de Correlaciones**: Matrices de correlación Pearson y Spearman con heatmaps
- **Tablas de Contingencia**: Análisis χ² con interpretaciones detalladas
- **Visualizaciones Avanzadas**: Boxplots, scatter plots, diagramas de densidad y más

### 🔍 **Filtros Dinámicos**
- **Filtros por Rango**: Sliders para variables numéricas
- **Filtros por Categoría**: Multiselect para variables categóricas
- **Aplicación Global**: Los filtros se aplican a todos los análisis
- **Vista Previa**: Resumen de datos filtrados en tiempo real

### 📊 **Visualizaciones Avanzadas**
- **📦 Boxplots**: Distribución de variables numéricas con opción de agrupación
- **🔄 Scatter Plots**: Relaciones entre variables con líneas de regresión
- **📈 Diagramas de Densidad**: Distribuciones de probabilidad
- **📊 Histogramas con Densidad**: Combinación de histograma y curva de densidad
- **🎻 Violin Plots**: Distribuciones completas por grupos
- **📊 Gráficos de Barras**: Frecuencias y promedios
- **🔥 Heatmaps Avanzados**: Correlaciones con análisis adicional
- **🔗 Matrices de Scatter Plots**: Visualización de múltiples relaciones
- **📊 Paneles Completos**: Múltiples visualizaciones en una sola vista

### 📤 **Exportación Profesional**
- **Formatos Múltiples**: CSV, Excel, HTML
- **Reportes Completos**: Todos los análisis en un solo archivo
- **Datos Filtrados**: Exportación de conjuntos de datos personalizados
- **Interpretaciones**: Guías y explicaciones incluidas

## 🛠️ Instalación

1. **Clona el repositorio**:
```bash
git clone <url-del-repositorio>
cd Proyecto_J
```

2. **Instala las dependencias**:
```bash
pip install -r requirements.txt
```

3. **Ejecuta la aplicación**:
```bash
streamlit run app_front.py
```

## 📋 Dependencias

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
plotly>=5.15.0
pyreadstat>=1.1.0
openpyxl>=3.1.0
```

## 🎯 Uso de la Aplicación

### 1. **Carga de Datos**
- Sube archivos `.sav` (SPSS) o `.dta` (Stata)
- La aplicación detecta automáticamente el formato
- Vista previa inmediata de los datos

### 2. **Configuración de Filtros** (Opcional)
- **Variables Numéricas**: Usa sliders para definir rangos
- **Variables Categóricas**: Selecciona categorías específicas
- **Aplicación Global**: Los filtros afectan todos los análisis
- **Resumen en Tiempo Real**: Porcentaje de datos utilizados

### 3. **Análisis Estadístico**

#### 📈 **Estadísticas Básicas**
- Selecciona variables numéricas
- Obtén estadísticas descriptivas completas
- Visualiza histogramas automáticamente

#### 🔗 **Análisis de Correlaciones**
- Selecciona múltiples variables numéricas
- Elige entre correlación Pearson o Spearman
- Visualiza matrices con heatmaps interactivos
- Interpreta la fuerza de las correlaciones

#### 📊 **Tablas de Contingencia**
- Selecciona dos variables categóricas
- Obtén tablas de contingencia completas
- Realiza pruebas χ² de independencia
- Interpreta resultados con guías detalladas

#### 📊 **Visualizaciones Avanzadas**
- **Panel Completo**: Múltiples gráficos para una variable
- **Boxplots**: Distribución y outliers
- **Scatter Plots**: Relaciones con líneas de regresión
- **Diagramas de Densidad**: Distribuciones de probabilidad
- **Violin Plots**: Distribuciones por grupos
- **Gráficos de Barras**: Frecuencias y promedios
- **Heatmaps Avanzados**: Correlaciones con análisis adicional
- **Matrices de Scatter Plots**: Relaciones múltiples

### 4. **Exportación de Resultados**

#### 📄 **Exportación Individual**
- **CSV**: Datos y resultados en formato tabular
- **Excel**: Múltiples hojas organizadas
- **HTML**: Reportes formateados profesionalmente

#### 📊 **Reportes Completos**
- **Excel Completo**: Todas las hojas en un archivo
- **HTML Completo**: Reporte profesional con interpretaciones
- **Datos Filtrados**: Conjuntos personalizados

## 📊 Tipos de Visualizaciones Disponibles

### 📦 **Boxplots**
- **Uso**: Visualizar distribución y detectar outliers
- **Opciones**: Simple o agrupado por variable categórica
- **Información**: Mediana, cuartiles, valores atípicos

### 🔄 **Scatter Plots**
- **Uso**: Analizar relaciones entre variables numéricas
- **Características**: Líneas de regresión automáticas
- **Opciones**: Coloreado por variable categórica

### 📈 **Diagramas de Densidad**
- **Uso**: Visualizar distribuciones de probabilidad
- **Ventajas**: No dependen del número de bins
- **Opciones**: Agrupación por variables categóricas

### 📊 **Histogramas con Densidad**
- **Uso**: Combinar histograma y curva de densidad
- **Beneficios**: Información completa de la distribución
- **Aplicaciones**: Análisis de normalidad

### 🎻 **Violin Plots**
- **Uso**: Comparar distribuciones entre grupos
- **Ventajas**: Muestra la forma completa de la distribución
- **Aplicaciones**: Análisis por grupos categóricos

### 📊 **Gráficos de Barras**
- **Uso**: Visualizar frecuencias y promedios
- **Tipos**: Frecuencias simples o promedios por grupo
- **Aplicaciones**: Análisis de variables categóricas

### 🔥 **Heatmaps de Correlación Avanzados**
- **Uso**: Análisis completo de correlaciones
- **Características**: Matriz + gráfico de correlaciones más fuertes
- **Información**: Top 10 correlaciones destacadas

### 🔗 **Matrices de Scatter Plots**
- **Uso**: Visualizar todas las relaciones entre variables
- **Límite**: Hasta 6 variables para claridad
- **Incluye**: Diagramas de densidad en la diagonal

### 📊 **Paneles Completos**
- **Uso**: Análisis exhaustivo de una variable
- **Contenido**: 4 visualizaciones diferentes
- **Opciones**: Con o sin agrupación

## 🔧 Características Técnicas

### **Gestión de Estado**
- **Session State**: Filtros persistentes entre secciones
- **Datos de Análisis**: Resultados guardados para exportación
- **Interfaz Reactiva**: Actualizaciones automáticas

### **Persistencia de Sesiones**
- **Datos Cargados**: Los archivos permanecen cargados entre recargas
- **Filtros Aplicados**: Configuraciones de filtros se mantienen
- **Selecciones de Usuario**: Variables y configuraciones persistentes
- **Análisis Realizados**: Resultados guardados para exportación
- **Navegación**: Estado mantenido al cambiar entre secciones
- **Limpieza de Sesión**: Botón para reiniciar completamente la aplicación

### **Validaciones**
- **Formato de Archivos**: Verificación automática
- **Variables Disponibles**: Detección de tipos de datos
- **Filtros Válidos**: Validación de rangos y categorías

### **Optimización**
- **Cálculos Eficientes**: Reutilización de resultados
- **Memoria**: Gestión optimizada de datos grandes
- **Interfaz**: Carga progresiva de componentes

## 📈 Interpretación de Resultados

### **Correlaciones**
- **0.7-1.0**: Muy fuerte positiva
- **0.5-0.7**: Fuerte positiva
- **0.3-0.5**: Moderada positiva
- **0.1-0.3**: Débil positiva
- **-0.1-0.1**: Sin correlación
- **-0.3-(-0.1)**: Débil negativa
- **-0.5-(-0.3)**: Moderada negativa
- **-0.7-(-0.5)**: Fuerte negativa
- **-1.0-(-0.7)**: Muy fuerte negativa

### **Pruebas χ²**
- **p < 0.001**: Muy altamente significativa
- **p < 0.01**: Altamente significativa
- **p < 0.05**: Significativa
- **p ≥ 0.05**: No significativa

### **Cramer's V**
- **< 0.1**: Efecto muy pequeño
- **0.1-0.3**: Efecto pequeño
- **0.3-0.5**: Efecto moderado
- **> 0.5**: Efecto grande

## 🎨 Personalización

### **Configuración de Gráficos**
- **Tamaños**: Automáticos según tipo de visualización
- **Colores**: Paletas profesionales de seaborn
- **Estilos**: Formato consistente en toda la aplicación

### **Opciones de Exportación**
- **Formatos**: CSV, Excel, HTML
- **Contenido**: Personalizable por sección
- **Calidad**: Alta resolución para gráficos

## 🚀 Próximas Mejoras

- [ ] **Gráficos Interactivos**: Plotly para zoom y hover
- [ ] **Análisis de Series Temporales**: Para datos longitudinales
- [ ] **Tests Estadísticos Adicionales**: t-tests, ANOVA, regresión
- [ ] **Machine Learning**: Clustering y clasificación básica
- [ ] **Reportes PDF**: Generación automática de PDFs
- [ ] **Base de Datos**: Almacenamiento de análisis previos
- [ ] **Colaboración**: Compartir análisis entre usuarios

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Soporte

Para soporte técnico o preguntas:
- Abre un issue en GitHub
- Contacta al equipo de desarrollo
- Consulta la documentación completa

---

**Desarrollado con ❤️ para la comunidad estadística** 