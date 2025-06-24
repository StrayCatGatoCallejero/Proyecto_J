#  Procesamiento Estadístico + Frontend

Una aplicación web completa para análisis estadístico de archivos `.sav` y `.dta` con interfaz moderna y funcionalidades avanzadas.

##  Características Principales

### **Análisis Estadístico Completo**
- **Estadísticas Descriptivas**: Media, mediana, moda, percentiles, desviación estándar
- **Análisis de Correlaciones**: Matrices de correlación Pearson y Spearman con heatmaps
- **Tablas de Contingencia**: Análisis χ² con interpretaciones detalladas
- **Visualizaciones Avanzadas**: Boxplots, scatter plots, diagramas de densidad y más
- **Ciencias Sociales**: Análisis especializado para investigación social y demográfica

###  **Análisis para Ciencias Sociales**
- **Clasificación Automática**: Detección automática del tipo y dominio de variables
- **Análisis Descriptivo Especializado**: Interpretación específica para variables sociales
- **Análisis Bivariado Avanzado**: Correlaciones, contingencia y diferencias de medias
- **Regresión Múltiple**: Con validación completa de supuestos
- **Análisis de Clusters**: Identificación de grupos en datos sociales
- **Índices Especializados**: Gini, calidad de vida, desarrollo humano
- **Manejo de Valores Perdidos**: Análisis de patrones y sugerencias de imputación

###  **Filtros Dinámicos**
- **Filtros por Rango**: Sliders para variables numéricas
- **Filtros por Categoría**: Multiselect para variables categóricas
- **Aplicación Global**: Los filtros se aplican a todos los análisis
- **Vista Previa**: Resumen de datos filtrados en tiempo real

###  **Visualizaciones Avanzadas**
- **Boxplots**: Distribución de variables numéricas con opción de agrupación
- **Scatter Plots**: Relaciones entre variables con líneas de regresión
- **Diagramas de Densidad**: Distribuciones de probabilidad
- **Histogramas con Densidad**: Combinación de histograma y curva de densidad
- **Violin Plots**: Distribuciones completas por grupos
- **Gráficos de Barras**: Frecuencias y promedios
- **Heatmaps Avanzados**: Correlaciones con análisis adicional
- **Matrices de Scatter Plots**: Visualización de múltiples relaciones
- **Paneles Completos**: Múltiples visualizaciones en una sola vista

### **Exportación Profesional**
- **Formatos Múltiples**: CSV, Excel, HTML
- **Reportes Completos**: Todos los análisis en un solo archivo
- **Datos Filtrados**: Exportación de conjuntos de datos personalizados
- **Interpretaciones**: Guías y explicaciones incluidas

## Instalación

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

##  Dependencias

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

##  Uso de la Aplicación

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

####  **Estadísticas Básicas**
- Selecciona variables numéricas
- Obtén estadísticas descriptivas completas
- Visualiza histogramas automáticamente

####  **Análisis de Correlaciones**
- Selecciona múltiples variables numéricas
- Elige entre correlación Pearson o Spearman
- Visualiza matrices con heatmaps interactivos
- Interpreta la fuerza de las correlaciones

####  **Tablas de Contingencia**
- Selecciona dos variables categóricas
- Obtén tablas de contingencia completas
- Realiza pruebas χ² de independencia
- Interpreta resultados con guías detalladas

####  **Visualizaciones Avanzadas**
- **Panel Completo**: Múltiples gráficos para una variable
- **Boxplots**: Distribución y outliers
- **Scatter Plots**: Relaciones con líneas de regresión
- **Diagramas de Densidad**: Distribuciones de probabilidad
- **Violin Plots**: Distribuciones por grupos
- **Gráficos de Barras**: Frecuencias y promedios
- **Heatmaps Avanzados**: Correlaciones con análisis adicional
- **Matrices de Scatter Plots**: Relaciones múltiples

### 4. **Exportación de Resultados**

####  **Exportación Individual**
- **CSV**: Datos y resultados en formato tabular
- **Excel**: Múltiples hojas organizadas
- **HTML**: Reportes formateados profesionalmente

####  **Reportes Completos**
- **Excel Completo**: Todas las hojas en un archivo
- **HTML Completo**: Reporte profesional con interpretaciones
- **Datos Filtrados**: Conjuntos personalizados

##  Tipos de Visualizaciones Disponibles

###  **Boxplots**
- **Uso**: Visualizar distribución y detectar outliers
- **Opciones**: Simple o agrupado por variable categórica
- **Información**: Mediana, cuartiles, valores atípicos

###  **Scatter Plots**
- **Uso**: Analizar relaciones entre variables numéricas
- **Características**: Líneas de regresión automáticas
- **Opciones**: Coloreado por variable categórica

###  **Diagramas de Densidad**
- **Uso**: Visualizar distribuciones de probabilidad
- **Ventajas**: No dependen del número de bins
- **Opciones**: Agrupación por variables categóricas

###  **Histogramas con Densidad**
- **Uso**: Combinar histograma y curva de densidad
- **Beneficios**: Información completa de la distribución
- **Aplicaciones**: Análisis de normalidad

###  **Violin Plots**
- **Uso**: Comparar distribuciones entre grupos
- **Ventajas**: Muestra la forma completa de la distribución
- **Aplicaciones**: Análisis por grupos categóricos

###  **Gráficos de Barras**
- **Uso**: Visualizar frecuencias y promedios
- **Tipos**: Frecuencias simples o promedios por grupo
- **Aplicaciones**: Análisis de variables categóricas

###  **Heatmaps de Correlación Avanzados**
- **Uso**: Análisis completo de correlaciones
- **Características**: Matriz + gráfico de correlaciones más fuertes
- **Información**: Top 10 correlaciones destacadas

###  **Matrices de Scatter Plots**
- **Uso**: Visualizar todas las relaciones entre variables
- **Límite**: Hasta 6 variables para claridad
- **Incluye**: Diagramas de densidad en la diagonal

###  **Paneles Completos**
- **Uso**: Análisis exhaustivo de una variable
- **Contenido**: 4 visualizaciones diferentes
- **Opciones**: Con o sin agrupación

##  Características Técnicas

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

##  Interpretación de Resultados

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

##  Personalización

### **Configuración de Gráficos**
- **Tamaños**: Automáticos según tipo de visualización
- **Colores**: Paletas profesionales de seaborn
- **Estilos**: Formato consistente en toda la aplicación

### **Opciones de Exportación**
- **Formatos**: CSV, Excel, HTML
- **Contenido**: Personalizable por sección
- **Calidad**: Alta resolución para gráficos

##  Próximas Mejoras

- [ ] **Gráficos Interactivos**: Plotly para zoom y hover
- [ ] **Análisis de Series Temporales**: Para datos longitudinales
- [ ] **Tests Estadísticos Adicionales**: t-tests, ANOVA, regresión
- [ ] **Machine Learning**: Clustering y clasificación básica
- [ ] **Reportes PDF**: Generación automática de PDFs
- [ ] **Base de Datos**: Almacenamiento de análisis previos
- [ ] **Colaboración**: Compartir análisis entre usuarios

##  Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

##  Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

##  Soporte

Para soporte técnico o preguntas:
- Abre un issue en GitHub
- Contacta al equipo de desarrollo
- Consulta la documentación completa

---

**Desarrollado con ❤️ para la comunidad estadística**

##  Análisis para Ciencias Sociales

### **Variables Demográficas**
- **Edad**: Análisis por rangos etarios y generaciones
- **Género e Identidad**: Categorías ampliadas y dimensiones de expresión
- **Estado Civil**: Tipos de unión y duración de relaciones
- **Nacionalidad y Ciudadanía**: Estatus migratorio y tiempo de residencia
- **Etnia y Raza**: Autoidentificación y pertenencia a pueblos originarios
- **Lengua Materna**: Multilingüismo y uso en diferentes contextos

### **Variables Socioeconómicas**
- **Ingresos y Riqueza**: Análisis de desigualdad y distribución
- **Empleo y Trabajo**: Formalidad, condiciones contractuales y sindicalización
- **Pobreza y Vulnerabilidad**: Indicadores multidimensionales
- **Vivienda y Hábitat**: Calidad constructiva y servicios básicos

### **Variables Educativas y de Salud**
- **Nivel de Escolaridad**: Años de educación y competencias
- **Estado de Salud**: Enfermedades crónicas y cobertura sanitaria
- **Salud Mental**: Escalas de depresión y ansiedad
- **Hábitos de Vida**: Consumo de sustancias y actividad física

### **Variables Culturales y Políticas**
- **Religión y Espiritualidad**: Práctica y afiliación religiosa
- **Participación Política**: Voto, identidad partidaria y confianza institucional
- **Valores y Actitudes**: Postmaterialismo y cohesión social
- **Tecnología y Comunicación**: Brecha digital y uso de redes sociales

### **Análisis Estadísticos Especializados**

#### **A. Clasificación Automática**
- **Detección de Tipo**: Continua, categórica, ordinal, binaria
- **Dominio de Variables**: Demográfico, socioeconómico, educativo, etc.
- **Validación de Datos**: Detección de outliers y valores atípicos

#### **B. Análisis Descriptivo Especializado**
- **Interpretación Contextual**: Según el dominio de la variable
- **Estadísticas Robustas**: Resistentes a outliers
- **Análisis de Distribución**: Normalidad y transformaciones

#### **C. Análisis Bivariado Avanzado**
- **Correlaciones Múltiples**: Pearson, Spearman, Kendall
- **Tablas de Contingencia**: Chi-cuadrado con medidas de asociación
- **Análisis de Grupos**: ANOVA y pruebas no paramétricas

#### **D. Regresión Múltiple**
- **Validación de Supuestos**: Normalidad, homocedasticidad, independencia
- **Multicolinealidad**: Detección y manejo
- **Diagnóstico de Residuos**: Análisis completo de residuos

#### **E. Análisis de Clusters**
- **K-means**: Identificación de grupos naturales
- **Caracterización**: Perfiles de cada cluster
- **Validación**: Métricas de calidad del clustering

#### **F. Índices Especializados**
- **Coeficiente de Gini**: Medida de desigualdad
- **Índice de Calidad de Vida**: Compuesto multidimensional
- **Índice de Desarrollo Humano**: Simplificado

#### **G. Manejo de Valores Perdidos**
- **Análisis de Patrones**: Detección de patrones sistemáticos
- **Sugerencias de Imputación**: Métodos apropiados por tipo de variable
- **Validación de Imputación**: Verificación de calidad

### **Recomendaciones para Investigación Social**

#### **Diseño de Investigación**
- **Muestreo Representativo**: Consideraciones para encuestas
- **Ponderación de Datos**: Ajustes por estratificación
- **Tamaño de Muestra**: Cálculos de potencia estadística

#### **Análisis Ético**
- **Anonimización**: Protección de datos personales
- **Consentimiento Informado**: Cumplimiento de estándares éticos
- **Transparencia**: Documentación completa de métodos

#### **Comparabilidad**
- **Estandarización**: Métodos para comparación internacional
- **Indicadores Compuestos**: Construcción de índices
- **Validación Cruzada**: Verificación de robustez

---

## 🤝 Agradecimientos y Reconocimientos

Este proyecto se ha enriquecido gracias a la inspiración, ideas y herramientas de la comunidad open source. Agradecemos especialmente a los siguientes proyectos y autores:

- **[pandas](https://pandas.pydata.org/)** y **[pyreadstat](https://github.com/Roche/pyreadstat)**: Carga y manipulación de datos en múltiples formatos.
- **[missingno](https://github.com/ResidentMario/missingno)**: Visualización de valores faltantes.
- **[scikit-learn](https://scikit-learn.org/)**: Imputación y procesamiento avanzado de datos.
- **[dataprep](https://github.com/sfu-db/dataprep)**: Generación de reportes EDA interactivos.
- **[statsmodels](https://www.statsmodels.org/)**: Modelos estadísticos y análisis de encuestas.
- **[FPDF](https://pyfpdf.github.io/)**: Exportación de resultados a PDF.
- **[LisaKey/convert-csv-to-sav](https://github.com/LisaKey/convert-csv-to-sav)**, **[jiogenes/predicting_missing_value](https://github.com/jiogenes/predicting_missing_value)**, **[Ree2612/csv-to-pdf-report-generator](https://github.com/Ree2612/csv-to-pdf-report-generator)**: Inspiración y ejemplos para robustecer la gestión de datos y reportes.

Agradecemos a la comunidad de desarrolladores y científic@s de datos por compartir conocimiento y herramientas que hacen posible sistemas más robustos, transparentes y útiles para todos.

--- 