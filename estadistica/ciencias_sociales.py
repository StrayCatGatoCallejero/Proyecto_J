# estadistica/ciencias_sociales.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, bartlett, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CLASIFICACIÓN AUTOMÁTICA DE VARIABLES
# ============================================================================

def clasificar_variable(df: pd.DataFrame, columna: str) -> dict:
    """
    Clasifica automáticamente una variable según su tipo y características.
    """
    valores = df[columna].dropna()
    
    # Detectar tipo de datos
    tipo_dato = df[columna].dtype
    
    # Contar valores únicos
    n_unicos = valores.nunique()
    n_total = len(valores)
    
    # Calcular proporción de valores únicos
    prop_unicos = n_unicos / n_total if n_total > 0 else 0
    
    # Detectar si es categórica
    es_categorica = (
        tipo_dato == 'object' or 
        tipo_dato == 'category' or 
        (tipo_dato in ['int64', 'float64'] and prop_unicos < 0.1 and n_unicos < 20)
    )
    
    # Detectar si es ordinal
    es_ordinal = False
    if es_categorica:
        # Buscar patrones de orden en categorías
        categorias = valores.unique()
        if len(categorias) <= 7:  # Típico para escalas Likert
            es_ordinal = True
    
    # Detectar si es continua
    es_continua = (
        tipo_dato in ['int64', 'float64'] and 
        not es_categorica and 
        prop_unicos > 0.1
    )
    
    # Detectar si es binaria
    es_binaria = n_unicos == 2
    
    # Detectar si es temporal
    es_temporal = pd.api.types.is_datetime64_any_dtype(df[columna])
    
    # Clasificar dominio de ciencias sociales
    dominio = clasificar_dominio_cs(columna, valores, es_categorica)
    
    return {
        'columna': columna,
        'tipo_dato': str(tipo_dato),
        'n_total': n_total,
        'n_unicos': n_unicos,
        'prop_unicos': prop_unicos,
        'es_categorica': es_categorica,
        'es_ordinal': es_ordinal,
        'es_continua': es_continua,
        'es_binaria': es_binaria,
        'es_temporal': es_temporal,
        'dominio': dominio,
        'valores_faltantes': df[columna].isna().sum(),
        'porcentaje_faltantes': (df[columna].isna().sum() / len(df)) * 100
    }

def clasificar_dominio_cs(columna: str, valores, es_categorica: bool) -> str:
    """
    Clasifica la variable según el dominio de ciencias sociales.
    """
    columna_lower = columna.lower()
    
    # Variables demográficas
    if any(palabra in columna_lower for palabra in ['edad', 'age', 'años', 'generacion']):
        return 'Demográfico - Edad'
    elif any(palabra in columna_lower for palabra in ['genero', 'sexo', 'gender', 'identidad']):
        return 'Demográfico - Género'
    elif any(palabra in columna_lower for palabra in ['estado_civil', 'civil', 'matrimonio']):
        return 'Demográfico - Estado Civil'
    elif any(palabra in columna_lower for palabra in ['nacionalidad', 'pais', 'ciudadania']):
        return 'Demográfico - Nacionalidad'
    elif any(palabra in columna_lower for palabra in ['etnia', 'raza', 'indigena']):
        return 'Demográfico - Etnia/Raza'
    
    # Variables socioeconómicas
    elif any(palabra in columna_lower for palabra in ['ingreso', 'salario', 'renta', 'income']):
        return 'Socioeconómico - Ingresos'
    elif any(palabra in columna_lower for palabra in ['empleo', 'trabajo', 'ocupacion', 'job']):
        return 'Socioeconómico - Empleo'
    elif any(palabra in columna_lower for palabra in ['educacion', 'escolaridad', 'estudios']):
        return 'Educativo'
    elif any(palabra in columna_lower for palabra in ['vivienda', 'hogar', 'casa']):
        return 'Vivienda y Hábitat'
    
    # Variables de salud
    elif any(palabra in columna_lower for palabra in ['salud', 'enfermedad', 'health']):
        return 'Salud'
    elif any(palabra in columna_lower for palabra in ['religion', 'religioso', 'espiritual']):
        return 'Religioso/Espiritual'
    elif any(palabra in columna_lower for palabra in ['politica', 'voto', 'partido']):
        return 'Político/Cívico'
    elif any(palabra in columna_lower for palabra in ['tecnologia', 'internet', 'digital']):
        return 'Tecnología y Comunicación'
    
    # Por defecto
    return 'General'

# ============================================================================
# ANÁLISIS DESCRIPTIVOS ESPECIALIZADOS
# ============================================================================

def analisis_descriptivo_cs(df: pd.DataFrame, columna: str) -> dict:
    """
    Realiza análisis descriptivo especializado para ciencias sociales.
    """
    clasificacion = clasificar_variable(df, columna)
    valores = df[columna].dropna()
    
    resultado = {
        'clasificacion': clasificacion,
        'estadisticas_basicas': {},
        'distribucion': {},
        'interpretacion': {}
    }
    
    # Estadísticas básicas según tipo de variable
    if clasificacion['es_continua']:
        resultado['estadisticas_basicas'] = {
            'n': len(valores),
            'media': float(valores.mean()),
            'mediana': float(valores.median()),
            'moda': valores.mode().tolist(),
            'desv_estandar': float(valores.std()),
            'varianza': float(valores.var()),
            'minimo': float(valores.min()),
            'maximo': float(valores.max()),
            'rango': float(valores.max() - valores.min()),
            'q1': float(valores.quantile(0.25)),
            'q3': float(valores.quantile(0.75)),
            'rango_intercuartil': float(valores.quantile(0.75) - valores.quantile(0.25)),
            'coef_variacion': float(valores.std() / valores.mean()) if valores.mean() != 0 else 0,
            'asimetria': float(stats.skew(valores)),
            'curtosis': float(stats.kurtosis(valores))
        }
        
        # Prueba de normalidad
        if len(valores) >= 3 and len(valores) <= 5000:
            try:
                shapiro_stat, shapiro_p = shapiro(valores)
                resultado['distribucion']['normalidad_shapiro'] = {
                    'estadistico': float(shapiro_stat),
                    'p_valor': float(shapiro_p),
                    'es_normal': shapiro_p > 0.05
                }
            except:
                resultado['distribucion']['normalidad_shapiro'] = None
    
    elif clasificacion['es_categorica']:
        frecuencias = valores.value_counts()
        frecuencias_rel = valores.value_counts(normalize=True) * 100
        
        resultado['estadisticas_basicas'] = {
            'n': len(valores),
            'n_categorias': len(frecuencias),
            'moda': frecuencias.index[0],
            'frecuencia_moda': int(frecuencias.iloc[0]),
            'porcentaje_moda': float(frecuencias_rel.iloc[0]),
            'indice_diversidad': float(1 - ((frecuencias / len(valores)) ** 2).sum()),
            'entropia': float(stats.entropy(frecuencias))
        }
        
        resultado['distribucion']['frecuencias'] = {
            'absolutas': frecuencias.to_dict(),
            'relativas': frecuencias_rel.to_dict()
        }
    
    # Interpretación específica por dominio
    resultado['interpretacion'] = interpretar_resultados_cs(columna, resultado, clasificacion)
    
    return resultado

def interpretar_resultados_cs(columna: str, resultado: dict, clasificacion: dict) -> dict:
    """
    Proporciona interpretación específica para ciencias sociales.
    """
    interpretacion = {}
    
    if clasificacion['dominio'] == 'Demográfico - Edad':
        if clasificacion['es_continua']:
            media_edad = resultado['estadisticas_basicas']['media']
            if media_edad < 25:
                interpretacion['poblacion'] = "Población predominantemente joven"
            elif media_edad < 45:
                interpretacion['poblacion'] = "Población de edad media"
            else:
                interpretacion['poblacion'] = "Población envejecida"
    
    elif clasificacion['dominio'] == 'Socioeconómico - Ingresos':
        if clasificacion['es_continua']:
            media_ingreso = resultado['estadisticas_basicas']['media']
            mediana_ingreso = resultado['estadisticas_basicas']['mediana']
            
            # Detectar desigualdad
            if media_ingreso > mediana_ingreso * 1.5:
                interpretacion['desigualdad'] = "Presencia de desigualdad (media > mediana)"
            else:
                interpretacion['desigualdad'] = "Distribución relativamente equitativa"
    
    elif clasificacion['dominio'] == 'Educativo':
        if clasificacion['es_continua']:
            media_educacion = resultado['estadisticas_basicas']['media']
            if media_educacion < 6:
                interpretacion['nivel'] = "Bajo nivel educativo promedio"
            elif media_educacion < 12:
                interpretacion['nivel'] = "Nivel educativo medio"
            else:
                interpretacion['nivel'] = "Alto nivel educativo promedio"
    
    return interpretacion

# ============================================================================
# ANÁLISIS BIVARIADOS ESPECIALIZADOS
# ============================================================================

def analisis_bivariado_cs(df: pd.DataFrame, var1: str, var2: str) -> dict:
    """
    Realiza análisis bivariado especializado para ciencias sociales.
    """
    clasif1 = clasificar_variable(df, var1)
    clasif2 = clasificar_variable(df, var2)
    
    # Filtrar datos válidos para ambas variables
    df_limpio = df[[var1, var2]].dropna()
    
    resultado = {
        'variables': {
            'var1': {'nombre': var1, 'clasificacion': clasif1},
            'var2': {'nombre': var2, 'clasificacion': clasif2}
        },
        'n_observaciones': len(df_limpio),
        'analisis': {}
    }
    
    # Análisis según combinación de tipos
    if clasif1['es_continua'] and clasif2['es_continua']:
        resultado['analisis'] = analisis_correlacion_continua(df_limpio, var1, var2)
    
    elif clasif1['es_categorica'] and clasif2['es_categorica']:
        resultado['analisis'] = analisis_contingencia_categorica(df_limpio, var1, var2)
    
    elif (clasif1['es_continua'] and clasif2['es_categorica']) or (clasif1['es_categorica'] and clasif2['es_continua']):
        # Determinar cuál es continua y cuál categórica
        if clasif1['es_continua']:
            var_continua, var_categorica = var1, var2
        else:
            var_continua, var_categorica = var2, var1
        
        resultado['analisis'] = analisis_grupos_continua(df_limpio, var_continua, var_categorica)
    
    # Interpretación específica
    resultado['interpretacion'] = interpretar_analisis_bivariado_cs(resultado)
    
    return resultado

def analisis_correlacion_continua(df: pd.DataFrame, var1: str, var2: str) -> dict:
    """
    Análisis de correlación entre variables continuas.
    """
    x = df[var1]
    y = df[var2]
    
    # Correlaciones
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    kendall_tau, kendall_p = stats.kendalltau(x, y)
    
    # Interpretar fuerza de correlación
    def interpretar_correlacion(r):
        if abs(r) >= 0.7:
            return "Muy fuerte"
        elif abs(r) >= 0.5:
            return "Fuerte"
        elif abs(r) >= 0.3:
            return "Moderada"
        elif abs(r) >= 0.1:
            return "Débil"
        else:
            return "Muy débil"
    
    return {
        'tipo': 'correlacion_continua',
        'pearson': {
            'coeficiente': float(pearson_r),
            'p_valor': float(pearson_p),
            'significativo': pearson_p < 0.05,
            'fuerza': interpretar_correlacion(pearson_r)
        },
        'spearman': {
            'coeficiente': float(spearman_r),
            'p_valor': float(spearman_p),
            'significativo': spearman_p < 0.05,
            'fuerza': interpretar_correlacion(spearman_r)
        },
        'kendall': {
            'coeficiente': float(kendall_tau),
            'p_valor': float(kendall_p),
            'significativo': kendall_p < 0.05,
            'fuerza': interpretar_correlacion(kendall_tau)
        }
    }

def analisis_contingencia_categorica(df: pd.DataFrame, var1: str, var2: str) -> dict:
    """
    Análisis de contingencia entre variables categóricas.
    """
    # Tabla de contingencia
    tabla = pd.crosstab(df[var1], df[var2], margins=True, margins_name='Total')
    
    # Chi-cuadrado
    chi2_stat, chi2_p, dof, expected = chi2_contingency(pd.crosstab(df[var1], df[var2]))
    
    # Coeficiente de contingencia
    n = tabla.loc['Total', 'Total']
    cramer_v = np.sqrt(chi2_stat / (n * min(tabla.shape[0]-1, tabla.shape[1]-1)))
    
    # Porcentajes
    porcentajes_fila = pd.crosstab(df[var1], df[var2], normalize='index') * 100
    porcentajes_columna = pd.crosstab(df[var1], df[var2], normalize='columns') * 100
    porcentajes_total = pd.crosstab(df[var1], df[var2], normalize='all') * 100
    
    return {
        'tipo': 'contingencia_categorica',
        'tabla_contingencia': tabla.to_dict(),
        'chi_cuadrado': {
            'estadistico': float(chi2_stat),
            'p_valor': float(chi2_p),
            'grados_libertad': int(dof),
            'significativo': chi2_p < 0.05
        },
        'cramer_v': float(cramer_v),
        'porcentajes': {
            'por_fila': porcentajes_fila.to_dict(),
            'por_columna': porcentajes_columna.to_dict(),
            'del_total': porcentajes_total.to_dict()
        }
    }

def analisis_grupos_continua(df: pd.DataFrame, var_continua: str, var_categorica: str) -> dict:
    """
    Análisis de diferencias de medias entre grupos.
    """
    grupos = df.groupby(var_categorica)[var_continua]
    
    # Estadísticas por grupo
    estadisticas_grupos = {}
    for grupo, datos in grupos:
        estadisticas_grupos[str(grupo)] = {
            'n': len(datos),
            'media': float(datos.mean()),
            'mediana': float(datos.median()),
            'desv_estandar': float(datos.std()),
            'error_estandar': float(datos.std() / np.sqrt(len(datos)))
        }
    
    # ANOVA
    grupos_lista = [grupo for _, grupo in grupos]
    f_stat, p_valor_anova = stats.f_oneway(*grupos_lista)
    
    # Test de Levene para homogeneidad de varianzas
    try:
        levene_stat, levene_p = levene(*grupos_lista)
        homogeneidad_varianzas = levene_p > 0.05
    except:
        levene_stat, levene_p = None, None
        homogeneidad_varianzas = None
    
    return {
        'tipo': 'grupos_continua',
        'estadisticas_grupos': estadisticas_grupos,
        'anova': {
            'f_statistico': float(f_stat),
            'p_valor': float(p_valor_anova),
            'significativo': p_valor_anova < 0.05
        },
        'homogeneidad_varianzas': {
            'levene_statistico': levene_stat,
            'levene_p_valor': levene_p,
            'homogeneas': homogeneidad_varianzas
        }
    }

def interpretar_analisis_bivariado_cs(resultado: dict) -> dict:
    """
    Interpreta los resultados del análisis bivariado.
    """
    interpretacion = {}
    
    if 'correlacion_continua' in resultado['analisis']:
        analisis = resultado['analisis']
        
        # Usar Pearson como referencia principal
        pearson = analisis['pearson']
        if pearson['significativo']:
            interpretacion['conclusion'] = f"Existe una correlación {pearson['fuerza'].lower()} y significativa entre las variables"
            if pearson['coeficiente'] > 0:
                interpretacion['direccion'] = "La relación es positiva"
            else:
                interpretacion['direccion'] = "La relación es negativa"
        else:
            interpretacion['conclusion'] = "No hay evidencia de correlación significativa entre las variables"
    
    elif 'contingencia_categorica' in resultado['analisis']:
        analisis = resultado['analisis']
        
        if analisis['chi_cuadrado']['significativo']:
            interpretacion['conclusion'] = "Existe una relación significativa entre las variables categóricas"
            
            # Interpretar Cramer's V
            if analisis['cramer_v'] < 0.1:
                interpretacion['fuerza'] = "Efecto muy pequeño"
            elif analisis['cramer_v'] < 0.3:
                interpretacion['fuerza'] = "Efecto pequeño"
            elif analisis['cramer_v'] < 0.5:
                interpretacion['fuerza'] = "Efecto moderado"
            else:
                interpretacion['fuerza'] = "Efecto grande"
        else:
            interpretacion['conclusion'] = "No hay evidencia de relación significativa entre las variables"
    
    elif 'grupos_continua' in resultado['analisis']:
        analisis = resultado['analisis']
        
        if analisis['anova']['significativo']:
            interpretacion['conclusion'] = "Existen diferencias significativas entre los grupos"
        else:
            interpretacion['conclusion'] = "No hay evidencia de diferencias significativas entre los grupos"
    
    return interpretacion

# ============================================================================
# ANÁLISIS MULTIVARIADOS ESPECIALIZADOS
# ============================================================================

def analisis_regresion_multiple_cs(df: pd.DataFrame, variable_dependiente: str, variables_independientes: list) -> dict:
    """
    Análisis de regresión múltiple para ciencias sociales.
    """
    # Preparar datos
    variables = [variable_dependiente] + variables_independientes
    df_limpio = df[variables].dropna()
    
    if len(df_limpio) < len(variables_independientes) + 1:
        return {'error': 'Insuficientes observaciones para el análisis'}
    
    X = df_limpio[variables_independientes]
    y = df_limpio[variable_dependiente]
    
    # Estandarizar variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Regresión
    modelo = LinearRegression()
    modelo.fit(X_scaled, y)
    
    # Predicciones
    y_pred = modelo.predict(X_scaled)
    
    # Estadísticas del modelo
    r2 = modelo.score(X_scaled, y)
    r2_ajustado = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(variables_independientes) - 1)
    
    # Coeficientes
    coeficientes = {}
    for i, var in enumerate(variables_independientes):
        coeficientes[var] = {
            'coeficiente': float(modelo.coef_[i]),
            'coeficiente_estandarizado': float(modelo.coef_[i] * X[var].std() / y.std())
        }
    
    # Análisis de residuos
    residuos = y - y_pred
    
    return {
        'tipo': 'regresion_multiple',
        'n_observaciones': len(df_limpio),
        'n_variables': len(variables_independientes),
        'r_cuadrado': float(r2),
        'r_cuadrado_ajustado': float(r2_ajustado),
        'intercepto': float(modelo.intercept_),
        'coeficientes': coeficientes,
        'residuos': {
            'media': float(residuos.mean()),
            'desv_estandar': float(residuos.std()),
            'minimo': float(residuos.min()),
            'maximo': float(residuos.max())
        }
    }

def analisis_clusters_cs(df: pd.DataFrame, variables: list, n_clusters: int = 3) -> dict:
    """
    Análisis de conglomerados (clustering) para ciencias sociales.
    """
    # Preparar datos
    df_limpio = df[variables].dropna()
    
    if len(df_limpio) < n_clusters:
        return {'error': 'Insuficientes observaciones para el clustering'}
    
    # Estandarizar variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_limpio[variables])
    
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Agregar clusters al DataFrame
    df_limpio['cluster'] = clusters
    
    # Estadísticas por cluster
    estadisticas_clusters = {}
    for i in range(n_clusters):
        cluster_data = df_limpio[df_limpio['cluster'] == i]
        estadisticas_clusters[f'cluster_{i}'] = {
            'n': len(cluster_data),
            'porcentaje': len(cluster_data) / len(df_limpio) * 100,
            'centroide': kmeans.cluster_centers_[i].tolist()
        }
        
        # Medias por variable en el cluster
        for var in variables:
            estadisticas_clusters[f'cluster_{i}'][f'media_{var}'] = float(cluster_data[var].mean())
    
    # Inercia (medida de calidad del clustering)
    inercia = kmeans.inertia_
    
    return {
        'tipo': 'clustering',
        'n_clusters': n_clusters,
        'n_observaciones': len(df_limpio),
        'inercia': float(inercia),
        'estadisticas_clusters': estadisticas_clusters,
        'variables_utilizadas': variables
    }

# ============================================================================
# ÍNDICES Y MEDIDAS ESPECIALIZADAS
# ============================================================================

def calcular_indice_gini(valores: pd.Series) -> float:
    """
    Calcula el coeficiente de Gini para medir desigualdad.
    """
    valores = valores.dropna()
    if len(valores) == 0:
        return 0.0
    
    # Ordenar valores
    valores_ordenados = np.sort(valores)
    n = len(valores_ordenados)
    
    # Calcular Gini
    numerador = np.sum((2 * np.arange(1, n + 1) - n - 1) * valores_ordenados)
    denominador = n * np.sum(valores_ordenados)
    
    if denominador == 0:
        return 0.0
    
    return float(numerador / denominador)

def calcular_indice_desarrollo_humano(indicadores: dict) -> float:
    """
    Calcula un índice de desarrollo humano simplificado.
    """
    # Normalizar indicadores (asumiendo valores entre 0 y 1)
    vida = indicadores.get('esperanza_vida', 0)
    educacion = indicadores.get('educacion', 0)
    ingresos = indicadores.get('ingresos', 0)
    
    # Media geométrica
    idh = (vida * educacion * ingresos) ** (1/3)
    return float(idh)

def calcular_indice_calidad_vida(df: pd.DataFrame, variables: list, pesos: list = None) -> pd.Series:
    """
    Calcula un índice de calidad de vida compuesto.
    """
    if pesos is None:
        pesos = [1/len(variables)] * len(variables)
    
    # Estandarizar variables
    df_estandarizado = df[variables].copy()
    for var in variables:
        media = df_estandarizado[var].mean()
        std = df_estandarizado[var].std()
        if std != 0:
            df_estandarizado[var] = (df_estandarizado[var] - media) / std
    
    # Calcular índice ponderado
    indice = pd.Series(0.0, index=df.index)
    for i, var in enumerate(variables):
        indice += df_estandarizado[var] * pesos[i]
    
    return indice

def calcular_indice_gini_simple(df: pd.DataFrame, columna: str) -> float:
    """
    Calcula el índice de Gini para una variable específica.
    """
    return calcular_indice_gini(df[columna])

def calcular_indice_calidad_vida_simple(df: pd.DataFrame, columna: str) -> float:
    """
    Calcula un índice de calidad de vida simple para una variable.
    """
    valores = df[columna].dropna()
    if len(valores) == 0:
        return 0.0
    
    # Normalizar entre 0 y 1
    min_val = valores.min()
    max_val = valores.max()
    if max_val == min_val:
        return 0.5
    
    valores_norm = (valores - min_val) / (max_val - min_val)
    return float(valores_norm.mean())

# ============================================================================
# VALIDACIÓN Y SUPUESTOS
# ============================================================================

def validar_supuestos_regresion(df: pd.DataFrame, variable_dependiente: str, variables_independientes: list) -> dict:
    """
    Valida los supuestos de la regresión múltiple.
    """
    variables = [variable_dependiente] + variables_independientes
    df_limpio = df[variables].dropna()
    
    X = df_limpio[variables_independientes]
    y = df_limpio[variable_dependiente]
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Regresión
    modelo = LinearRegression()
    modelo.fit(X_scaled, y)
    y_pred = modelo.predict(X_scaled)
    residuos = y - y_pred
    
    # 1. Normalidad de residuos
    shapiro_stat, shapiro_p = shapiro(residuos)
    
    # 2. Homocedasticidad (test de Levene)
    # Dividir residuos en grupos
    n_grupos = min(3, len(residuos) // 10)
    grupos_residuos = np.array_split(residuos, n_grupos)
    try:
        levene_stat, levene_p = levene(*grupos_residuos)
    except:
        levene_stat, levene_p = None, None
    
    # 3. Independencia (Durbin-Watson)
    dw_stat = np.sum(np.diff(residuos) ** 2) / np.sum(residuos ** 2)
    
    # 4. Multicolinealidad (VIF simplificado)
    vif_values = {}
    for i, var in enumerate(variables_independientes):
        # Regresión de cada variable independiente contra las demás
        otras_vars = [v for v in variables_independientes if v != var]
        if otras_vars:
            X_otras = X_scaled[:, [variables_independientes.index(v) for v in otras_vars]]
            y_var = X_scaled[:, i]
            modelo_vif = LinearRegression()
            modelo_vif.fit(X_otras, y_var)
            r2_var = modelo_vif.score(X_otras, y_var)
            vif = 1 / (1 - r2_var) if r2_var < 1 else float('inf')
        else:
            vif = 1.0
        vif_values[var] = float(vif)
    
    return {
        'normalidad_residuos': {
            'shapiro_statistico': float(shapiro_stat),
            'p_valor': float(shapiro_p),
            'cumple_supuesto': shapiro_p > 0.05
        },
        'homocedasticidad': {
            'levene_statistico': levene_stat,
            'p_valor': levene_p,
            'cumple_supuesto': levene_p > 0.05 if levene_p is not None else None
        },
        'independencia': {
            'durbin_watson': float(dw_stat),
            'cumple_supuesto': 1.5 < dw_stat < 2.5
        },
        'multicolinealidad': {
            'vif_values': vif_values,
            'problema_multicolinealidad': any(vif > 10 for vif in vif_values.values())
        }
    }

# ============================================================================
# MANEJO DE VALORES PERDIDOS
# ============================================================================

def analizar_valores_perdidos(df: pd.DataFrame) -> dict:
    """
    Analiza patrones de valores perdidos en el dataset.
    """
    # Conteo de valores perdidos por variable
    valores_perdidos = df.isnull().sum()
    porcentajes_perdidos = (valores_perdidos / len(df)) * 100
    
    # Patrones de valores perdidos
    patrones = {}
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            # Variables que tienen valores cuando esta variable está perdida
            mascara_perdidos = df[col].isnull()
            otras_vars = [c for c in df.columns if c != col]
            
            patrones[col] = {}
            for otra_var in otras_vars:
                if df[otra_var].dtype in ['int64', 'float64']:
                    # Para variables numéricas, calcular media
                    media_con_perdidos = df.loc[mascara_perdidos, otra_var].mean()
                    media_sin_perdidos = df.loc[~mascara_perdidos, otra_var].mean()
                    patrones[col][otra_var] = {
                        'tipo': 'numerica',
                        'media_con_perdidos': media_con_perdidos,
                        'media_sin_perdidos': media_sin_perdidos,
                        'diferencia': media_con_perdidos - media_sin_perdidos if pd.notna(media_con_perdidos) and pd.notna(media_sin_perdidos) else None
                    }
                else:
                    # Para variables categóricas, calcular frecuencias
                    frecuencias_con_perdidos = df.loc[mascara_perdidos, otra_var].value_counts(normalize=True)
                    frecuencias_sin_perdidos = df.loc[~mascara_perdidos, otra_var].value_counts(normalize=True)
                    patrones[col][otra_var] = {
                        'tipo': 'categorica',
                        'frecuencias_con_perdidos': frecuencias_con_perdidos.to_dict(),
                        'frecuencias_sin_perdidos': frecuencias_sin_perdidos.to_dict()
                    }
    
    return {
        'conteo_por_variable': valores_perdidos.to_dict(),
        'porcentajes_por_variable': porcentajes_perdidos.to_dict(),
        'total_valores_perdidos': int(valores_perdidos.sum()),
        'porcentaje_total_perdidos': float(valores_perdidos.sum() / (len(df) * len(df.columns)) * 100),
        'patrones_valores_perdidos': patrones
    }

def sugerir_imputacion(df: pd.DataFrame, columna: str) -> dict:
    """
    Sugiere método de imputación basado en el tipo de variable.
    """
    clasificacion = clasificar_variable(df, columna)
    valores_perdidos = df[columna].isnull().sum()
    
    sugerencias = {
        'columna': columna,
        'n_valores_perdidos': int(valores_perdidos),
        'porcentaje_perdidos': float(valores_perdidos / len(df) * 100),
        'metodos_recomendados': []
    }
    
    if clasificacion['es_continua']:
        sugerencias['metodos_recomendados'].extend([
            'Media',
            'Mediana',
            'Interpolación lineal',
            'Regresión múltiple'
        ])
    elif clasificacion['es_categorica']:
        sugerencias['metodos_recomendados'].extend([
            'Moda',
            'Categoría "No especificado"',
            'Imputación por regresión logística'
        ])
    
    # Ajustar según porcentaje de valores perdidos
    if sugerencias['porcentaje_perdidos'] > 50:
        sugerencias['advertencia'] = "Alto porcentaje de valores perdidos. Considerar eliminar la variable."
    elif sugerencias['porcentaje_perdidos'] > 20:
        sugerencias['advertencia'] = "Porcentaje moderado de valores perdidos. Usar métodos de imputación con precaución."
    
    return sugerencias 