"""
Subsistema de I/O
================

Módulo responsable de la detección, carga y normalización de archivos de datos.
Soporta múltiples formatos: CSV, Excel, SPSS (.sav), Stata (.dta).
"""

import pandas as pd
import numpy as np
import chardet
import os
import re
from typing import Dict, Any, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar logging
from .logging import log_action

def detect_format(file_name: str, file_bytes: bytes) -> str:
    """
    Detecta el formato del archivo basado en su nombre y contenido.
    
    Args:
        file_name: Nombre del archivo
        file_bytes: Contenido del archivo en bytes
        
    Returns:
        Formato detectado: 'csv', 'excel', 'sav', 'dta'
        
    Raises:
        ValueError: Si no se puede detectar el formato
    """
    file_name_lower = file_name.lower()
    
    # Detección por extensión
    if file_name_lower.endswith('.csv'):
        return 'csv'
    elif file_name_lower.endswith(('.xlsx', '.xls')):
        return 'excel'
    elif file_name_lower.endswith('.sav'):
        return 'sav'
    elif file_name_lower.endswith('.dta'):
        return 'dta'
    
    # Detección por contenido (magic bytes)
    if file_bytes.startswith(b'PK\x03\x04'):  # ZIP header (Excel)
        return 'excel'
    elif file_bytes.startswith(b'$FL2'):  # SPSS
        return 'sav'
    elif file_bytes.startswith(b'<stata_dta>'):  # Stata
        return 'dta'
    elif b',' in file_bytes[:1000] or b';' in file_bytes[:1000]:  # CSV
        return 'csv'
    
    raise ValueError(f"No se pudo detectar el formato del archivo: {file_name}")

def load_csv(file_path: str, file_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carga un archivo CSV con detección automática de encoding y separador.
    
    Args:
        file_path: Ruta del archivo
        file_bytes: Contenido del archivo en bytes
        
    Returns:
        Tuple: (DataFrame, metadata)
    """
    # Detectar encoding
    encoding_result = chardet.detect(file_bytes)
    encoding = encoding_result['encoding'] if encoding_result['confidence'] > 0.7 else 'utf-8'
    
    # Detectar separador
    sample_text = file_bytes[:10000].decode(encoding, errors='ignore')
    separators = [',', ';', '\t', '|']
    separator_scores = {}
    
    for sep in separators:
        lines = sample_text.split('\n')[:10]
        if len(lines) > 1:
            columns = [line.split(sep) for line in lines if line.strip()]
            if columns:
                avg_columns = np.mean([len(col) for col in columns])
                std_columns = np.std([len(col) for col in columns])
                separator_scores[sep] = avg_columns / (std_columns + 1)  # Penalizar variabilidad
    
    separator = max(separator_scores.items(), key=lambda x: x[1])[0] if separator_scores else ','
    
    # Cargar DataFrame
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=separator,
        engine='python',
        on_bad_lines='skip'
    )
    
    # Metadata
    metadata = {
        'format': 'csv',
        'encoding': encoding,
        'encoding_confidence': encoding_result.get('confidence', 0),
        'separator': separator,
        'original_rows': len(df),
        'original_columns': len(df.columns),
        'file_size': len(file_bytes)
    }
    
    return df, metadata

def load_excel(file_path: str, file_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carga un archivo Excel con manejo de múltiples hojas y metadatos.
    
    Args:
        file_path: Ruta del archivo
        file_bytes: Contenido del archivo en bytes
        
    Returns:
        Tuple: (DataFrame, metadata)
    """
    # Leer todas las hojas
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    # Si hay múltiples hojas, usar la primera no vacía
    df = None
    selected_sheet = None
    
    for sheet_name in sheet_names:
        temp_df = pd.read_excel(file_path, sheet_name=sheet_name)
        if not temp_df.empty and len(temp_df.columns) > 1:
            df = temp_df
            selected_sheet = sheet_name
            break
    
    if df is None:
        # Si todas las hojas están vacías, usar la primera
        df = pd.read_excel(file_path, sheet_name=sheet_names[0])
        selected_sheet = sheet_names[0]
    
    # Metadata
    metadata = {
        'format': 'excel',
        'sheet_names': sheet_names,
        'selected_sheet': selected_sheet,
        'original_rows': len(df),
        'original_columns': len(df.columns),
        'file_size': len(file_bytes)
    }
    
    return df, metadata

def load_sav(file_path: str, file_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carga un archivo SPSS (.sav) con etiquetas de variables y valores.
    
    Args:
        file_path: Ruta del archivo
        file_bytes: Contenido del archivo en bytes
        
    Returns:
        Tuple: (DataFrame, metadata)
    """
    try:
        import pyreadstat
        
        # Cargar datos con metadatos
        df, meta = pyreadstat.read_sav(file_path)
        
        # Extraer etiquetas
        variable_labels = meta.variable_labels if hasattr(meta, 'variable_labels') else {}
        value_labels = meta.value_labels if hasattr(meta, 'value_labels') else {}
        
        # Metadata
        metadata = {
            'format': 'sav',
            'variable_labels': variable_labels,
            'value_labels': value_labels,
            'original_rows': len(df),
            'original_columns': len(df.columns),
            'file_size': len(file_bytes),
            'spss_version': getattr(meta, 'version', 'unknown')
        }
        
        return df, metadata
        
    except ImportError:
        raise ImportError("pyreadstat no está instalado. Instálalo con: pip install pyreadstat")
    except Exception as e:
        raise ValueError(f"Error al cargar archivo SPSS: {str(e)}")

def load_dta(file_path: str, file_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carga un archivo Stata (.dta) con etiquetas de variables y valores.
    
    Args:
        file_path: Ruta del archivo
        file_bytes: Contenido del archivo en bytes
        
    Returns:
        Tuple: (DataFrame, metadata)
    """
    try:
        import pyreadstat
        
        # Cargar datos con metadatos
        df, meta = pyreadstat.read_dta(file_path)
        
        # Extraer etiquetas
        variable_labels = meta.variable_labels if hasattr(meta, 'variable_labels') else {}
        value_labels = meta.value_labels if hasattr(meta, 'value_labels') else {}
        
        # Metadata
        metadata = {
            'format': 'dta',
            'variable_labels': variable_labels,
            'value_labels': value_labels,
            'original_rows': len(df),
            'original_columns': len(df.columns),
            'file_size': len(file_bytes),
            'stata_version': getattr(meta, 'version', 'unknown')
        }
        
        return df, metadata
        
    except ImportError:
        raise ImportError("pyreadstat no está instalado. Instálalo con: pip install pyreadstat")
    except Exception as e:
        raise ValueError(f"Error al cargar archivo Stata: {str(e)}")

def normalize_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normaliza el DataFrame: estandariza nombres, limpia datos, etc.
    
    Args:
        df: DataFrame a normalizar
        metadata: Metadatos del archivo
        
    Returns:
        Tuple: (DataFrame normalizado, metadata actualizado)
    """
    df_normalized = df.copy()
    
    # 1. Estandarizar nombres de columnas (snake_case)
    df_normalized.columns = [
        re.sub(r'[^a-zA-Z0-9_]', '_', col.lower().strip())
        for col in df_normalized.columns
    ]
    
    # 2. Recortar espacios en valores de texto
    for col in df_normalized.select_dtypes(include=['object']).columns:
        df_normalized[col] = df_normalized[col].astype(str).str.strip()
    
    # 3. Reemplazar valores de datos faltantes
    missing_values = ['NA', 'Missing', 'N/A', 'n/a', 'NULL', 'null', '', 'nan']
    for col in df_normalized.columns:
        df_normalized[col] = df_normalized[col].replace(missing_values, np.nan)
    
    # 4. Normalizar separadores decimales
    for col in df_normalized.select_dtypes(include=['object']).columns:
        # Intentar convertir a numérico si es posible
        try:
            # Reemplazar comas por puntos en números
            temp_col = df_normalized[col].astype(str).str.replace(',', '.')
            pd.to_numeric(temp_col, errors='raise')
            df_normalized[col] = pd.to_numeric(temp_col, errors='coerce')
        except (ValueError, TypeError):
            pass
    
    # 5. Eliminar columnas 100% vacías
    empty_columns = df_normalized.columns[df_normalized.isnull().all()].tolist()
    df_normalized = df_normalized.drop(columns=empty_columns)
    
    # 6. Convertir fechas
    for col in df_normalized.select_dtypes(include=['object']).columns:
        try:
            df_normalized[col] = pd.to_datetime(df_normalized[col], errors='coerce')
        except (ValueError, TypeError):
            pass
    
    # Actualizar metadata
    metadata.update({
        'normalized_rows': len(df_normalized),
        'normalized_columns': len(df_normalized.columns),
        'empty_columns_removed': empty_columns,
        'normalization_applied': True
    })
    
    return df_normalized, metadata

def load_file(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Función principal para cargar archivos. Detecta formato y aplica normalización.
    
    Args:
        uploaded_file: Archivo subido (Streamlit UploadedFile o similar)
        
    Returns:
        Tuple: (DataFrame normalizado, metadata completo)
        
    Raises:
        ValueError: Si hay error en la carga o formato no soportado
    """
    start_time = datetime.now()
    
    try:
        # Leer contenido del archivo
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        
        # Métricas antes de la carga
        before_metrics = {
            'file_name': file_name,
            'file_size': len(file_bytes)
        }
        
        # Detectar formato
        file_format = detect_format(file_name, file_bytes)
        
        # Cargar según formato
        if file_format == 'csv':
            df, metadata = load_csv(uploaded_file, file_bytes)
        elif file_format == 'excel':
            df, metadata = load_excel(uploaded_file, file_bytes)
        elif file_format == 'sav':
            df, metadata = load_sav(uploaded_file, file_bytes)
        elif file_format == 'dta':
            df, metadata = load_dta(uploaded_file, file_bytes)
        else:
            raise ValueError(f"Formato no soportado: {file_format}")
        
        # Normalizar DataFrame
        df_normalized, metadata = normalize_dataframe(df, metadata)
        
        # Calcular tiempo de ejecución
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Métricas después de la carga
        after_metrics = {
            'rows': len(df_normalized),
            'columns': len(df_normalized.columns),
            'data_types': df_normalized.dtypes.to_dict(),
            'missing_values': df_normalized.isnull().sum().to_dict()
        }
        
        # Registrar acción
        log_action(
            function='load_file',
            step='data_load',
            parameters={'file_name': file_name, 'format': file_format},
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            status='success',
            message=f"Archivo {file_name} cargado exitosamente ({file_format})",
            execution_time=execution_time
        )
        
        return df_normalized, metadata
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Registrar error
        log_action(
            function='load_file',
            step='data_load',
            parameters={'file_name': file_name if 'file_name' in locals() else 'unknown'},
            before_metrics=before_metrics if 'before_metrics' in locals() else {},
            after_metrics={},
            status='error',
            message=f"Error al cargar archivo: {str(e)}",
            execution_time=execution_time,
            error_details=str(e)
        )
        
        raise ValueError(f"Error al cargar archivo: {str(e)}")

def get_file_info(uploaded_file) -> Dict[str, Any]:
    """
    Obtiene información básica del archivo sin cargarlo completamente.
    
    Args:
        uploaded_file: Archivo subido
        
    Returns:
        Diccionario con información del archivo
    """
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name
    
    try:
        file_format = detect_format(file_name, file_bytes)
        
        info = {
            'file_name': file_name,
            'file_size': len(file_bytes),
            'format': file_format,
            'size_mb': len(file_bytes) / (1024 * 1024)
        }
        
        # Información adicional según formato
        if file_format == 'excel':
            excel_file = pd.ExcelFile(uploaded_file)
            info['sheets'] = excel_file.sheet_names
        elif file_format == 'csv':
            # Detectar encoding y separador
            encoding_result = chardet.detect(file_bytes)
            info['encoding'] = encoding_result['encoding']
            info['encoding_confidence'] = encoding_result['confidence']
        
        return info
        
    except Exception as e:
        return {
            'file_name': file_name,
            'file_size': len(file_bytes),
            'error': str(e)
        }

class DataLoader:
    """
    Clase principal para cargar y normalizar archivos de datos.
    Encapsula todas las funciones de carga y detección de formato.
    """
    
    def __init__(self):
        """Inicializar el DataLoader."""
        self.detected_format = None
        self.file_metadata = {}
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        Carga un archivo de datos y detecta su formato automáticamente.
        
        Args:
            file_path: Ruta del archivo a cargar
            
        Returns:
            DataFrame cargado y normalizado
            
        Raises:
            ValueError: Si no se puede cargar el archivo
        """
        try:
            # Leer archivo en bytes para detección de formato
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Detectar formato
            file_name = os.path.basename(file_path)
            self.detected_format = detect_format(file_name, file_bytes)
            
            # Cargar según el formato
            if self.detected_format == 'csv':
                df, metadata = load_csv(file_path, file_bytes)
            elif self.detected_format == 'excel':
                df, metadata = load_excel(file_path, file_bytes)
            elif self.detected_format == 'sav':
                df, metadata = load_sav(file_path, file_bytes)
            elif self.detected_format == 'dta':
                df, metadata = load_dta(file_path, file_bytes)
            else:
                raise ValueError(f"Formato no soportado: {self.detected_format}")
            
            # Guardar metadata
            self.file_metadata = metadata
            
            # Normalizar DataFrame
            df, normalized_metadata = normalize_dataframe(df, metadata)
            self.file_metadata.update(normalized_metadata)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error al cargar archivo {file_path}: {str(e)}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Obtener metadata del archivo cargado."""
        return self.file_metadata.copy()
    
    def get_format(self) -> str:
        """Obtener el formato detectado del archivo."""
        return self.detected_format 