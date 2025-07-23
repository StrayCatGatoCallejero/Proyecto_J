"""
Módulo de I/O Robusto y Seguro - Proyecto J
===========================================

Responsabilidades:
- Carga segura de archivos con validación de tipos
- Detección automática de encoding
- Manejo robusto de errores de archivo
- Validación de seguridad de archivos
- Gestión de memoria para archivos grandes
- Logging detallado de operaciones I/O
"""

import os
import sys
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import warnings
import logging
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from chardet import detect as chardet_detect

# Importar módulos del sistema
from .logging import log_action
from .error_reporter import report_dataframe_error
from .validation_decorators import validate_io, create_dataframe_schema

# Configurar logging
logger = logging.getLogger(__name__)

# Constantes de seguridad
MAX_FILE_SIZE_MB = 100  # Tamaño máximo de archivo en MB
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.sav', '.dta', '.json', '.parquet'}
DANGEROUS_EXTENSIONS = {'.exe', '.bat', '.cmd', '.sh', '.py', '.js', '.html', '.htm'}
MAX_COLUMNS = 1000  # Máximo número de columnas
MAX_ROWS_PREVIEW = 10000  # Máximo filas para preview

# Configuración de encoding
ENCODINGS_TO_TRY = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']


class FileSecurityError(Exception):
    """Excepción para errores de seguridad de archivos"""
    pass


class FileValidationError(Exception):
    """Excepción para errores de validación de archivos"""
    pass


class MemoryLimitError(Exception):
    """Excepción para errores de límite de memoria"""
    pass


@dataclass
class FileMetadata:
    """Metadatos de archivo para auditoría y validación"""
    filename: str
    file_size: int
    file_hash: str
    extension: str
    encoding: Optional[str] = None
    loaded_at: datetime = field(default_factory=datetime.now)
    columns_count: int = 0
    rows_count: int = 0
    memory_usage_mb: float = 0.0
    load_time_seconds: float = 0.0


def validate_file_security(file_path: Union[str, Path]) -> None:
    """
    Valida la seguridad de un archivo antes de procesarlo.
    
    Args:
        file_path: Ruta al archivo a validar
        
    Raises:
        FileSecurityError: Si el archivo no es seguro
    """
    file_path = Path(file_path)
    
    # Verificar extensión peligrosa
    if file_path.suffix.lower() in DANGEROUS_EXTENSIONS:
        raise FileSecurityError(
            f"Extensión de archivo peligrosa: {file_path.suffix}"
        )
    
    # Verificar que el archivo existe
    if not file_path.exists():
        raise FileSecurityError(f"El archivo no existe: {file_path}")
    
    # Verificar tamaño del archivo
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise FileSecurityError(
            f"Archivo demasiado grande: {file_size_mb:.2f}MB > {MAX_FILE_SIZE_MB}MB"
        )
    
    # Verificar que es un archivo regular
    if not file_path.is_file():
        raise FileSecurityError(f"No es un archivo regular: {file_path}")
    
    logger.info(f"✅ Validación de seguridad exitosa para: {file_path}")


def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calcula el hash SHA-256 de un archivo para verificación de integridad.
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Hash SHA-256 del archivo
    """
    file_path = Path(file_path)
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def detect_encoding(file_path: Union[str, Path]) -> str:
    """
    Detecta el encoding de un archivo de forma robusta.
        
        Args:
        file_path: Ruta al archivo
            
        Returns:
        Encoding detectado
    """
    file_path = Path(file_path)
    
    try:
        # Leer una muestra del archivo para detectar encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Leer primeros 10KB
        
        if not raw_data:
            return 'utf-8'
        
        # Usar chardet para detectar encoding
        result = chardet_detect(raw_data)
        detected_encoding = result['encoding']
        confidence = result['confidence']
        
        logger.info(f"Encoding detectado: {detected_encoding} (confianza: {confidence:.2f})")
        
        # Si la confianza es baja, usar utf-8 por defecto
        if confidence < 0.7:
            logger.warning(f"Baja confianza en encoding detectado: {confidence:.2f}")
            return 'utf-8'
        
        return detected_encoding or 'utf-8'
        
    except Exception as e:
        logger.warning(f"Error detectando encoding: {e}")
        return 'utf-8'


@validate_io
def load_csv_safe(
    file_path: Union[str, Path], 
    encoding: Optional[str] = None,
    chunk_size: Optional[int] = None
) -> Tuple[pd.DataFrame, FileMetadata]:
    """
    Carga un archivo CSV de forma segura con manejo robusto de encoding.
    
    Args:
        file_path: Ruta al archivo CSV
        encoding: Encoding específico (opcional)
        chunk_size: Tamaño de chunk para archivos grandes (opcional)
        
    Returns:
        Tupla con DataFrame y metadatos del archivo
        
    Raises:
        FileSecurityError: Si el archivo no es seguro
        FileValidationError: Si hay errores de validación
    """
    start_time = datetime.now()
    file_path = Path(file_path)
    
    # Validar seguridad del archivo
    validate_file_security(file_path)
    
    # Calcular hash del archivo
    file_hash = calculate_file_hash(file_path)
    
    # Detectar encoding si no se especifica
    if encoding is None:
        encoding = detect_encoding(file_path)
    
    # Intentar diferentes encodings si el detectado falla
    encodings_to_try = [encoding] + [enc for enc in ENCODINGS_TO_TRY if enc != encoding]
    
    df = None
    used_encoding = None
    
    for enc in encodings_to_try:
        try:
            if chunk_size:
                # Cargar en chunks para archivos grandes
                chunks = []
                for chunk in pd.read_csv(file_path, encoding=enc, chunksize=chunk_size):
                    chunks.append(chunk)
                    if len(chunks) * chunk_size > MAX_ROWS_PREVIEW:
                        logger.warning(f"Archivo muy grande, cargando solo preview")
                        break
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path, encoding=enc)
            
            used_encoding = enc
            logger.info(f"✅ CSV cargado exitosamente con encoding: {enc}")
            break
            
        except UnicodeDecodeError:
            logger.debug(f"Encoding {enc} falló, intentando siguiente...")
            continue
        except Exception as e:
            logger.warning(f"Error con encoding {enc}: {e}")
            continue
    
    if df is None:
        raise FileValidationError(f"No se pudo cargar el archivo CSV: {file_path}")
    
    # Validar DataFrame
    validate_dataframe(df, file_path)
    
    # Calcular metadatos
    load_time = (datetime.now() - start_time).total_seconds()
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    metadata = FileMetadata(
        filename=file_path.name,
        file_size=file_path.stat().st_size,
        file_hash=file_hash,
        extension=file_path.suffix.lower(),
        encoding=used_encoding,
        columns_count=len(df.columns),
        rows_count=len(df),
        memory_usage_mb=memory_usage,
        load_time_seconds=load_time
    )
    
    logger.info(f"📊 Archivo cargado: {metadata.rows_count:,} filas × {metadata.columns_count} columnas")
    logger.info(f"💾 Uso de memoria: {memory_usage:.2f}MB")
    
    return df, metadata


@validate_io
def load_excel_safe(
    file_path: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0,
    engine: Optional[str] = None
) -> Tuple[pd.DataFrame, FileMetadata]:
    """
    Carga un archivo Excel de forma segura.
    
    Args:
        file_path: Ruta al archivo Excel
        sheet_name: Nombre o índice de la hoja
        engine: Motor de Excel (openpyxl, xlrd)
        
    Returns:
        Tupla con DataFrame y metadatos del archivo
    """
    start_time = datetime.now()
    file_path = Path(file_path)
    
    # Validar seguridad del archivo
    validate_file_security(file_path)
    
    # Calcular hash del archivo
    file_hash = calculate_file_hash(file_path)
    
    # Determinar engine basado en extensión
    if engine is None:
        if file_path.suffix.lower() == '.xlsx':
            engine = 'openpyxl'
        elif file_path.suffix.lower() == '.xls':
            engine = 'xlrd'
        else:
            engine = 'openpyxl'  # Por defecto
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)
        logger.info(f"✅ Excel cargado exitosamente con engine: {engine}")
        
    except Exception as e:
        # Intentar con engine alternativo
        alternative_engine = 'openpyxl' if engine == 'xlrd' else 'xlrd'
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine=alternative_engine)
            logger.info(f"✅ Excel cargado con engine alternativo: {alternative_engine}")
        except Exception as e2:
            raise FileValidationError(f"No se pudo cargar el archivo Excel: {e2}")
    
    # Validar DataFrame
    validate_dataframe(df, file_path)
    
    # Calcular metadatos
    load_time = (datetime.now() - start_time).total_seconds()
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    metadata = FileMetadata(
        filename=file_path.name,
        file_size=file_path.stat().st_size,
        file_hash=file_hash,
        extension=file_path.suffix.lower(),
        columns_count=len(df.columns),
        rows_count=len(df),
        memory_usage_mb=memory_usage,
        load_time_seconds=load_time
    )
    
    return df, metadata


@validate_io
def load_spss_safe(file_path: Union[str, Path]) -> Tuple[pd.DataFrame, FileMetadata]:
    """
    Carga un archivo SPSS (.sav) de forma segura.
    
    Args:
        file_path: Ruta al archivo SPSS
        
    Returns:
        Tupla con DataFrame y metadatos del archivo
    """
    start_time = datetime.now()
    file_path = Path(file_path)
    
    # Validar seguridad del archivo
    validate_file_security(file_path)
    
    # Verificar que pyreadstat esté disponible
    try:
        import pyreadstat
    except ImportError:
        raise FileValidationError(
            "pyreadstat no está instalado. Instala con: pip install pyreadstat"
        )
    
    # Calcular hash del archivo
    file_hash = calculate_file_hash(file_path)
    
    try:
        df, meta = pyreadstat.read_sav(file_path)
        logger.info(f"✅ Archivo SPSS cargado exitosamente")
        
    except Exception as e:
        raise FileValidationError(f"No se pudo cargar el archivo SPSS: {e}")
    
    # Validar DataFrame
    validate_dataframe(df, file_path)
    
    # Calcular metadatos
    load_time = (datetime.now() - start_time).total_seconds()
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    metadata = FileMetadata(
        filename=file_path.name,
        file_size=file_path.stat().st_size,
        file_hash=file_hash,
        extension=file_path.suffix.lower(),
        columns_count=len(df.columns),
        rows_count=len(df),
        memory_usage_mb=memory_usage,
        load_time_seconds=load_time
    )
    
    return df, metadata


def validate_dataframe(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """
    Valida un DataFrame cargado para asegurar que es seguro y válido.
    
    Args:
        df: DataFrame a validar
        file_path: Ruta del archivo original
        
    Raises:
        FileValidationError: Si el DataFrame no es válido
        MemoryLimitError: Si el DataFrame es demasiado grande
    """
    if df is None or df.empty:
        raise FileValidationError("El DataFrame está vacío")
    
    if len(df.columns) > MAX_COLUMNS:
        raise FileValidationError(
            f"Demasiadas columnas: {len(df.columns)} > {MAX_COLUMNS}"
        )
    
    # Verificar uso de memoria
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    if memory_usage_mb > MAX_FILE_SIZE_MB:
        raise MemoryLimitError(
            f"DataFrame demasiado grande en memoria: {memory_usage_mb:.2f}MB"
        )
    
    # Verificar tipos de datos sospechosos
    suspicious_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Verificar si hay código potencialmente peligroso
            sample_values = df[col].dropna().astype(str).str[:100]
            if any('eval(' in val or 'exec(' in val for val in sample_values):
                suspicious_columns.append(col)
    
    if suspicious_columns:
        logger.warning(f"Columnas sospechosas detectadas: {suspicious_columns}")
    
    logger.info(f"✅ DataFrame validado: {len(df)} filas × {len(df.columns)} columnas")


def save_dataframe_safe(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    format: str = 'csv',
    **kwargs
) -> FileMetadata:
    """
    Guarda un DataFrame de forma segura con validaciones.
    
    Args:
        df: DataFrame a guardar
        file_path: Ruta de destino
        format: Formato de salida (csv, excel, parquet, json)
        **kwargs: Argumentos adicionales para el formato
        
    Returns:
        Metadatos del archivo guardado
    """
    file_path = Path(file_path)
    
    # Validar DataFrame
    if df is None or df.empty:
        raise FileValidationError("No hay datos para guardar")
    
    # Crear directorio si no existe
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    
    try:
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False, **kwargs)
        elif format.lower() in ['excel', 'xlsx']:
            df.to_excel(file_path, index=False, **kwargs)
        elif format.lower() == 'parquet':
            df.to_parquet(file_path, **kwargs)
        elif format.lower() == 'json':
            df.to_json(file_path, orient='records', **kwargs)
        else:
            raise FileValidationError(f"Formato no soportado: {format}")
        
        logger.info(f"✅ DataFrame guardado exitosamente: {file_path}")
        
    except Exception as e:
        raise FileValidationError(f"Error guardando archivo: {e}")
    
    # Calcular metadatos
    save_time = (datetime.now() - start_time).total_seconds()
    file_hash = calculate_file_hash(file_path)
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    metadata = FileMetadata(
        filename=file_path.name,
        file_size=file_path.stat().st_size,
        file_hash=file_hash,
        extension=file_path.suffix.lower(),
        columns_count=len(df.columns),
        rows_count=len(df),
        memory_usage_mb=memory_usage,
        load_time_seconds=save_time
    )
    
    return metadata


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Obtiene información detallada de un archivo sin cargarlo.
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Diccionario con información del archivo
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"error": "Archivo no encontrado"}
    
    stat = file_path.stat()
        
        info = {
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "is_safe_extension": file_path.suffix.lower() in ALLOWED_EXTENSIONS,
        "is_dangerous_extension": file_path.suffix.lower() in DANGEROUS_EXTENSIONS,
    }
    
    # Detectar encoding si es un archivo de texto
    if file_path.suffix.lower() in ['.csv', '.txt', '.json']:
        try:
            info["encoding"] = detect_encoding(file_path)
        except Exception as e:
            info["encoding_error"] = str(e)
        
        return info
