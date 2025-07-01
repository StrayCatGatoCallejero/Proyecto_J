# tasks.py
from celery import Celery
import pandas as pd
import os
import time
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN DE CELERY
# ============================================================================

# Configurar Celery con Redis como broker
celery_app = Celery("proyecto_j_tasks", broker="redis://localhost:6379/0")

# Configuración adicional para producción
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutos máximo por tarea
    task_soft_time_limit=25 * 60,  # 25 minutos soft limit
)

# ============================================================================
# TAREAS ASÍNCRONAS
# ============================================================================


@celery_app.task(name="procesar_archivo", bind=True)
def procesar_archivo(self, temp_path: str):
    """
    Procesa un archivo CSV en chunks para manejar archivos grandes.

    Args:
        temp_path (str): Ruta al archivo CSV temporal

    Returns:
        dict: Diccionario con información del procesamiento
    """
    try:
        # Actualizar estado de la tarea
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Iniciando procesamiento..."},
        )

        # Asegurar que existe la carpeta de resultados
        resultados_dir = Path("./resultados")
        resultados_dir.mkdir(exist_ok=True)

        # Obtener información del archivo
        file_size = os.path.getsize(temp_path) / (1024 * 1024)  # MB
        base_name = Path(temp_path).stem

        self.update_state(
            state="PROGRESS",
            meta={
                "current": 10,
                "total": 100,
                "status": f"Archivo: {base_name} ({file_size:.1f} MB)",
            },
        )

        # Procesamiento por chunks
        total_filas = 0
        total_columnas = 0
        chunks_procesados = 0
        estadisticas = {}

        # Leer CSV en chunks de 10,000 filas
        chunk_size = 10_000

        # Primera pasada para contar filas y obtener estadísticas básicas
        for chunk in pd.read_csv(temp_path, chunksize=chunk_size):
            total_filas += len(chunk)
            total_columnas = len(chunk.columns)
            chunks_procesados += 1

            # Calcular progreso
            progress = min(10 + (chunks_procesados * 70), 80)  # 10-80%

            # Actualizar estado
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": progress,
                    "total": 100,
                    "status": f"Procesando chunk {chunks_procesados} - {total_filas:,} filas leídas",
                },
            )

            # Simular procesamiento adicional (aquí puedes agregar tu lógica)
            time.sleep(0.1)  # Simular trabajo

        # Segunda pasada para análisis más detallado
        self.update_state(
            state="PROGRESS",
            meta={"current": 85, "total": 100, "status": "Generando estadísticas..."},
        )

        # Leer todo el archivo para estadísticas completas
        df_completo = pd.read_csv(temp_path)

        # Calcular estadísticas básicas
        estadisticas = {
            "total_filas": len(df_completo),
            "total_columnas": len(df_completo.columns),
            "columnas_numericas": len(
                df_completo.select_dtypes(include=["number"]).columns
            ),
            "columnas_categoricas": len(
                df_completo.select_dtypes(include=["object"]).columns
            ),
            "valores_faltantes": df_completo.isnull().sum().sum(),
            "memoria_utilizada_mb": df_completo.memory_usage(deep=True).sum()
            / (1024 * 1024),
        }

        # Generar resumen de columnas
        resumen_columnas = []
        for col in df_completo.columns:
            col_info = {
                "columna": col,
                "tipo": str(df_completo[col].dtype),
                "valores_unicos": df_completo[col].nunique(),
                "valores_faltantes": df_completo[col].isnull().sum(),
            }

            # Estadísticas específicas por tipo
            if df_completo[col].dtype in ["int64", "float64"]:
                col_info.update(
                    {
                        "min": df_completo[col].min(),
                        "max": df_completo[col].max(),
                        "media": df_completo[col].mean(),
                        "mediana": df_completo[col].median(),
                    }
                )
            else:
                col_info.update(
                    {
                        "valor_mas_frecuente": (
                            df_completo[col].mode().iloc[0]
                            if not df_completo[col].mode().empty
                            else None
                        )
                    }
                )

            resumen_columnas.append(col_info)

        # Crear DataFrame con el resumen completo
        df_resumen = pd.DataFrame(
            {"metrica": list(estadisticas.keys()), "valor": list(estadisticas.values())}
        )

        # Guardar resultados
        self.update_state(
            state="PROGRESS",
            meta={"current": 95, "total": 100, "status": "Guardando resultados..."},
        )

        # Guardar resumen general
        out_path_resumen = resultados_dir / f"{base_name}_resumen.parquet"
        df_resumen.to_parquet(out_path_resumen)

        # Guardar resumen de columnas
        df_columnas = pd.DataFrame(resumen_columnas)
        out_path_columnas = resultados_dir / f"{base_name}_columnas.parquet"
        df_columnas.to_parquet(out_path_columnas)

        # Guardar muestra de datos (primeras 1000 filas)
        muestra_path = resultados_dir / f"{base_name}_muestra.parquet"
        df_completo.head(1000).to_parquet(muestra_path)

        # Resultado final
        resultado = {
            "archivo_original": base_name,
            "tamaño_mb": file_size,
            "resumen_path": str(out_path_resumen),
            "columnas_path": str(out_path_columnas),
            "muestra_path": str(muestra_path),
            "estadisticas": estadisticas,
            "tiempo_procesamiento": (
                time.time() - self.request.start_time
                if hasattr(self.request, "start_time")
                else None
            ),
        }

        self.update_state(
            state="SUCCESS",
            meta={
                "current": 100,
                "total": 100,
                "status": "Procesamiento completado",
                "resultado": resultado,
            },
        )

        return resultado

    except Exception as e:
        # En caso de error, actualizar estado
        self.update_state(
            state="FAILURE",
            meta={"current": 0, "total": 100, "status": f"Error: {str(e)}"},
        )
        raise e


@celery_app.task(name="validar_archivo")
def validar_archivo(temp_path: str):
    """
    Valida rápidamente un archivo CSV para verificar que sea legible.

    Args:
        temp_path (str): Ruta al archivo CSV

    Returns:
        dict: Información básica del archivo
    """
    try:
        # Leer solo las primeras filas para validación
        df_muestra = pd.read_csv(temp_path, nrows=100)

        # Información básica
        info = {
            "columnas": list(df_muestra.columns),
            "tipos_columnas": df_muestra.dtypes.to_dict(),
            "filas_muestra": len(df_muestra),
            "es_valido": True,
        }

        return info

    except Exception as e:
        return {"es_valido": False, "error": str(e)}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================


def obtener_estado_tarea(task_id: str):
    """
    Obtiene el estado actual de una tarea.

    Args:
        task_id (str): ID de la tarea

    Returns:
        dict: Estado de la tarea
    """
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return {"estado": "PENDING", "progreso": 0, "mensaje": "Tarea en cola..."}
    elif result.state == "PROGRESS":
        return {
            "estado": "PROGRESS",
            "progreso": result.info.get("current", 0),
            "mensaje": result.info.get("status", "Procesando..."),
        }
    elif result.state == "SUCCESS":
        return {
            "estado": "SUCCESS",
            "progreso": 100,
            "mensaje": "Procesamiento completado",
            "resultado": result.info.get("resultado", {}),
        }
    elif result.state == "FAILURE":
        return {
            "estado": "FAILURE",
            "progreso": 0,
            "mensaje": f'Error: {result.info.get("status", "Error desconocido")}',
        }
    else:
        return {
            "estado": result.state,
            "progreso": 0,
            "mensaje": f"Estado: {result.state}",
        }
