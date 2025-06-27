# 🔄 Sistema de Procesamiento Asíncrono - Proyecto J

## 📋 Descripción

Sistema de procesamiento asíncrono para archivos CSV grandes que proporciona una experiencia de usuario amena y profesional. Utiliza **Celery** para tareas en segundo plano y **Redis** como broker de mensajes.

## ✨ Características

### 🚀 **Experiencia de Usuario**
- ✅ **Carga de archivos hasta 200 MB**
- ✅ **Vista previa instantánea** (primeras 5 filas)
- ✅ **Procesamiento en segundo plano** (no bloquea la interfaz)
- ✅ **Monitoreo en tiempo real** con barras de progreso
- ✅ **Indicadores visuales** de estado (En cola, Procesando, Completado)
- ✅ **Interfaz responsiva** y profesional

### 🔧 **Funcionalidades Técnicas**
- ✅ **Procesamiento en chunks** (10,000 filas por vez)
- ✅ **Análisis estadístico completo** de columnas
- ✅ **Generación de archivos Parquet** optimizados
- ✅ **Manejo robusto de errores**
- ✅ **Limpieza automática** de archivos temporales

### 📊 **Archivos Generados**
1. **Resumen General** (`{archivo}_resumen.parquet`)
   - Estadísticas básicas del archivo
   - Total de filas, columnas, memoria utilizada

2. **Análisis de Columnas** (`{archivo}_columnas.parquet`)
   - Información detallada de cada columna
   - Tipos de datos, valores únicos, estadísticas

3. **Muestra de Datos** (`{archivo}_muestra.parquet`)
   - Primera 1000 filas del archivo original
   - Para análisis rápido sin cargar todo

## 🛠️ Instalación

### 1. **Instalar Dependencias**
```bash
pip install -r requirements_async.txt
```

### 2. **Configurar Redis**

#### Opción A: Instalación Local
```bash
# Windows
# Descargar desde https://redis.io/download

# Linux/Mac
brew install redis
redis-server

# Ubuntu/Debian
sudo apt-get install redis-server
redis-server
```

#### Opción B: Docker
```bash
docker run -d -p 6379:6379 redis:alpine
```

### 3. **Verificar Configuración**
```bash
python run_async_system.py --check
```

## 🚀 Uso

### **Paso 1: Iniciar Worker (Terminal 1)**
```bash
python run_async_system.py --worker
```

### **Paso 2: Iniciar Aplicación (Terminal 2)**
```bash
python run_async_system.py --app
```

### **Paso 3: Usar la Aplicación**
1. Abrir http://localhost:8501
2. Subir archivo CSV (máximo 200 MB)
3. Revisar vista previa
4. Hacer clic en "Procesar Archivo Completo"
5. Monitorear progreso en tiempo real
6. Descargar resultados cuando termine

## 📁 Estructura de Archivos

```
Proyecto_J/
├── tasks.py                 # Tareas de Celery
├── streamlit_app.py         # Aplicación Streamlit
├── run_async_system.py      # Script de configuración
├── requirements_async.txt   # Dependencias
├── README_ASYNC.md         # Esta documentación
├── temp/                   # Archivos temporales
└── resultados/             # Archivos de resultados
    ├── archivo_resumen.parquet
    ├── archivo_columnas.parquet
    └── archivo_muestra.parquet
```

## 🔧 Configuración Avanzada

### **Variables de Entorno**
Crear archivo `.env`:
```env
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
MAX_FILE_SIZE_MB=200
CHUNK_SIZE=10000
```

### **Personalizar Procesamiento**
Editar `tasks.py` para agregar lógica personalizada:

```python
@celery_app.task(name='procesar_archivo', bind=True)
def procesar_archivo(self, temp_path: str):
    # Tu lógica personalizada aquí
    for chunk in pd.read_csv(temp_path, chunksize=10_000):
        # Procesar cada chunk
        resultado = tu_funcion_personalizada(chunk)
        
        # Actualizar progreso
        self.update_state(
            state='PROGRESS',
            meta={'current': progreso, 'total': 100, 'status': mensaje}
        )
```

## 📊 Monitoreo y Logs

### **Estados de Tareas**
- **PENDING**: Tarea en cola
- **PROGRESS**: Procesando con progreso
- **SUCCESS**: Completado exitosamente
- **FAILURE**: Error en el procesamiento

### **Logs del Worker**
```bash
# Ver logs detallados
celery -A tasks worker --loglevel=debug

# Ver tareas activas
celery -A tasks inspect active
```

## 🐛 Solución de Problemas

### **Error: Redis no conecta**
```bash
# Verificar que Redis esté ejecutándose
redis-cli ping
# Debe responder: PONG
```

### **Error: Worker no inicia**
```bash
# Verificar dependencias
pip install -r requirements_async.txt

# Verificar configuración
python run_async_system.py --check
```

### **Error: Archivo muy grande**
- El límite es 200 MB por defecto
- Modificar en `streamlit_app.py` línea 200
- Considerar procesamiento por partes

### **Error: Memoria insuficiente**
- Reducir `chunk_size` en `tasks.py`
- Usar archivos más pequeños
- Aumentar memoria del sistema

## 🔒 Seguridad

### **Archivos Temporales**
- Se almacenan en `./temp/`
- Se eliminan automáticamente al limpiar sesión
- No se almacenan permanentemente

### **Validación de Archivos**
- Solo archivos CSV
- Validación de tamaño
- Verificación de formato

## 📈 Rendimiento

### **Optimizaciones Implementadas**
- ✅ Procesamiento en chunks
- ✅ Actualización de progreso eficiente
- ✅ Archivos Parquet para resultados
- ✅ Limpieza automática de memoria

### **Métricas Típicas**
- **Archivo 1 MB**: ~5-10 segundos
- **Archivo 10 MB**: ~30-60 segundos
- **Archivo 100 MB**: ~5-10 minutos
- **Archivo 200 MB**: ~10-20 minutos

## 🔄 Integración con Proyecto J

### **Compatibilidad**
- ✅ Mantiene el diseño visual fijo
- ✅ Usa la misma paleta de colores
- ✅ Integra con el sistema de temas
- ✅ Compatible con aplicaciones existentes

### **Extensibilidad**
- Fácil agregar nuevos tipos de procesamiento
- Configuración modular
- API consistente con el resto del proyecto

## 📞 Soporte

### **Comandos de Diagnóstico**
```bash
# Verificar sistema completo
python run_async_system.py --check

# Ver logs de Celery
celery -A tasks worker --loglevel=info

# Ver estado de Redis
redis-cli info
```

### **Contacto**
Para problemas específicos, revisar:
1. Logs del worker
2. Estado de Redis
3. Configuración de dependencias
4. Tamaño y formato del archivo

---

**Desarrollado para Proyecto J** | **Versión**: 1.0 | **Última actualización**: 2025 