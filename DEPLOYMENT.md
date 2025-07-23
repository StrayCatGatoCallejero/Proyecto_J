# 🚀 Guía de Despliegue en Render - Proyecto J

## 📋 Requisitos Previos

1. **Cuenta en Render**: Regístrate en [render.com](https://render.com)
2. **Repositorio Git**: Tu proyecto debe estar en GitHub, GitLab o similar
3. **Archivos de configuración**: Ya están creados en este proyecto

## 📁 Archivos de Configuración Creados

- ✅ `requirements.txt` - Dependencias de Python
- ✅ `render.yaml` - Configuración de Render
- ✅ `Procfile` - Comando de inicio
- ✅ `.streamlit/config.toml` - Configuración de Streamlit
- ✅ `setup.sh` - Script de configuración del sistema

## 🛠️ Pasos para Desplegar

### 1. Preparar el Repositorio

```bash
# Asegúrate de que todos los archivos estén en tu repositorio
git add .
git commit -m "Preparar para despliegue en Render"
git push origin main
```

### 2. Crear Servicio en Render

1. **Ve a [render.com](https://render.com)** y inicia sesión
2. **Haz clic en "New +"** → **"Web Service"**
3. **Conecta tu repositorio** de GitHub/GitLab
4. **Selecciona el repositorio** de Proyecto J

### 3. Configurar el Servicio

**Configuración automática** (si usas `render.yaml`):
- Render detectará automáticamente la configuración

**Configuración manual**:
- **Name**: `proyecto-j-streamlit`
- **Environment**: `Python`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run streamlit_app_simple.py --server.port $PORT --server.address 0.0.0.0`
- **Plan**: `Free`

### 4. Variables de Entorno (Opcional)

Si necesitas configuraciones adicionales:
- **PYTHON_VERSION**: `3.11.5`
- **DISPLAY**: `:99` (para Kaleido)

### 5. Desplegar

1. **Haz clic en "Create Web Service"**
2. **Espera** a que se complete el build (5-10 minutos)
3. **Tu app estará disponible** en la URL proporcionada

## 🌐 URLs de Acceso

- **URL de desarrollo**: `https://tu-app.onrender.com`
- **URL personalizada**: Puedes configurar un dominio personalizado

## 🔧 Solución de Problemas

### Error de Kaleido
Si ves errores relacionados con Kaleido:
- El `setup.sh` ya incluye la instalación de Chrome
- Render debería manejar esto automáticamente

### Error de Puerto
- El comando de inicio ya incluye `--server.port $PORT`
- Render asigna automáticamente el puerto

### Error de Dependencias
- Verifica que `requirements.txt` contenga todas las dependencias
- Los logs de build mostrarán errores específicos

## 📊 Monitoreo

- **Logs**: Disponibles en el dashboard de Render
- **Métricas**: Uso de CPU, memoria y tiempo de respuesta
- **Uptime**: Monitoreo automático de disponibilidad

## 🔄 Actualizaciones

Para actualizar la aplicación:
```bash
git add .
git commit -m "Actualización de la aplicación"
git push origin main
```
Render detectará automáticamente los cambios y redeployará.

## 💰 Costos

- **Plan Free**: $0/mes (con limitaciones)
- **Plan Paid**: Desde $7/mes (sin limitaciones)

## 🎉 ¡Listo!

Tu aplicación **Asistente de Visualización de Datos - Proyecto J** estará disponible públicamente en internet. 