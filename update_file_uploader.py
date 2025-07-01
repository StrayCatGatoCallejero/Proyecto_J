#!/usr/bin/env python3
"""
Script para actualizar el file_uploader en app_front.py
"""

def update_file_uploader():
    # Leer el archivo
    with open('examples/app_front.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Reemplazar la línea del file_uploader
    old_line = 'archivo = st.file_uploader("📂 Sube tu archivo .sav o .dta", type=["sav", "dta"])'
    new_line = '''archivo = st.file_uploader(
    "📂 Sube tu archivo de datos", 
    type=["sav", "dta", "csv", "xlsx", "xls"],
    help="Formatos soportados: SPSS (.sav), Stata (.dta), Excel (.xlsx, .xls), CSV (.csv)"
)'''
    
    # Hacer el reemplazo
    new_content = content.replace(old_line, new_line)
    
    # Escribir el archivo actualizado
    with open('examples/app_front.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ File uploader actualizado correctamente")
    print("📁 Ahora soporta: .sav, .dta, .csv, .xlsx, .xls")

if __name__ == "__main__":
    update_file_uploader() 