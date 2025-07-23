import pandas as pd

file_path = r"C:\Users\StrayCat\Downloads\ejemplo.xlsx"

print("Intentando abrir como Excel...")
try:
    df = pd.read_excel(file_path, engine="openpyxl")
    print("¡Éxito! Shape:", df.shape)
except Exception as e:
    print("Error con openpyxl:", e)
    try:
        df = pd.read_excel(file_path)
        print("¡Éxito con engine por defecto! Shape:", df.shape)
    except Exception as e2:
        print("Error con engine por defecto:", e2) 