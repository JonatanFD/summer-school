import pandas as pd
from pathlib import Path

# Ruta al archivo de datos (futbol.xlsx está en la carpeta datos)
DATA_DIR = Path(__file__).resolve().parent.parent / "datos"
FUTBOL_XLSX = DATA_DIR / "futbol.xlsx"

# Leer la base de datos 'futbol'
try:
    futbol = pd.read_excel(FUTBOL_XLSX)
    print(f"Archivo cargado: {FUTBOL_XLSX}")
except FileNotFoundError:
    raise FileNotFoundError(f"No se encontró {FUTBOL_XLSX}. Asegúrese de que el archivo exista.")

# a) Seleccionar las primeras dos filas y las columnas 2 hasta 5
print("a) Primeras dos filas y columnas 2 hasta 5:")
print(futbol.iloc[0:2, 1:5])
print()

# b) Seleccionar la tercera fila
print("b) Tercera fila:")
print(futbol.iloc[2])
print()

# c) Elemento de la cuarta columna y fila 7
print("c) Elemento en la columna 4 y fila 7:")
print(futbol.iloc[6, 3])
print()

# d) Seleccionar las primeras dos filas y las primeras dos columnas,
# guardarlos en una base denominada fulbito y exportarla en csv
fulbito = futbol.iloc[0:2, 0:2]

print("d) Base fulbito:")
print(fulbito)

OUT_CSV = DATA_DIR / "fulbito.csv"
fulbito.to_csv(OUT_CSV, index=False)

print(f"Archivo {OUT_CSV} exportado exitosamente")
