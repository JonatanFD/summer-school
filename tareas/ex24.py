import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Argumentos de linea de comando: usar --show para abrir las ventanas interactivas (bloquea)
parser = argparse.ArgumentParser(description="Ejercicio 24: graficos para razaperros")
parser.add_argument('--show', action='store_true', help='Mostrar ventanas interactivas (bloquea)')
args = parser.parse_args()

# Cargar base razaperros en el objeto perros
csv_path = r"C:\Users\nanab\Downloads\OneDrive_1_2-9-2026\datos\razaperros.csv"
try:
    # Intentar UTF-8 primero, luego latin-1 y cp1252 como fallback
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            perros = pd.read_csv(csv_path, encoding=enc)
            print(f"Datos cargados desde {csv_path} con encoding={enc}. Filas: {len(perros)}, columnas: {len(perros.columns)}")
            break
        except UnicodeDecodeError:
            continue
    else:
        # Si ninguno funcionó, leer reemplazando caracteres inválidos
        perros = pd.read_csv(csv_path, encoding="utf-8", errors="replace")
        print(f"Se cargó {csv_path} con errors='replace' (caracteres inválidos reemplazados). Filas: {len(perros)}, columnas: {len(perros.columns)}")
except FileNotFoundError:
    raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

# Grafico de barras para la variable funcion
funcion_counts = perros["funcion"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
funcion_counts.plot(kind="bar", color="#4C78A8", edgecolor="black")
plt.title("Distribucion de la variable funcion")
plt.xlabel("Funcion")
plt.ylabel("Frecuencia")
plt.tight_layout()
# Guardar en archivo PNG por defecto (no bloquea). Use --show para ver la ventana interactiva.
if args.show:
    plt.show()
else:
    out = r"tareas/funcion_barras.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Grafico de 'funcion' guardado en {out}")
plt.close()

# Diagrama circular para la variable inteligencia
inteligencia_counts = perros["inteligencia"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    inteligencia_counts,
    labels=inteligencia_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
)
plt.title("Distribucion de la variable inteligencia")
plt.tight_layout()
# Guardar en archivo PNG por defecto (no bloquea). Use --show para ver la ventana interactiva.
if args.show:
    plt.show()
else:
    out = r"tareas/inteligencia_pie.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Grafico de 'inteligencia' guardado en {out}")
plt.close()

# Comentario:
# La variable funcion muestra que algunas categorias concentran mas razas,
# mientras que la variable inteligencia se reparte de forma desigual, con
# niveles predominantes frente a otros menos frecuentes.
