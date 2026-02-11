"""Genera gráficas del dataset Titanic y guarda resultados en la carpeta tareas.

- Carga `datos/titanic.csv` (delimitador `;`).
- Calcula tasas de supervivencia por `pclass` y proporción de supervivientes.
- Guarda: `tareas/survival_by_pclass.png`, `tareas/survived_pie.png`, `tareas/titanic_summary.csv`.

Uso:
    python tareas/grafico.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datos' / 'titanic.csv'
OUT_DIR = ROOT / 'tareas'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    dtypes = df.dtypes
    summary = pd.DataFrame({'missing': missing, 'dtype': dtypes})
    return summary


def plot_survival_by_pclass(df: pd.DataFrame, outpath: Path):
    # Nos aseguramos de tener valores numéricos y no nulos para pclass y survived
    df2 = df.dropna(subset=['pclass', 'survived'])
    survival_by_pclass = df2.groupby('pclass')['survived'].agg(['sum', 'count'])
    survival_by_pclass['rate'] = survival_by_pclass['sum'] / survival_by_pclass['count']
    survival_by_pclass = survival_by_pclass.sort_index()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(survival_by_pclass.index.astype(str), survival_by_pclass['rate'],
                   color=['#2ca02c', '#ff7f0e', '#d62728'])
    plt.ylim(0, 1)
    plt.title('Tasa de supervivencia por clase (pclass)')
    plt.xlabel('pclass')
    plt.ylabel('Tasa de supervivencia')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Añadir etiquetas con porcentaje encima de barras
    for rect, val in zip(bars, survival_by_pclass['rate']):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.02,
                 f"{val:.1%}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_survived_pie(df: pd.DataFrame, outpath: Path):
    counts = df['survived'].value_counts().reindex([1, 0]).fillna(0)
    labels = ['Survived', 'Not Survived']
    colors = ['#2ca02c', '#d62728']

    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Proporción supervivientes vs no supervivientes')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def print_key_findings(df: pd.DataFrame):
    rows = len(df)
    total_survived = int(df['survived'].sum())
    survival_rate = total_survived / rows
    print(f"Filas: {rows} | Sobrevivientes: {total_survived} | Tasa: {survival_rate:.2%}")

    by_sex = df.groupby('sex')['survived'].agg(['sum', 'count'])
    by_sex['rate'] = by_sex['sum'] / by_sex['count']
    print('\nTasa de supervivencia por sexo:')
    print(by_sex[['sum', 'count', 'rate']].to_string())

    by_class = df.groupby('pclass')['survived'].agg(['sum', 'count'])
    by_class['rate'] = by_class['sum'] / by_class['count']
    print('\nTasa de supervivencia por pclass:')
    print(by_class[['sum', 'count', 'rate']].to_string())


if __name__ == '__main__':
    df = load_data(DATA)

    # Guardar resumen (faltantes y tipos)
    summary = summary_stats(df)
    summary.to_csv(OUT_DIR / 'titanic_summary.csv')

    # Imprimir hallazgos clave
    print_key_findings(df)

    # Generar y guardar gráficas
    plot_survival_by_pclass(df, OUT_DIR / 'survival_by_pclass.png')
    plot_survived_pie(df, OUT_DIR / 'survived_pie.png')

    print('\nGráficas guardadas en:', OUT_DIR)
    print('Files: survival_by_pclass.png, survived_pie.png, titanic_summary.csv')
