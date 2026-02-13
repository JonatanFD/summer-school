import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

# Configuración de estilo para los gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ==========================================
print("--- 1. Cargando y Limpiando Datos ---")

# Cargar datasets
# Asegúrate de que los nombres de archivo coincidan con los tuyos
data_dir = Path(__file__).resolve().parent.parent / "4. Economy" / "Data"
df_rep = pd.read_csv(data_dir / "fig_4.2.21.csv")  # AI Talent Representation
df_conc = pd.read_csv(data_dir / "fig_4.2.19.csv") # AI Talent Concentration
df_hiring = pd.read_csv(data_dir / "fig_4.2.14.csv") # Relative AI Hiring

# Función auxiliar para limpiar porcentajes
def clean_pct(x):
    if isinstance(x, str):
        return float(x.replace('%', ''))
    return x

# Limpieza
df_rep['AI talent representation'] = df_rep['AI talent representation'].apply(clean_pct)
df_conc['AI talent concentration'] = df_conc['AI talent concentration'].apply(clean_pct)
df_hiring['Relative AI hiring year-over-year ratio'] = df_hiring['Relative AI hiring year-over-year ratio'].apply(clean_pct)

# Procesar fechas en Hiring data
df_hiring['Date'] = pd.to_datetime(df_hiring['Date'])
df_hiring['Year'] = df_hiring['Date'].dt.year

# Agregación: Promedio anual de Hiring por país
df_hiring_agg = df_hiring.groupby(['Year', 'Geographic area'])['Relative AI hiring year-over-year ratio'].mean().reset_index()
df_hiring_agg.rename(columns={'Relative AI hiring year-over-year ratio': 'Avg_Hiring_Ratio'}, inplace=True)

# Pivoteo: Separar Géneros en columnas distintas para Representación y Concentración
df_rep_pivot = df_rep.pivot_table(index=['Year', 'Geographic area'], columns='Gender', values='AI talent representation').reset_index()
df_rep_pivot.columns = ['Year', 'Geographic area', 'Rep_Female', 'Rep_Male']

df_conc_pivot = df_conc.pivot_table(index=['Year', 'Geographic area'], columns='Gender', values='AI talent concentration').reset_index()
df_conc_pivot.columns = ['Year', 'Geographic area', 'Conc_Female', 'Conc_Male']

# Fusión (Merge) de todos los datos
df_merged = pd.merge(df_rep_pivot, df_conc_pivot, on=['Year', 'Geographic area'], how='outer')
df_final = pd.merge(df_merged, df_hiring_agg, on=['Year', 'Geographic area'], how='outer')

# Eliminar filas con nulos para el análisis multivariado
df_analysis = df_final.dropna().copy()
print(f"Datos listos. Dimensiones finales: {df_analysis.shape}")
print(df_analysis.head())

# ==========================================
# 2. ESTADÍSTICA UNIVARIADA
# ==========================================
print("\n--- 2. Análisis Univariado ---")
print(df_analysis.describe())

# Histogramas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df_analysis['Rep_Female'], kde=True, ax=axes[0], color='skyblue').set_title('Distribución: Representación Femenina (%)')
sns.histplot(df_analysis['Conc_Female'], kde=True, ax=axes[1], color='salmon').set_title('Distribución: Concentración Femenina (%)')
sns.histplot(df_analysis['Avg_Hiring_Ratio'], kde=True, ax=axes[2], color='lightgreen').set_title('Distribución: Hiring Ratio Anual')
plt.tight_layout()
plt.savefig('1_univariado_histogramas.png')
plt.show()

# ==========================================
# 3. ESTADÍSTICA MULTIVARIADA (Clásica y Robusta)
# ==========================================
print("\n--- 3. Análisis Multivariado ---")

# Matrices de Correlación
cols_corr = ['Rep_Female', 'Conc_Female', 'Conc_Male', 'Avg_Hiring_Ratio']
corr_pearson = df_analysis[cols_corr].corr(method='pearson')
corr_spearman = df_analysis[cols_corr].corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', ax=axes[0], vmin=-1, vmax=1).set_title('Correlación de Pearson (Clásica)')
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1).set_title('Correlación de Spearman (Robusta)')
plt.tight_layout()
plt.savefig('2_multivariado_correlacion.png')
plt.show()

# Scatter Plot: Representación vs Hiring
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_analysis, x='Rep_Female', y='Avg_Hiring_Ratio', hue='Year', palette='viridis', size='Conc_Female')
plt.title('Representación Femenina vs Crecimiento en Contratación')
plt.xlabel('Representación Femenina (%)')
plt.ylabel('Avg Hiring Ratio')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig('3_scatter_rep_hiring.png')
plt.show()

# ==========================================
# 4. REDUCCIÓN DE DIMENSIÓN: PCA
# ==========================================
print("\n--- 4. Análisis de Componentes Principales (PCA) ---")

# Estandarización de datos
features = ['Rep_Female', 'Conc_Female', 'Conc_Male', 'Avg_Hiring_Ratio']
x = df_analysis[features].values
x_std = StandardScaler().fit_transform(x)

# Modelo PCA
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x_std)
pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4'])

# Varianza Explicada
print("Varianza Explicada por componente:", pca.explained_variance_ratio_)

# Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 5), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Varianza Explicada Acumulada')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.grid(True)
plt.savefig('4_pca_scree_plot.png')
plt.show()

# Biplot (PC1 vs PC2)
plt.figure(figsize=(10, 8))
# Puntos (Países/Años)
sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=df_analysis['Geographic area'].tolist(), legend=False, alpha=0.6)

# Vectores de carga (Loadings)
coeff = np.transpose(pca.components_[0:2, :])
n = coeff.shape[0]
for i in range(n):
    plt.arrow(0, 0, coeff[i,0]*3, coeff[i,1]*3, color='r', alpha=0.8, head_width=0.05)
    plt.text(coeff[i,0]*3.5, coeff[i,1]*3.5, features[i], color='darkred', ha='center', va='center', weight='bold')

plt.title(f'PCA Biplot: PC1 ({pca.explained_variance_ratio_[0]:.1%}) vs PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.savefig('5_pca_biplot.png')
plt.show()

# ==========================================
# 5. ANÁLISIS DE CORRESPONDENCIAS (Aproximación)
# ==========================================
print("\n--- 5. Análisis de Correspondencias (Visualización) ---")

# Discretizar la variable continua Hiring Ratio en niveles
df_analysis['Hiring_Level'] = pd.qcut(df_analysis['Avg_Hiring_Ratio'], q=3, labels=['Low_Growth', 'Med_Growth', 'High_Growth'])

# Crear tabla de contingencia
contingency_table = pd.crosstab(df_analysis['Geographic area'], df_analysis['Hiring_Level'])

# Cálculo manual de coordenadas para mapa perceptual (Simulación CA simple via SVD sobre residuos estandarizados)
# Nota: Para un CA riguroso se usa librería 'prince' o 'mca', aquí usamos numpy/scipy para no depender de libs externas.
N = contingency_table.sum().sum()
P = contingency_table / N
r = P.sum(axis=1)
c = P.sum(axis=0)
Expected = np.outer(r, c)
Residuals = (P - Expected) / np.sqrt(Expected)

U, s, Vt = np.linalg.svd(Residuals, full_matrices=False)

# Coordenadas
row_coords = U[:, :2] * s[:2] / np.sqrt(r.values[:, None])
col_coords = Vt[:2, :].T * s[:2] / np.sqrt(c.values[:, None])

# Gráfico del Mapa Perceptual
plt.figure(figsize=(12, 10))

# Plot Países (Filas)
plt.scatter(row_coords[:, 0], row_coords[:, 1], c='blue', alpha=0.4, label='Países')
# Etiquetas Países (Solo algunos para no saturar)
for i, txt in enumerate(contingency_table.index):
    # Etiquetar si están lejos del origen (más significativos)
    if (abs(row_coords[i, 0]) > 0.4 or abs(row_coords[i, 1]) > 0.4): 
        plt.annotate(txt, (row_coords[i, 0], row_coords[i, 1]), fontsize=8, alpha=0.7)

# Plot Niveles de Hiring (Columnas)
plt.scatter(col_coords[:, 0], col_coords[:, 1], c='red', marker='D', s=100, label='Nivel de Hiring')
for i, txt in enumerate(contingency_table.columns):
    plt.annotate(txt, (col_coords[i, 0], col_coords[i, 1]), color='red', fontsize=12, weight='bold', ha='right')

plt.axhline(0, color='grey', lw=1, linestyle='--')
plt.axvline(0, color='grey', lw=1, linestyle='--')
plt.title('Mapa de Correspondencias: Países vs Nivel de Crecimiento en Contratación')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.legend()
plt.savefig('6_correspondence_analysis.png')
plt.show()

print("\nAnálisis completado. Gráficos guardados.")