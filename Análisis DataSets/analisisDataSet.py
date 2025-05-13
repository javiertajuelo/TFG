import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
from itertools import combinations

plt.style.use('ggplot')
sns.set_palette("husl")
datasets = ['Books_5','Music','Movies_and_TV_5','Video_Games_5']


def cargar_datos_completos(dominio):
    dfs = []
    for sufijo in ['_train', '_test', '_val']:
        try:
            archivo = f"{dominio}{sufijo}.csv"
            df = pd.read_csv(archivo, header=None, names=['parent_asin', 'user_id', 'rating'])
            dfs.append(df)
        except FileNotFoundError:
            continue
    return pd.concat(dfs) if dfs else None


def analisis_basico_datos():
    resultados = []

    for dominio in datasets:
        df = cargar_datos_completos(dominio)
        if df is None or df.empty:
            continue

        # Cálculos básicos
        num_usuarios = df['user_id'].nunique()
        num_items = df['parent_asin'].nunique()
        num_interacciones = len(df)
        densidad = num_interacciones / (num_usuarios * num_items)

        # Distribución de ratings
        rating_stats = df['rating'].describe()

        # Frecuencia de items
        item_counts = df['parent_asin'].value_counts()

        # Variabilidad de ratings por item
        item_rating_var = df.groupby('parent_asin')['rating'].var().fillna(0)

        resultados.append({
            'Dominio': dominio,
            'Usuarios': num_usuarios,
            'Items': num_items,
            'Interacciones': num_interacciones,
            'Densidad': densidad,
            'Rating_mean': rating_stats['mean'],
            'Rating_std': rating_stats['std'],
            'Item_freq_mean': item_counts.mean(),
            'Item_freq_std': item_counts.std(),
            'Rating_var_mean': item_rating_var.mean(),
            'Rating_var_std': item_rating_var.std()
        })

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análisis para dominio: {dominio}', fontsize=16)

        sns.countplot(x='rating', data=df, ax=axes[0, 0])
        axes[0, 0].set_title('Distribución de Ratings')

        item_counts.head(20).sort_values().plot(kind='barh', ax=axes[0, 1])
        axes[0, 1].set_title('Top 20 Items más Frecuentes')

        sns.histplot(item_counts, bins=50, ax=axes[1, 0], log_scale=(True, True))
        axes[1, 0].set_title('Distribución de Frecuencia de Items (log-log)')
        axes[1, 0].set_xlabel('Frecuencia')
        axes[1, 0].set_ylabel('Número de Items')

        sns.histplot(item_rating_var, bins=50, ax=axes[1, 1])
        axes[1, 1].set_title('Distribución de Variabilidad de Ratings por Item')
        axes[1, 1].set_xlabel('Varianza del Rating')

        plt.tight_layout()
        plt.savefig(f'analisis_{dominio}.png')
        plt.close()

    df_resultados = pd.DataFrame(resultados)

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Comparativa entre Dominios', fontsize=16)

    df_resultados.set_index('Dominio')[['Usuarios', 'Items']].plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Usuarios e Items Únicos')
    axes[0, 0].set_yscale('log')

    df_resultados.set_index('Dominio')['Interacciones'].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Total de Interacciones')
    axes[0, 1].set_yscale('log')

    df_resultados.set_index('Dominio')['Densidad'].plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Densidad de la Matriz (Interacciones/(Usuarios*Items))')

    df_resultados.set_index('Dominio')[['Rating_mean', 'Rating_std']].plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Media y Desviación Estándar de Ratings')

    df_resultados.set_index('Dominio')[['Item_freq_mean', 'Item_freq_std']].plot(kind='bar', ax=axes[2, 0])
    axes[2, 0].set_title('Frecuencia Media y Desviación de Items')

    df_resultados.set_index('Dominio')[['Rating_var_mean', 'Rating_var_std']].plot(kind='bar', ax=axes[2, 1])
    axes[2, 1].set_title('Variabilidad Media de Ratings por Item')

    plt.tight_layout()
    plt.savefig('comparativa_dominios.png')
    plt.close()

    return df_resultados


def analisis_correlacion(df_resultados):
    numeric_df = df_resultados.select_dtypes(include=[np.number])

    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación entre Métricas')
    plt.tight_layout()
    plt.savefig('correlacion_metricas.png')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.scatterplot(data=df_resultados, x='Usuarios', y='Interacciones', hue='Dominio', s=100, ax=axes[0])
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title('Usuarios vs Interacciones (log-log)')

    sns.scatterplot(data=df_resultados, x='Item_freq_mean', y='Rating_var_mean', hue='Dominio', s=100, ax=axes[1])
    axes[1].set_title('Frecuencia Media de Items vs Variabilidad de Ratings')

    plt.tight_layout()
    plt.savefig('scatter_plots.png')
    plt.close()

def contar_lineas_archivos():

    for dominio in datasets:
        total_lineas = 0
        print(f"\nDominio: {dominio}")
        for sufijo in ['_train', '_test', '_val']:
            try:
                archivo = f"{dominio}{sufijo}.csv"
                with open(archivo, 'r', encoding='utf-8') as f:
                    lineas = sum(1 for _ in f)
                total_lineas += lineas
                print(f"{archivo:}: {lineas:} líneas")
            except FileNotFoundError:
                continue
        print(f"Total {dominio:}: {total_lineas:} líneas")


def analizar_usuarios_comunes():

    usuarios_por_dominio = defaultdict(set)
    for dominio in datasets:
        df = cargar_datos_completos(dominio)
        if df is not None:
            usuarios_por_dominio[dominio] = set(df['user_id'].unique())

    # Calcular intersecciones
    usuarios_comunes_matrix = pd.DataFrame(0, index=datasets, columns=datasets)
    for (dom1, dom2) in combinations(datasets, 2):
        comunes = usuarios_por_dominio[dom1] & usuarios_por_dominio[dom2]
        usuarios_comunes_matrix.loc[dom1, dom2] = len(comunes)
        usuarios_comunes_matrix.loc[dom2, dom1] = len(comunes)

    # Usuarios únicos por dominio
    usuarios_unicos = {dom: len(usuarios) for dom, usuarios in usuarios_por_dominio.items()}

    # Usuarios en todos los dominios
    usuarios_todos = len(set.intersection(*usuarios_por_dominio.values()))

    plt.figure(figsize=(10, 8))
    sns.heatmap(usuarios_comunes_matrix, annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Usuarios Comunes'})
    plt.title('Usuarios Comunes entre Pares de Dominios', pad=20)
    plt.tight_layout()
    plt.savefig('usuarios_comunes_heatmap.png')
    plt.close()

    print("\nANÁLISIS DE USUARIOS COMUNES")
    print("\nUsuarios únicos por dominio:")
    for dom, count in usuarios_unicos.items():
        print(f"{dom:}: {count:} usuarios")

    print("\nUsuarios comunes entre pares:")
    print(usuarios_comunes_matrix)

    print(f"\nUsuarios presentes en TODOS los dominios: {usuarios_todos}")

def generar_reporte_completo():
    print("Numero de lineas de los archivos")
    contar_lineas_archivos()
    print("\nUsuarios en comun")
    analizar_usuarios_comunes()

    print("\nRealizando análisis básico de datos")
    df_resultados = analisis_basico_datos()
    print("\nResultados básicos:")
    print(df_resultados.to_string())

    print("\nAnalizando correlaciones")
    analisis_correlacion(df_resultados)

    print("\nAnálisis completado.")


if __name__ == "__main__":
    generar_reporte_completo()