import pandas as pd
import numpy as np
import os
from collections import defaultdict
from pathlib import Path

SEED = 121
np.random.seed(SEED)

# Tamaños objetivo
MAX_TRAIN = 200000
MAX_TEST = 60000
MAX_VAL = 30000

# Parámetros para densidad
MIN_USER_INTERACTIONS = 10  # Usuarios con al menos X interacciones
MIN_ITEM_INTERACTIONS = 50  # Ítems con al menos X interacciones


def crear_directorio_salida(base_dir="datasets_procesados"):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def aumentar_densidad(input_csv):
    print(f"\nProcesando {input_csv} para aumentar densidad")

    df = pd.read_csv(input_csv, header=None,
                     names=['asin', 'reviewerID', 'overall'],
                     dtype={'asin': 'category', 'reviewerID': 'category'})

    # 1. Filtrado de usuarios activos
    contador_usuarios = df['reviewerID'].value_counts()
    usuarios_activos = contador_usuarios[contador_usuarios >= MIN_USER_INTERACTIONS].index
    df = df[df['reviewerID'].isin(usuarios_activos)]

    # 2. Filtrado de ítems populares
    contador_usuarios = df['asin'].value_counts()
    items_populares = contador_usuarios[contador_usuarios >= MIN_ITEM_INTERACTIONS].index
    df = df[df['asin'].isin(items_populares)]

    # 3. Mezcla final
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"Post-filtrado: {len(df):,} interacciones")
    print(f"Usuarios únicos: {df['reviewerID'].nunique():,}")
    print(f"Ítems únicos: {df['asin'].nunique():,}")
    densidad = len(df) / (df['reviewerID'].nunique() * df['asin'].nunique())
    print(f"Densidad: {densidad:.6f}")

    return df


def dividir_dataset(df, base_output_name, output_dir):
    items = df['asin'].unique()
    np.random.shuffle(items)

    train, val, test = [], [], []
    train_counts = defaultdict(int)

    for item in items:
        grupo = df[df['asin'] == item]

        # Priorizar train para mantener densidad
        if sum(train_counts.values()) < MAX_TRAIN:
            muestras = grupo.sample(n=min(len(grupo), 200), random_state=SEED)
            train.append(muestras)
            train_counts[item] += len(muestras)
        elif len(val) < MAX_VAL:
            val.append(grupo.sample(n=min(len(grupo), 50), random_state=SEED))
        else:
            test.append(grupo.sample(n=min(len(grupo), 50), random_state=SEED))

    # 2. Concatenación y ajuste de tamaños
    df_train = pd.concat(train).sample(frac=1, random_state=SEED).head(MAX_TRAIN)
    df_val = pd.concat(val).sample(frac=1, random_state=SEED).head(MAX_VAL)
    df_test = pd.concat(test).sample(frac=1, random_state=SEED).head(MAX_TEST)

    df_train.to_csv(f"{output_dir}/{base_output_name}_train.csv", header=False, index=False)
    df_val.to_csv(f"{output_dir}/{base_output_name}_val.csv", header=False, index=False)
    df_test.to_csv(f"{output_dir}/{base_output_name}_test.csv", header=False, index=False)

    print(f"\nDivisión completada en {output_dir}:")
    print(f"Train: {len(df_train):,} líneas")
    print(f"Val: {len(df_val):,} líneas")
    print(f"Test: {len(df_test):,} líneas")


def procesar_csv(input_csv, output_dir=None):
    if output_dir is None:
        output_dir = crear_directorio_salida()

    base_name = Path(input_csv).stem
    print(f"\n{'=' * 50}\nProcesando: {base_name}\n{'=' * 50}")

    # Paso 1: Aumentar densidad
    df_filtrado = aumentar_densidad(input_csv)

    # Paso 2: Dividir dataset
    dividir_dataset(df_filtrado, base_name, output_dir)


if __name__ == "__main__":
    directorio = "datasets_densidad"

    procesar_csv('Books_5_completo.csv', output_dir=directorio)

    print(f"\nProceso completado. Datasets guardados en /{directorio}")