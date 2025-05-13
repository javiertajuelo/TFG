import pandas as pd
import numpy as np
import os
from pathlib import Path

SEED = 121
np.random.seed(SEED)

# Proporciones para la división
TRAIN_RATIO = 0.70
TEST_RATIO = 0.20
VAL_RATIO = 0.10

# Tamaños máximos
MAX_TRAIN = 200000
MAX_TEST = 60000
MAX_VAL = 30000

def concatenar_y_mezclar_datasets(input_files, output_csv):

    print("Concatenando y randomizando datasets")
    
    dfs = []
    for file in input_files:
        print(f"Leyendo {file}")
        df = pd.read_csv(file, header=None)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Randomizar las líneas
    print(f"Randomizando {len(combined_df)} filas")
    combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    combined_df.to_csv(output_csv, header=False, index=False)
    print(f"Dataset combinado guardado en {output_csv} ({len(combined_df)} filas)")
    
    return len(combined_df)

def dividir_dataset(input_csv, total_lineas, base_output_name):

    # Calcular tamaños basados en el total de filas o los máximos definidos
    train_size = min(int(total_lineas * TRAIN_RATIO), MAX_TRAIN)
    test_size = min(int(total_lineas * TEST_RATIO), MAX_TEST)
    val_size = min(int(total_lineas * VAL_RATIO), MAX_VAL)
    
    print(f"\nDividiendo {input_csv} (total: {total_lineas} filas)")
    print(f"Train: {train_size}, Test: {test_size}, Val: {val_size}")
    df = pd.read_csv(input_csv, header=None)
    
    # Dividir los datos
    train = df.iloc[:train_size]
    restante = df.iloc[train_size:]
    
    test = restante.iloc[:test_size]
    val = restante.iloc[test_size:test_size+val_size]
    
    output_dir = "datasets_divididos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar los datasets
    train_file = os.path.join(output_dir, f"Books_train.csv")
    test_file = os.path.join(output_dir, f"Books_test.csv")
    val_file = os.path.join(output_dir, f"Books_val.csv")
    
    train.to_csv(train_file, header=False, index=False)
    test.to_csv(test_file, header=False, index=False)
    val.to_csv(val_file, header=False, index=False)
    
    print(f"Datasets guardados en {output_dir}/")
    print(f"{Path(train_file).name}: {len(train)} filas")
    print(f"{Path(test_file).name}: {len(test)} filas")
    print(f"{Path(val_file).name}: {len(val)} filas")

def procesar_datasets(input_files, nombre_salida):

    combined_csv = f"{nombre_salida}_combined.csv"
    total_lineas = concatenar_y_mezclar_datasets(input_files, combined_csv)
    dividir_dataset(combined_csv, total_lineas, nombre_salida)


if __name__ == "__main__":
    input_files = [
        "Books_train.csv",
        "Books_test.csv",
        "Books_val.csv"
    ]
    
    # Verificar que los archivos existan
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Los siguientes archivos no se encontraron:")
        for f in missing_files:
            print(f"  - {f}")
        exit(1)
    
    nombre_salida  = "dataset_reordenado"

    procesar_datasets(input_files, nombre_salida)
    
    print("\nProceso completado")