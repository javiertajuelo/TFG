import pandas as pd
import numpy as np
import os
import json
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

def json_to_csv(input_json, output_csv):
    print(f"Procesando {input_json}")
    
    data = []
    with open(input_json, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Seleccionar solo las columnas que necesitamos
                row = {
                    'asin': item.get('asin', ''),
                    'reviewerID': item.get('reviewerID', ''),
                    'overall': item.get('overall', '')
                }
                data.append(row)
            except json.JSONDecodeError:
                print(f"Error al decodificar línea: {line}")
                continue
    
    df = pd.DataFrame(data)
    
    # Reordenar aleatoriamente
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    df.to_csv(output_csv, header=False, index=False)
    print(f"CSV guardado en {output_csv} ({len(df)} filas)")
    return len(df)

def dividir_dataset(input_csv, total_lineas, nombre_base):
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
    
    output_dir = "division_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar los datasets
    train_file = os.path.join(output_dir, f"{nombre_base}_train.csv")
    test_file = os.path.join(output_dir, f"{nombre_base}_test.csv")
    val_file = os.path.join(output_dir, f"{nombre_base}_val.csv")
    
    train.to_csv(train_file, header=False, index=False)
    test.to_csv(test_file, header=False, index=False)
    val.to_csv(val_file, header=False, index=False)
    
    print(f"Datasets guardados en {output_dir}/")
    print(f"{Path(train_file).name}: {len(train)} filas")
    print(f"{Path(test_file).name}: {len(test)} filas")
    print(f"{Path(val_file).name}: {len(val)} filas")

def procesar_json_file(json_file):
    nombre_base = Path(json_file).stem
    csv_file = f"{nombre_base}.csv"
    
    total_lineas = json_to_csv(json_file, csv_file)
    
    dividir_dataset(csv_file, total_lineas, nombre_base)


if __name__ == "__main__":
    json_files = [f for f in os.listdir() if f.endswith('.json')]
    
    if not json_files:
        print("No se encontraron archivos JSON en el directorio actual")
    else:
        for json_file in json_files:
            procesar_json_file(json_file)
    
    print("\nProceso completado")