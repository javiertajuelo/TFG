import pandas as pd
import json

jsonl_file = "meta_Movies_and_TV.jsonl"
csv_file = "meta_Movies_and_TV.csv"

data = []

# Leer el archivo JSON línea por línea
with open(jsonl_file, "r", encoding="utf-8") as file:
    for line in file:
        review = json.loads(line)
        parent_asin = review.get("parent_asin", "")
        main_category = review.get("main_category", "")
        title = review.get("title", "")
        price = review.get("price", "")
        description = review.get("description", "")

        data.append([parent_asin, main_category, title, price, description])

# Convertir la lista en un DataFrame
df = pd.DataFrame(data, columns=["parent_asin", "main_category","title","price", "description"])

# Guardar en CSV
df.to_csv(csv_file, index=False, header=False)

print(f"Archivo CSV generado: {csv_file}")
