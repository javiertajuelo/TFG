import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import json

def extraer_generos_libros(df_libros):

    df_libros['Categories'] = df_libros['Categories'].fillna('')

    generos_libros = set()
    for categorias in df_libros['Categories']:
        if categorias:
            generos_libros.update([cat.strip() for cat in categorias.split(',')])

    return sorted(generos_libros)




def extraer_generos_peliculas(df_peliculas):

    generos_peliculas = set()

    for fila in df_peliculas['genres']:
        try:
            generos = json.loads(fila)
            generos_peliculas.update([genero['name'] for genero in generos])
        except (json.JSONDecodeError, TypeError):
            continue

    return sorted(generos_peliculas)

import json

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

df_peliculas = pd.read_csv('tmdb_5000_movies.csv')
generos_peliculas = extraer_generos_peliculas(df_peliculas)

df_libros = pd.read_csv('Books_transformado.csv')
generos_libros = extraer_generos_libros(df_libros)

embeddings_libros = [get_bert_embedding(libro) for libro in generos_libros]
embeddings_peliculas = [get_bert_embedding(pelicula) for pelicula in generos_peliculas]

similitudes = cosine_similarity(
    [embedding.numpy().flatten() for embedding in embeddings_libros],
    [embedding.numpy().flatten() for embedding in embeddings_peliculas]
)

umbral = 0.85
diccionario_similitudes = {}

for i, libro in enumerate(generos_libros):
    for j, pelicula in enumerate(generos_peliculas):
        if similitudes[i][j] >= umbral:
            if pelicula not in diccionario_similitudes:
                diccionario_similitudes[pelicula] = []
            diccionario_similitudes[pelicula].append((libro, similitudes[i][j]))

for pelicula, libros in diccionario_similitudes.items():
    diccionario_similitudes[pelicula] = sorted(libros, key=lambda x: x[1], reverse=True)

diccionario_similitudes_serializable = {
    pelicula: [{"libro": libro, "similitud": float(sim)} for libro, sim in libros]
    for pelicula, libros in diccionario_similitudes.items()
}

archivo_salida = 'generos_similitudes.json'
with open(archivo_salida, 'w', encoding='utf-8') as f:
    json.dump(diccionario_similitudes_serializable, f, ensure_ascii=False, indent=4)

print(f"El diccionario se ha guardado en el archivo '{archivo_salida}'.")




