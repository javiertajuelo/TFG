import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score
from tqdm import tqdm  # Para mostrar progreso

pd.set_option('display.max_colwidth', None)  # No trunca el contenido de las columnas
pd.set_option('display.width', None)  # No trunca el ancho total de la salida
pd.set_option('display.max_columns', None)

N = 200000
N2 = 1000000


# Función para cargar datasets y tomar muestras
def cargar_datos(movie_csv, book_csv, song_csv, movie_sample_size=1000, book_sample_size=1000,
                         combined_filename='combinedDataSetWE.pkl'):
    # Si el archivo combinado ya existe, simplemente cargarlo
    if os.path.exists(combined_filename):
        print("Cargando datos combinados desde el archivo existente")
        return pd.read_pickle(combined_filename)

    # Cargar datasets sin encabezados y asignar nombres de columna según corresponda.
    df_movies = pd.read_csv(movie_csv, header=None, names=["id", "main_category", "title", "price", "description"],
                            nrows=N2)
    df_books = pd.read_csv(book_csv, header=None, names=["id", "main_category", "title", "price", "description"],
                           nrows=N2)
    df_songs = pd.read_csv(song_csv, header=None, names=["id", "main_category", "title", "price", "description"],
                           nrows=N2)

    df_movies['type'] = 'movie'
    df_books['type'] = 'book'
    df_songs['type'] = 'song'

    # Concatenar y renombrar columnas para tener un formato uniforme.
    combined_sample_df = pd.concat([
        df_movies[['id', 'title', 'description', 'type']],
        df_books[['id', 'title', 'description', 'type']],
        df_songs[['id', 'title', 'description', 'type']]
    ])

    # Guardar el DataFrame combinado para futuras ejecuciones
    combined_sample_df.to_pickle(combined_filename)
    print("Datos combinados guardados con éxito en", combined_filename)
    return combined_sample_df


# Función para precalcular y guardar embeddings
def calcular_y_guardar_embeddings(data, model_name='paraphrase-MiniLM-L6-v2', filename='embeddingsWE.pkl'):
    # Si el archivo de embeddings ya existe, simplemente cargarlo
    if os.path.exists(filename):
        print("Cargando embeddings desde el archivo existente")
        with open(filename, 'rb') as f:
            return pickle.load(f)

    print("Calculando y guardando nuevos embeddings")

    model = SentenceTransformer(model_name)

    # Eliminar filas con descripciones vacías o null
    data = data[data['description'].notna() & (data['description'] != '')]

    data = data.copy()
    data['embedding'] = data['description'].apply(lambda x: model.encode(x))

    # Guardar los embeddings precalculados en un archivo
    with open(filename, 'wb') as f:
        pickle.dump(data[['id', 'title', 'embedding', 'type']], f)

    print("Embeddings guardados con éxito!")
    # print("Embeddings (primeras 5 filas):")
    # print(data.head())
    return data[['id', 'title', 'embedding', 'type']]


def recomendar_con_embeddings(titulo, data):

    titulo = titulo.strip().lower()

    # Verificar si el título está en la base de datos
    if titulo not in data['title'].str.lower().values:
        print(f"El título '{titulo}' no se encuentra en la base de datos.")
        return [], [], []

    # Obtener el embedding del título buscado
    target_embedding = data[data['title'].str.lower() == titulo]['embedding'].values[0]

    # Calcular la similitud de coseno con todos los demás embeddings
    data['similarity'] = data['embedding'].apply(lambda x: cosine_similarity([x], [target_embedding]).item())

    recommended_movies = data[(data['type'] == 'movie') & (data['title'].str.lower() != titulo)]
    recommended_books = data[(data['type'] == 'book') & (data['title'].str.lower() != titulo)]
    recommended_songs = data[(data['type'] == 'song') & (data['title'].str.lower() != titulo)]

    # Ordenar por similitud y recomendar los 10 más cercanos
    recommended_movies = recommended_movies.sort_values('similarity', ascending=False).head(10)
    recommended_books = recommended_books.sort_values('similarity', ascending=False).head(10)
    recommended_songs = recommended_songs.sort_values('similarity', ascending=False).head(10)

    return (recommended_movies[['title', 'similarity']],
            recommended_books[['title', 'similarity']],
            recommended_songs[['title', 'similarity']])


def sistema_evaluacion(embeddings_df, train_ratings, test_ratings, top_k=10):
    print("INICIANDO EVALUACIÓN")

    # Convertir IDs a strings consistentemente
    embeddings_df['id'] = embeddings_df['id'].astype(str)
    train_ratings['item_id'] = train_ratings['item_id'].astype(str)
    test_ratings['item_id'] = test_ratings['item_id'].astype(str)

    print("\nESTADÍSTICAS INICIALES:")
    print(f"Usuarios únicos en train: {train_ratings['user_id'].nunique()}")
    print(f"Usuarios únicos en test: {test_ratings['user_id'].nunique()}")
    print(f"Ítems únicos en embeddings: {len(embeddings_df)}")
    print(f"Ratings positivos en test (>=4): {len(test_ratings[test_ratings['rating'] >= 4])}")

    item_embeddings = {row['id']: row['embedding'] for _, row in embeddings_df.iterrows()}
    all_items = set(item_embeddings.keys())

    # Preparar datos de usuarios
    user_history = train_ratings.groupby('user_id')['item_id'].apply(
        lambda x: [i for i in x if i in item_embeddings]).to_dict()
    user_history = {k: v for k, v in user_history.items() if len(v) > 0}

    test_positives = test_ratings[test_ratings['rating'] >= 4].groupby('user_id')['item_id'].apply(list).to_dict()

    debug_stats = {
        'users_processed': 0,
        'users_with_valid_data': 0,
        'positive_items_processed': 0,
        'positive_items_with_embedding': 0,
        'cases_with_enough_negatives': 0,
        'successful_evals': 0,
        'recall_hits': 0
    }
    debug_stats['recall3_hits'] = 0

    total_recall = 0.0
    total_auc = 0.0
    total_recall3 = 0.0

    # Evaluación por usuario
    evaluable_users = set(test_positives.keys()) & set(user_history.keys())
    print(f"\nUsuarios evaluables (aparecen en train y test): {len(evaluable_users)}")

    for user_id in tqdm(evaluable_users, desc="Procesando usuarios"):
        debug_stats['users_processed'] += 1
        train_items = user_history[user_id]
        positives = test_positives[user_id]

        # Debug:
        if debug_stats['users_processed'] == 1:
            print("\nEJEMPLO DE PRIMER USUARIO:")
            print(f"User ID: {user_id}")
            print(f"Ítems en train: {len(train_items)}")
            print(f"Ítems positivos en test: {len(positives)}")
            print(f"Primer ítem positivo: {positives[0]}")

        # Calcular embedding de usuario
        user_embedding = np.mean([item_embeddings[item] for item in train_items], axis=0)
        if np.isnan(user_embedding).any():
            continue

        debug_stats['users_with_valid_data'] += 1

        # Procesar cada ítem positivo del usuario
        for positive_item in positives:
            debug_stats['positive_items_processed'] += 1

            if positive_item not in item_embeddings:
                continue
            debug_stats['positive_items_with_embedding'] += 1

            # Seleccionar negativos
            negative_candidates = list(all_items - set(train_items) - {positive_item})
            if len(negative_candidates) < 10:
                continue
            debug_stats['cases_with_enough_negatives'] += 1

            negatives = np.random.choice(negative_candidates, 10, replace=False)
            eval_items = [positive_item] + list(negatives)

            # Calcular similitudes
            similarities = []
            for item in eval_items:
                sim = cosine_similarity([user_embedding], [item_embeddings[item]]).item()
                similarities.append(sim)

            if debug_stats['positive_items_processed'] == 1:
                print("\nEJEMPLO DE CÁLCULO PARA PRIMER ÍTEM:")
                print(f"Ítem positivo: {positive_item}")
                print(f"10 ítems negativos: {negatives[:3]}...")
                print(f"Similitudes: {[round(s, 3) for s in similarities[:3]]}...")
                print(f"Posición del positivo en ranking: {np.argsort(similarities)[::-1].tolist().index(0) + 1}")

            ranked_items = [eval_items[i] for i in np.argsort(similarities)[::-1]]

            # Recall@1
            recall_at_1 = 1 if ranked_items[0] == positive_item else 0
            total_recall += recall_at_1
            debug_stats['recall_hits'] += recall_at_1

            # Cálculo de Recall@3
            recall_at_3 = 1 if positive_item in ranked_items[:3] else 0
            total_recall3 += recall_at_3
            debug_stats['recall3_hits'] += recall_at_3

            # AUC
            y_true = [1] + [0] * 10
            y_score = similarities
            if len(set(y_true)) > 1:
                total_auc += roc_auc_score(y_true, y_score)
                debug_stats['successful_evals'] += 1

    print("ESTADÍSTICAS DE EJECUCIÓN:")
    for k, v in debug_stats.items():
        print(f"- {k}: {v}")

    print("\nDISTRIBUCIÓN DE PÉRDIDAS:")
    print(f"Usuarios sin datos válidos: {debug_stats['users_processed'] - debug_stats['users_with_valid_data']}")
    print(
        f"Ítems positivos sin embedding: {debug_stats['positive_items_processed'] - debug_stats['positive_items_with_embedding']}")
    print(
        f"Casos sin suficientes negativos: {debug_stats['positive_items_with_embedding'] - debug_stats['cases_with_enough_negatives']}")

    avg_recall = total_recall / debug_stats['successful_evals'] if debug_stats['successful_evals'] > 0 else 0
    avg_auc = total_auc / debug_stats['successful_evals'] if debug_stats['successful_evals'] > 0 else 0
    # Cálculo de promedio para Recall@3
    avg_recall3 = total_recall3 / debug_stats['successful_evals'] if debug_stats['successful_evals'] > 0 else 0

    print("RESULTADOS FINALES:")
    print(f"Evaluaciones exitosas: {debug_stats['successful_evals']}")
    print(f"Recall@1: {avg_recall:.4f} (aciertos: {debug_stats['recall_hits']}/{debug_stats['successful_evals']})")
    print(f"Recall@3: {avg_recall3:.4f} (aciertos: {debug_stats['recall3_hits']}/{debug_stats['successful_evals']})")
    print(f"AUC: {avg_auc:.4f}")

    return avg_recall, avg_auc


if __name__ == "__main__":

    books_train = pd.read_csv('Books_train.csv', header=None, names=["item_id", "user_id", "rating"])
    books_test = pd.read_csv('Books_test.csv', header=None, names=["item_id", "user_id", "rating"])
    digital_train = pd.read_csv('Digital_Music_train.csv', header=None, names=["item_id", "user_id", "rating"])
    digital_test = pd.read_csv('Digital_Music_test.csv', header=None, names=["item_id", "user_id", "rating"])
    movies_train = pd.read_csv('MoviesandTV_train.csv', header=None, names=["item_id", "user_id", "rating"])
    movies_test = pd.read_csv('MoviesandTV_test.csv', header=None, names=["item_id", "user_id", "rating"])

    combined_train = pd.concat([books_train, digital_train, movies_train])
    combined_test = pd.concat([books_test, digital_test, movies_test])
    all_ratings = pd.concat([combined_train, combined_test])

    # Obtener item_id únicos presentes en ratings
    valid_item_ids = set(all_ratings['item_id'].astype(str))

    # Cargar solo metadata de ítems que están en ratings
    movie_csv = '/home/jupyter-tfg2425multidomini-163ed/meta_Movies_and_TV.csv'
    book_csv = '/home/jupyter-tfg2425multidomini-163ed/meta_Books.csv'
    song_csv = '/home/jupyter-tfg2425multidomini-163ed/meta_Digital_Music.csv'


    def cargar_filtrado(csv_path, item_ids):
        chunks = pd.read_csv(
            csv_path,
            header=None,
            names=["id", "main_category", "title", "price", "description"],
            chunksize=100000,
            low_memory=False
        )
        filtered_chunks = []
        for chunk in chunks:
            chunk = chunk[chunk['id'].astype(str).isin(item_ids)]
            filtered_chunks.append(chunk)
        return pd.concat(filtered_chunks) if filtered_chunks else pd.DataFrame()


    df_movies = cargar_filtrado(movie_csv, valid_item_ids)
    df_books = cargar_filtrado(book_csv, valid_item_ids)
    df_songs = cargar_filtrado(song_csv, valid_item_ids)

    # Añadir tipo y combinar
    df_movies['type'] = 'movie'
    df_books['type'] = 'book'
    df_songs['type'] = 'song'

    combined_sample_df = pd.concat([
        df_movies[['id', 'title', 'description', 'type']],
        df_books[['id', 'title', 'description', 'type']],
        df_songs[['id', 'title', 'description', 'type']]
    ])

    print(f"Ítems cargados en metadata: {len(combined_sample_df)} (de {len(valid_item_ids)} ítems en ratings)")

    # Verificar ítems faltantes
    items_perdidos = valid_item_ids - set(combined_sample_df['id'].astype(str))
    if items_perdidos:
        print(f"{len(items_perdidos)} ítems en ratings no tienen metadata (ej: {list(items_perdidos)[:5]})")
    else:
        print("Todos los ítems en ratings tienen metadata asociada.")

    embeddings_df = calcular_y_guardar_embeddings(combined_sample_df)

    # Ejemplo de recomendación
    titulo = "One Perfect Wedding"
    rec_movies, rec_books, rec_songs = recomendar_con_embeddings(titulo, embeddings_df)

    # Evaluación 
    recall, auc = sistema_evaluacion(embeddings_df, combined_train, combined_test)
