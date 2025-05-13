import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

N = 200000
N2 = 1000000


def cargar_datos(movie_csv, book_csv, song_csv, combined_filename='combinedDataSetCV.pkl'):
    if os.path.exists(combined_filename):
        print("Cargando datos combinados desde el archivo existente")
        return pd.read_pickle(combined_filename)

    print("Cargando y combinando datasets...")
    df_movies = pd.read_csv(movie_csv, header=None, names=["id", "main_category", "title", "price", "description"],
                            nrows=N)
    df_books = pd.read_csv(book_csv, header=None, names=["id", "main_category", "title", "price", "description"],
                           nrows=N)
    df_songs = pd.read_csv(song_csv, header=None, names=["id", "main_category", "title", "price", "description"],
                           nrows=N)

    df_movies['type'] = 'movie'
    df_books['type'] = 'book'
    df_songs['type'] = 'song'

    combined_sample_df = pd.concat([
        df_movies[['id', 'title', 'description', 'type']],
        df_books[['id', 'title', 'description', 'type']],
        df_songs[['id', 'title', 'description', 'type']]
    ])

    combined_sample_df.to_pickle(combined_filename)
    print(f"Datos combinados guardados. Total de ítems: {len(combined_sample_df)}")
    return combined_sample_df


def calcular_y_guardar_embeddings(data, filename='embeddingsCV.pkl', sparse_filename='embeddings_sparse.npz'):
    if os.path.exists(filename) and os.path.exists(sparse_filename):
        print("Cargando embeddings existentes")
        with open(filename, 'rb') as f:
            data_loaded = pickle.load(f)
        embeddings_sparse = load_npz(sparse_filename)
        return data_loaded, embeddings_sparse

    print("Calculando embeddings con CountVectorizer")

    # Limpieza de datos
    initial_count = len(data)
    data = data.dropna(subset=['description'])  # Eliminar filas con null en description
    data = data[data['description'].astype(str).str.strip() != '']  # Eliminar strings vacíos
    print(f"Se eliminaron {initial_count - len(data)} filas con descripciones inválidas")

    if len(data) == 0:
        raise ValueError("No hay datos válidos después de la limpieza")

    # Configuración optimizada de CountVectorizer
    vectorizer = CountVectorizer(
        stop_words='english',
        max_features=10000,
        max_df=0.8,
        min_df=5,
        binary=True
    )

    # Procesamiento por lotes
    batch_size = 500
    embeddings_list = []

    # Primero ajustamos el vectorizador con una muestra limpia
    sample_data = data['description'].sample(min(1000, len(data)), random_state=42).astype(str)
    vectorizer.fit(sample_data)

    for i in (range(0, len(data), batch_size)):
        batch = data['description'].iloc[i:i + batch_size].astype(str)  # Asegurar que son strings
        try:
            batch_embeddings = vectorizer.transform(batch)
            embeddings_list.append(batch_embeddings)
        except Exception as e:
            print(f"Error procesando lote {i}: {str(e)}")
            continue

    if not embeddings_list:
        raise ValueError("No se pudo procesar ningún lote de datos")

    # Combinar todos los lotes en una matriz dispersa CSR
    embeddings_sparse = vstack(embeddings_list, format='csr')

    # Normalizar los embeddings
    from sklearn.preprocessing import normalize
    embeddings_sparse = normalize(embeddings_sparse, norm='l2', axis=1)

    # Guardar matriz dispersa
    save_npz(sparse_filename, embeddings_sparse)

    data = data.copy()
    data['embedding_idx'] = range(len(data))

    with open(filename, 'wb') as f:
        pickle.dump(data[['id', 'title', 'embedding_idx', 'type']], f)

    print(f"Embeddings guardados. Total de ítems procesados: {len(data)}")
    return data[['id', 'title', 'embedding_idx', 'type']], embeddings_sparse


def recomendar_con_embeddings(titulo, data, embeddings_sparse):
    titulo = titulo.strip().lower()

    if titulo not in data['title'].str.lower().values:
        print(f"El título '{titulo}' no se encuentra en la base de datos.")
        return [], [], []

    # Obtener índice del título buscado
    target_id = data[data['title'].str.lower() == titulo]['embedding_idx'].values[0]
    target_embedding = embeddings_sparse[target_id]

    # Calcular similitudes con toda la matriz dispersa
    print("Calculando similitudes...")
    similarities = cosine_similarity(embeddings_sparse, target_embedding).flatten()
    data['similarity'] = similarities

    recommended_movies = data[(data['type'] == 'movie') & (data['title'].str.lower() != titulo)]
    recommended_books = data[(data['type'] == 'book') & (data['title'].str.lower() != titulo)]
    recommended_songs = data[(data['type'] == 'song') & (data['title'].str.lower() != titulo)]

    recommended_movies = recommended_movies.sort_values('similarity', ascending=False).head(10)
    recommended_books = recommended_books.sort_values('similarity', ascending=False).head(10)
    recommended_songs = recommended_songs.sort_values('similarity', ascending=False).head(10)

    return (
        recommended_movies[['title', 'similarity']],
        recommended_books[['title', 'similarity']],
        recommended_songs[['title', 'similarity']]
    )


def sistema_evaluacion(embeddings_df, embeddings_sparse, train_ratings, test_ratings, top_k=10):
    print("INICIANDO EVALUACIÓN")
    embeddings_df['id'] = embeddings_df['id'].astype(str)
    train_ratings['item_id'] = train_ratings['item_id'].astype(str)
    test_ratings['item_id'] = test_ratings['item_id'].astype(str)

    # Crear mapeo de ID a índice
    item_to_id = {row['id']: row['embedding_idx'] for _, row in embeddings_df.iterrows()}

    print("\nESTADÍSTICAS INICIALES:")
    print(f"- Usuarios únicos en train: {train_ratings['user_id'].nunique()}")
    print(f"- Usuarios únicos en test: {test_ratings['user_id'].nunique()}")
    print(f"- Ítems únicos en embeddings: {len(embeddings_df)}")

    # Preparar datos de usuarios
    user_history = train_ratings.groupby('user_id')['item_id'].apply(
        lambda x: [i for i in x if i in item_to_id]).to_dict()

    test_positives = test_ratings[test_ratings['rating'] >= 4].groupby('user_id')['item_id'].apply(list).to_dict()

    debug_stats = {
        'users_processed': 0,
        'users_with_valid_data': 0,
        'positive_items_processed': 0,
        'positive_items_with_embedding': 0,
        'cases_with_enough_negatives': 0,
        'successful_evals': 0,
        'recall_hits': 0,
        'recall3_hits': 0
    }

    total_recall = 0.0
    total_recall3 = 0.0
    total_auc = 0.0

    # Evaluación por usuario
    evaluable_users = set(test_positives.keys()) & set(user_history.keys())
    print(f"\nUsuarios evaluables: {len(evaluable_users)}")

    for user_id in tqdm(evaluable_users, desc="Procesando usuarios"):
        debug_stats['users_processed'] += 1
        train_items = user_history[user_id]
        positives = test_positives[user_id]

        # Obtener embeddings de los ítems del usuario
        user_item_indices = [item_to_id[item] for item in train_items if item in item_to_id]
        if not user_item_indices:
            continue

        # Calcular embedding promedio del usuario
        user_embedding = embeddings_sparse[user_item_indices].mean(axis=0)
        user_embedding = np.asarray(user_embedding).reshape(1, -1)

        debug_stats['users_with_valid_data'] += 1

        # Procesar cada ítem positivo
        for positive_item in positives:
            debug_stats['positive_items_processed'] += 1

            if positive_item not in item_to_id:
                continue
            debug_stats['positive_items_with_embedding'] += 1

            positive_id = item_to_id[positive_item]

            # Seleccionar negativos
            all_items = set(item_to_id.keys())
            negative_candidates = list(all_items - set(train_items) - {positive_item})
            if len(negative_candidates) < 10:
                continue
            debug_stats['cases_with_enough_negatives'] += 1

            negatives = np.random.choice(negative_candidates, 10, replace=False)
            negative_indices = [item_to_id[item] for item in negatives]

            # Índices de todos los ítems a evaluar (1 positivo + 10 negativos)
            eval_indices = [positive_id] + negative_indices

            # Calcular similitudes
            eval_embeddings = embeddings_sparse[eval_indices]
            similarities = cosine_similarity(eval_embeddings, user_embedding).flatten()

            # Ranking y métricas
            ranked_items = [eval_indices[i] for i in np.argsort(similarities)[::-1]]

            # Recall@1
            recall_at_1 = 1 if ranked_items[0] == positive_id else 0
            total_recall += recall_at_1
            debug_stats['recall_hits'] += recall_at_1

            # Recall@3
            recall_at_3 = 1 if positive_id in ranked_items[:3] else 0
            total_recall3 += recall_at_3
            debug_stats['recall3_hits'] += recall_at_3

            # AUC
            y_true = [1] + [0] * 10
            y_score = similarities
            if len(set(y_true)) > 1:
                total_auc += roc_auc_score(y_true, y_score)
                debug_stats['successful_evals'] += 1

    print("\nESTADÍSTICAS DE EJECUCIÓN:")
    for k, v in debug_stats.items():
        print(f"- {k}: {v}")

    avg_recall = total_recall / debug_stats['successful_evals'] if debug_stats['successful_evals'] > 0 else 0
    avg_recall3 = total_recall3 / debug_stats['successful_evals'] if debug_stats['successful_evals'] > 0 else 0
    avg_auc = total_auc / debug_stats['successful_evals'] if debug_stats['successful_evals'] > 0 else 0

    print("\nRESULTADOS FINALES:")
    print(f"Recall@1: {avg_recall:.4f}")
    print(f"Recall@3: {avg_recall3:.4f}")
    print(f"AUC: {avg_auc:.4f}")

    return avg_recall, avg_recall3, avg_auc


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

    #Obtener item_id únicos presentes en ratings
    valid_item_ids = set(all_ratings['item_id'].astype(str))

    #Cargar solo metadata de ítems que están en ratings
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

    embeddings_df, embeddings_sparse = calcular_y_guardar_embeddings(combined_sample_df)

    print("\nEJEMPLO DE RECOMENDACIÓN: ")
    titulo = "One Perfect Wedding"
    print(titulo)
    rec_movies, rec_books, rec_songs = recomendar_con_embeddings(titulo, embeddings_df, embeddings_sparse)

    recall, recall3, auc = sistema_evaluacion(
        embeddings_df,
        embeddings_sparse,
        combined_train,
        combined_test,
        top_k=10
    )