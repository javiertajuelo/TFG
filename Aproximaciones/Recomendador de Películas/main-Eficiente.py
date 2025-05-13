import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from sklearn.model_selection import KFold
from sklearn.metrics import ndcg_score
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack
import numpy as np
import pickle
import os
import psutil
from tqdm import tqdm


def weighted_rating(v, R, m, C):
    return (v / (v + m)) * R + (m / (v + m)) * C


def ratingRecommendator():
    C = dataSet['vote_average'].mean()
    m = dataSet['vote_count'].quantile(0.8)
    peliculasMasVotadas = dataSet.copy().loc[dataSet['vote_count'] >= m]
    peliculasMasVotadas['score'] = peliculasMasVotadas.apply(
        lambda x: weighted_rating(x['vote_count'], x['vote_average'], m, C), axis=1
    )
    print(peliculasMasVotadas[['title', 'vote_count', 'vote_average', 'score']].head(10))


def popularRecommendator():
    popularPeliculas = dataSet.copy()
    popularPeliculas = popularPeliculas.sort_values('popularity', ascending=False)
    print(popularPeliculas[['title', 'popularity']].head(10))


def get_director(crew):
    for person in crew:
        if person['job'] == 'Director':
            return person['name']
    return 0


def get_list(obj):
    if isinstance(obj, list):
        names = [i['name'] for i in obj]
        return names[3]
    return []


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace("", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace("", ""))
        else:
            return ''


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


def sistema_evaluacion(dataSet, ratings_train=None, ratings_test=None, n_folds=5, top_k=5, eval_mode=True):

    # Limpieza  de datos
    cantidad_inicial = len(dataSet)
    dataSet = dataSet.dropna(subset=['description'])
    dataSet = dataSet[dataSet['description'].astype(str).str.strip() != '']
    print(f"Se eliminaron {cantidad_inicial - len(dataSet)} filas con descripciones inválidas")

    if len(dataSet) == 0:
        raise ValueError("No hay datos válidos después de la limpieza")

    # Configuración optimizada de CountVectorizer
    count = CountVectorizer(
        stop_words='english',
        max_features=10000,
        max_df=0.8,
        min_df=5,
        binary=True
    )

    # Procesamiento por lotes
    batch_size = 1000
    embeddings_list = []

    # Ajustar el vectorizador con una muestra
    muestra = dataSet['description'].sample(min(1000, len(dataSet)), random_state=42).astype(str)
    count.fit(muestra)

    for i in range(0, len(dataSet), batch_size):
        batch = dataSet['description'].iloc[i:i + batch_size].astype(str)
        try:
            batch_embeddings = count.transform(batch)
            embeddings_list.append(batch_embeddings)
        except Exception as e:
            print(f"Error procesando lote {i}: {str(e)}")
            continue

    if not embeddings_list:
        raise ValueError("No se pudo procesar ningún lote de datos")

    # Combinar matrices dispersas
    count_matrix = vstack(embeddings_list, format='csr')

    # Normalización L2 para calcular similitud coseno eficientemente
    from sklearn.preprocessing import normalize
    normalized_embeddings = normalize(count_matrix, norm='l2', axis=1)

    embeddings_data = {
        'embeddings': normalized_embeddings,
        'titles': dataSet['title'].values,
        'ids': dataSet['id'].values,
        'vectorizer': count
    }

    with open('embeddingsRB.pkl', 'wb') as f:
        pickle.dump(embeddings_data, f)
    print("Embeddings normalizados guardados en embeddingsRB.pkl")

    indices = pd.Series(dataSet.index, index=dataSet['title']).drop_duplicates()

    if not eval_mode:
        return normalized_embeddings, indices

    # Evaluación con matrices dispersas
    if ratings_train is not None and ratings_test is not None:
        dataSet_eval = dataSet.copy()
        dataSet_eval['embedding_idx'] = range(len(dataSet_eval))

        # Convertir IDs a strings
        dataSet_eval['id'] = dataSet_eval['id'].astype(str)
        ratings_train['item_id'] = ratings_train['item_id'].astype(str)
        ratings_test['item_id'] = ratings_test['item_id'].astype(str)

        # Mapeo de ID a índice
        item_to_id = {row['id']: row['embedding_idx'] for _, row in dataSet_eval.iterrows()}
        all_items = set(item_to_id.keys())

        # Preparar historial de usuarios
        user_history = ratings_train.groupby('user_id')['item_id'].apply(
            lambda x: [i for i in x if i in item_to_id]).to_dict()
        user_history = {k: v for k, v in user_history.items() if len(v) > 0}

        test_positives = ratings_test[ratings_test['ratings'] >= 4].groupby('user_id')['item_id'].apply(list).to_dict()

        # Contadores para métricas
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

        evaluable_users = set(test_positives.keys()) & set(user_history.keys())
        print(f"\nUsuarios evaluables: {len(evaluable_users)}")

        for user_id in tqdm(evaluable_users, "Procesando Usuarios:"):
            debug_stats['users_processed'] += 1
            train_items = user_history[user_id]
            positives = test_positives[user_id]

            # Obtener índices de los ítems del usuario
            user_item_indices = [item_to_id[item] for item in train_items if item in item_to_id]
            if not user_item_indices:
                continue

            # Calcular embedding promedio del usuario
            user_embedding = normalized_embeddings[user_item_indices].mean(axis=0)
            user_embedding = np.asarray(user_embedding).reshape(1, -1)
            user_embedding = normalize(user_embedding, norm='l2')

            debug_stats['users_with_valid_data'] += 1

            for positive_item in positives:
                debug_stats['positive_items_processed'] += 1

                if positive_item not in item_to_id:
                    continue
                debug_stats['positive_items_with_embedding'] += 1

                positive_id = item_to_id[positive_item]

                # Seleccionar negativos
                negative_candidates = list(all_items - set(train_items) - {positive_item})
                if len(negative_candidates) < 10:
                    continue
                debug_stats['cases_with_enough_negatives'] += 1

                negatives = np.random.choice(negative_candidates, 10, replace=False)
                negative_indices = [item_to_id[item] for item in negatives]

                # Índices para evaluación (1 positivo + 10 negativos)
                eval_indices = [positive_id] + negative_indices

                # Calcular similitudes
                eval_embeddings = normalized_embeddings[eval_indices]
                similarities = cosine_similarity(eval_embeddings, user_embedding).flatten()

                # Calcular métricas
                ranked_items = [eval_indices[i] for i in np.argsort(similarities)[::-1]]

                recall_at_1 = 1 if ranked_items[0] == positive_id else 0
                total_recall += recall_at_1
                debug_stats['recall_hits'] += recall_at_1

                recall_at_3 = 1 if positive_id in ranked_items[:3] else 0
                total_recall3 += recall_at_3
                debug_stats['recall3_hits'] += recall_at_3

                y_true = [1] + [0] * 10
                y_score = similarities
                if len(set(y_true)) > 1:
                    total_auc += roc_auc_score(y_true, y_score)
                    debug_stats['successful_evals'] += 1

        # Calcular métricas finales
        avg_recall = total_recall / debug_stats['successful_evals'] if debug_stats['successful_evals'] > 0 else 0
        avg_recall3 = total_recall3 / debug_stats['successful_evals'] if debug_stats['successful_evals'] > 0 else 0
        avg_auc = total_auc / debug_stats['successful_evals'] if debug_stats['successful_evals'] > 0 else 0

        print("\nRESULTADOS FINALES:")
        print(f"- Recall@1: {avg_recall:.4f}")
        print(f"- Recall@3: {avg_recall3:.4f}")
        print(f"- AUC: {avg_auc:.4f}")
        print("\nDEBUG STATS:")
        print(f"Usuarios procesados: {debug_stats['users_processed']}")
        print(f"Usuarios con datos válidos: {debug_stats['users_with_valid_data']}")
        print(f"Ítems positivos procesados: {debug_stats['positive_items_processed']}")
        print(f"Ítems positivos con embedding: {debug_stats['positive_items_with_embedding']}")
        print(f"Casos con suficientes negativos: {debug_stats['cases_with_enough_negatives']}")
        print(f"Evaluaciones exitosas: {debug_stats['successful_evals']}")
        print(f"Aciertos Recall@1: {debug_stats['recall_hits']}")
        print(f"Aciertos Recall@3: {debug_stats['recall3_hits']}")

        metricas = {'avg_recall': avg_recall, 'avg_recall3': avg_recall3, 'avg_auc': avg_auc}
        return normalized_embeddings, indices, metricas

    return normalized_embeddings, indices


def get_recomendaciones(title, normalized_embeddings, indices, dataSet, top_n=10):
    if title not in indices:
        print(f"El título '{title}' no se encuentra en la base de datos.")
        return []

    idx = indices[title]
    sim_scores = normalized_embeddings.dot(normalized_embeddings[idx].T).toarray().flatten()

    similar_indices = np.argsort(-sim_scores)[1:top_n + 1]

    return dataSet['title'].iloc[similar_indices]


def userRecommendations(ratings):
    reader = Reader()
    data = Dataset.load_from_df(ratings, reader)
    svd = SVD()
    print("\nEvaluando modelo con validación cruzada...")
    cv_results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    print("\nResultados por fold:")
    for i in range(5):
        print(f"\nFold {i + 1}:")
        print(f"  RMSE: {cv_results['test_rmse'][i]:.4f}")
        print(f"  MAE: {cv_results['test_mae'][i]:.4f}")

    print("\nResultados de la evaluación")
    print(f"RMSE promedio {cv_results['test_rmse'].mean():.4f}")
    print(f"MAE promedio {cv_results['test_mae'].mean():.4f}")
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    print(
        f"Para el usuario 1 la puntuación estimada para la película con ID 302 es de (sobre 5): {svd.predict(1, 302, 3)}")


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    ratings_test = pd.read_csv('MoviesandTV_test.csv', names=['item_id', 'user_id', 'ratings'])
    ratings_train = pd.read_csv('MoviesandTV_train.csv', names=['item_id', 'user_id', 'ratings'])
    ratings_val = pd.read_csv('MoviesandTV_val.csv', names=['item_id', 'user_id', 'ratings'])
    ratings = pd.concat([ratings_train, ratings_test, ratings_val])

    # Obtener los item_id únicos presentes en ratings
    valid_item_ids = set(ratings['item_id'].astype(str)) 

    # Cargar solo las filas de metadata cuyos IDs están en ratings
    # Usar chunks si el metadata es muy grande
    chunks = pd.read_csv(
        '/home/jupyter-tfg2425multidomini-163ed/meta_Movies_and_TV.csv',
        names=['id', 'main_category', 'title', 'price', 'description'],
        chunksize=100000,
        low_memory=False
    )
    dataSet = pd.concat([chunk[chunk['id'].astype(str).isin(valid_item_ids)] for chunk in chunks])

    print(f"Ítems cargados en metadata: {len(dataSet)} (de {len(valid_item_ids)} ítems en ratings)")

    # Verificar que no hay ítems perdidos
    items_perdidos = valid_item_ids - set(dataSet['id'].astype(str))
    if items_perdidos:
        print(f"{len(items_perdidos)} ítems en ratings no tienen metadata (ej: {list(items_perdidos)[:5]})")
    else:
        print("Todos los ítems en ratings tienen metadata asociada.")

    normalized_embeddings, indices, metricas = sistema_evaluacion(dataSet, ratings_train, ratings_test,
                                                                           eval_mode=True)

    titulo = 'Gold Fever'
    if titulo in indices:
        recomendaciones = get_recomendaciones(titulo, normalized_embeddings, indices, dataSet)
        if not recomendaciones.empty:
            print(f"Se encontraron las siguientes recomendaciones sobre {titulo}:\n{recomendaciones}")
        else:
            print(f"No se encontraron recomendaciones para '{titulo}'.")
    else:
        print(f"El título '{titulo}' no se encuentra en la base de datos.")

    userRecommendations(ratings)


if __name__ == "__main__":
    main()
