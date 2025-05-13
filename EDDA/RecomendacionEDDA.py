import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import traceback
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import threading

# Cargar diccionarios
TOTAL_USER_ID_DICT = pickle.load(open("total_user_id_dict.pkl", "rb"))
TRAIN_RECORDS = pickle.load(open("train_records.pkl", "rb"))
ITEM_ID_DICT = pickle.load(open("item_id_dict.pkl", "rb"))
ID_ITEM_DICT = pickle.load(open("id_item_dict.pkl", "rb"))
ITEM_DICT = pickle.load(open("item_dict.pkl", "rb"))


def show_about_dialog():
    messagebox.showinfo("Acerca de", "Sistema Recomendador Multidominio v1.0")


def load_model(model_path="DDALG_Amazon_5core.pth"):
    return torch.load(model_path, map_location=torch.device('cpu'))


def load_product_info(product_data_path="data/combinedmetaData.csv"):
    return pd.read_csv(product_data_path, header=None,
                       names=["parent_asin", "main_category", "title", "price", "description"])


def get_embeddings(data):
    min_users = min(data[f"embedding_user.{i}.weight"].shape[0] for i in range(3))
    min_items = min(data[f"embedding_item.{i}.weight"].shape[0] for i in range(3))
    user_intra_embeddings = torch.cat([data[f"embedding_user.{i}.weight"][:min_users] for i in range(3)], dim=1)
    item_intra_embeddings = torch.cat([data[f"embedding_item.{i}.weight"][:min_items] for i in range(3)], dim=1)
    user_inter_embeddings = data["aggr_user.weight"][:min_users]
    item_inter_embeddings = data["aggr_item.weight"][:min_items]
    user_embeddings = torch.cat([user_intra_embeddings, user_inter_embeddings], dim=1).numpy()
    item_embeddings = torch.cat([item_intra_embeddings, item_inter_embeddings], dim=1).numpy()
    return user_embeddings, item_embeddings


def recommend():

    def generate_recommendations_thread():
        progress.start()
        threading.Thread(target=generate_recommendations).start()

    def generate_recommendations():
        try:
            output_text.config(state=tk.NORMAL)
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, "Cargando modelo y datos...\n")
            output_text.config(state=tk.DISABLED)

            product_info_df = load_product_info()
            data = load_model()

            output_text.config(state=tk.NORMAL)
            output_text.insert(tk.END, "Obteniendo embeddings...\n")
            output_text.config(state=tk.DISABLED)

            user_embeddings, item_embeddings = get_embeddings(data)

            user_input = user_id_entry.get().strip()
            try:
                user_index = int(user_input)
                if user_index < 0 or user_index >= user_embeddings.shape[0]:
                    raise ValueError(f"Índice de usuario fuera de rango: {user_index}")
            except ValueError:
                if user_input in TOTAL_USER_ID_DICT:
                    user_index = TOTAL_USER_ID_DICT[user_input]
                else:
                    raise ValueError(f"ID de usuario desconocido: {user_input}")

            output_text.config(state=tk.NORMAL)
            output_text.insert(tk.END, "Calculando recomendaciones...\n")
            output_text.config(state=tk.DISABLED)

            n_per_category = int(n_recs_entry.get())
            selected_domain = domain_cb.get()

            # Seleccionar dominios a mostrar
            if selected_domain == "Todos":
                domains_to_show = list(ITEM_DICT.keys())
            else:
                domains_to_show = [selected_domain] if selected_domain in ITEM_DICT else []

            # Similitud usuario-ítems
            user_vec = user_embeddings[user_index:user_index+1]
            user_vector = cosine_similarity(user_vec, item_embeddings).flatten()
            num_items = user_vector.shape[0]

            top_items = {}
            for dom in domains_to_show:

                # Historial del entrenamiento
                history_asins = set()
                if dom in TRAIN_RECORDS and user_index in TRAIN_RECORDS[dom]:
                    history_asins.update(TRAIN_RECORDS[dom][user_index])
                    output_text.insert(tk.END, "TRAIN_RECORDS\n")

                # Ids de items del entrenamiento
                history_ids = []
                for idx in sorted(history_asins):
                    asin_hist = ID_ITEM_DICT[dom].get(idx)
                    if asin_hist:
                        history_ids.append(asin_hist)

                # Listado de índices válidos y no vistos
                idxs = []
                for asin in ITEM_DICT[dom]:
                    if asin in ITEM_ID_DICT[dom]:
                        idx = ITEM_ID_DICT[dom][asin]
                        if idx < num_items and idx not in history_asins:
                            idxs.append(idx)

                # Ordenar por similitud
                idxs_sorted = sorted(idxs, key=lambda i: user_vector[i], reverse=True)

                picks = []
                for i in idxs_sorted:
                    if len(picks) >= n_per_category:
                        break
                    asin = ID_ITEM_DICT[dom][i]
                    row = product_info_df[product_info_df['parent_asin'] == asin]
                    if row.empty or pd.isna(row['title'].iloc[0]):
                        continue
                    picks.append((asin, float(user_vector[i])))
                top_items[dom] = picks

            # Salida por pantalla de los resultados
            result_text = "\n**Productos Recomendados:**\n"
            for dom, picks in top_items.items():
                result_text += f"\nDominio: {dom}\n"
                for asin, score in picks:
                    prod = product_info_df[product_info_df['parent_asin'] == asin].iloc[0]
                    result_text += f"  - ID: {asin} (Score: {score:.4f})\n"
                    result_text += f"      * Nombre: {prod['title']}\n"
                    result_text += f"      * Precio: {prod['price']}\n"

            output_text.config(state=tk.NORMAL)
            output_text.insert(tk.END, result_text)
            output_text.config(state=tk.DISABLED)

        except Exception as e:
            output_text.config(state=tk.NORMAL)
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, f"Error: {str(e)}\n\n{traceback.format_exc()}")
            output_text.config(state=tk.DISABLED)
        finally:
            progress.stop()

    # GUI
    root = tk.Tk()
    root.title("Recomendador Multidominio")
    root.geometry("700x600")

    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('Header.TLabel', font=('Arial', 18, 'bold'))
    style.configure('Recom.TButton', font=('Arial', 12, 'bold'), padding=6)

    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Salir", command=root.quit)
    menubar.add_cascade(label="Archivo", menu=file_menu)
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="Acerca de", command=show_about_dialog)
    menubar.add_cascade(label="Ayuda", menu=help_menu)
    root.config(menu=menubar)

    ttk.Label(root, text="Sistema Recomendador Multidominio", style='Header.TLabel').pack(pady=20)

    input_frame = ttk.Frame(root, padding=10)
    input_frame.pack(fill='x')

    ttk.Label(input_frame, text="ID de Usuario:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    user_id_entry = ttk.Entry(input_frame, width=30)
    user_id_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(input_frame, text="Nº Recomendaciones:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    n_recs_entry = ttk.Entry(input_frame, width=30)
    n_recs_entry.grid(row=1, column=1, padx=5, pady=5)

    ttk.Label(input_frame, text="Dominio:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    domain_cb = ttk.Combobox(input_frame, state='readonly')
    domain_cb['values'] = ["Todos"] + list(ITEM_DICT.keys())
    domain_cb.current(0)
    domain_cb.grid(row=2, column=1, padx=5, pady=5)

    ttk.Button(root, text="Obtener Recomendaciones", style='Recom.TButton', command=generate_recommendations_thread).pack(pady=10)
    progress = ttk.Progressbar(root, mode='indeterminate')
    progress.pack(fill='x', padx=20)

    output_frame = ttk.LabelFrame(root, text="Resultados", padding=10)
    output_frame.pack(fill='both', expand=True, padx=20, pady=10)
    output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state=tk.DISABLED)
    output_text.pack(fill='both', expand=True)

    root.mainloop()

if __name__ == "__main__":
    recommend()