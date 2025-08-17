import os

from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sqlite3
import numpy as np
import h5py
import torch
import faiss
import cv2
from PIL import Image
from torchvision import models
from tkinter import filedialog as fd

from src.utils.disk_drive import get_data_folder, get_drive_letter, get_images_folder

DB_PATH = get_data_folder() + "\\images.db"
H5_PATH = get_data_folder() + "\\vit_b_16_embeddings.h5"
TOP_N = 5

# === MODEL & TRANSFORM ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
model = models.vit_b_16(weights=weights)
model.heads = torch.nn.Identity()
model.eval().to(device)

transform = weights.transforms()

# === EMBEDDING FUNKTION ===
def compute_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).cpu().numpy().squeeze()
    return embedding.astype('float32')  # Faiss braucht float32

# === EMBEDDINGS LADEN ===
def load_embeddings(h5_path):
    with h5py.File(h5_path, "r") as f:
        ids = f["ids"][:]
        embeddings = f["embeddings"][:].astype('float32')
    return ids, embeddings

# === PFADE AUS DER DB LADEN (FIXED VERSION) ===
def load_paths_from_db(db_path, bild_ids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Konvertiere numpy.int32 zu normalen Python ints
    bild_ids = [int(id) for id in bild_ids]
    
    placeholder = ",".join(["?"] * len(bild_ids))
    query = f"SELECT id_image, path FROM images WHERE id_image IN ({placeholder})"
    cursor.execute(query, bild_ids)
    results = dict(cursor.fetchall())
    
    conn.close()
    return results

# === HILFSFUNKTIONEN FÜR BILDANZEIGE ===
def get_paths_from_db(img_ids):
    """Lade nur die Pfade aus der DB (ohne Debug-Output)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Konvertiere numpy.int32 zu normalen Python ints
    img_ids = [int(id) for id in img_ids]
    
    placeholder = ",".join(["?"] * len(img_ids))
    query = f"SELECT id_image, path FROM images WHERE id_image IN ({placeholder})"
    cursor.execute(query, img_ids)
    results = dict(cursor.fetchall())
    conn.close()
    
    # Gib Pfade in der richtigen Reihenfolge zurück
    return [results.get(img_id) for img_id in img_ids]

def load_image_for_display(img_path):
    """Lade ein Bild für die Anzeige"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

def display_results(query_paths: list, result_ids: np.ndarray, distances: np.ndarray):
    """Display search results in a matplotlib figure."""
    # Get paths for result images
    result_paths = get_paths_from_db(result_ids)
    drive_letter = get_drive_letter()
    full_paths = [drive_letter + p if p else None for p in result_paths]
    
    n_results = len(result_ids)
    n_queries = len(query_paths)
    
    # Besseres Layout: 2 Zeilen, query oben, results unten
    fig, axes = plt.subplots(2, max(n_queries, n_results), figsize=(15, 8))
    fig.suptitle("Image Search Results - FAISS", fontsize=16, y=0.95)
    
    # Falls nur eine Spalte, mache axes zu 2D array
    if max(n_queries, n_results) == 1:
        axes = axes.reshape(-1, 1)
    
    # Query-Bilder in der oberen Zeile
    for i in range(max(n_queries, n_results)):
        ax = axes[0, i]
        if i < n_queries:
            query_img = load_image_for_display(query_paths[i])
            if query_img is not None:
                ax.imshow(query_img)
                ax.set_title(f"Query {i+1}\n{os.path.basename(query_paths[i])}", 
                           fontsize=10, pad=10)
            else:
                ax.text(0.5, 0.5, "Query\nLoad Error", ha='center', va='center', fontsize=12)
                ax.set_title(f"Query {i+1}", fontsize=10)
        else:
            # Leere Felder ausblenden
            ax.axis('off')
            continue
        ax.axis('off')
    
    # Result-Bilder in der unteren Zeile
    for i in range(max(n_queries, n_results)):
        ax = axes[1, i]
        if i < n_results:
            img_path = full_paths[i]
            if img_path and os.path.exists(img_path):
                try:
                    result_img = load_image_for_display(img_path)
                    if result_img is not None:
                        ax.imshow(result_img)
                    else:
                        ax.text(0.5, 0.5, "Load\nError", ha='center', va='center', fontsize=10)
                        ax.set_facecolor('lightgray')
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{str(e)[:15]}", ha='center', va='center', fontsize=8)
                    ax.set_facecolor('lightcoral')
            else:
                ax.text(0.5, 0.5, "Image\nNot Found", ha='center', va='center', fontsize=10)
                ax.set_facecolor('lightgray')
            
            # Bessere Titel-Formatierung
            score_text = f"#{i+1} | ID: {result_ids[i]}\nScore: {distances[i]:.3f}"
            ax.set_title(score_text, fontsize=9, pad=8)
        else:
            # Leere Felder ausblenden
            ax.axis('off')
            continue
        ax.axis('off')
    
    # Layout anpassen
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    plt.show()

def find_similar_images_faiss(query_embedding, all_embeddings, all_ids, top_n=5):
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2-Distanz
    index.add(all_embeddings)  # Index mit allen Embeddings füllen

    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, top_n)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        img_id = all_ids[idx]
        score = 1 / (1 + dist)  # inverser Abstand als Score (optional)
        results.append((img_id, score))
    return results

def main():
    paths = fd.askopenfilenames(title="Select one or two images", initialdir=get_images_folder())[:2]
        
    ids, embeddings = load_embeddings(H5_PATH)

    if len(paths) == 1:
        query_emb = compute_embedding(paths[0])
    if len(paths) == 2:
        a = input("Mit wie viel Prozent soll das erste Bild gewichtet werden?\n")
        a = float(a) / 100
        query_emb = a * compute_embedding(paths[0]) + (1 - a) * compute_embedding(paths[1])
        
    top_matches = find_similar_images_faiss(query_emb, embeddings, ids, top_n=TOP_N)

    ähnlichste_ids = [img_id for img_id, _ in top_matches]
    pfade = load_paths_from_db(DB_PATH, ähnlichste_ids)
    
    for rank, (img_id, score) in enumerate(top_matches, start=1):
        path = pfade.get(img_id, "Pfad nicht gefunden")
        print(f"{rank}. ID: {img_id} | Score: {score:.4f} | Pfad: {path}")
    
    # === BILDANZEIGE ===
    result_ids = np.array(ähnlichste_ids)
    distances = np.array([score for _, score in top_matches])
    display_results(paths, result_ids, distances)

if __name__ == "__main__":
    main()