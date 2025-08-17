import sqlite3
import numpy as np
import h5py
import torch
import faiss
from PIL import Image
from torchvision import models
from tkinter import filedialog as fd

from src.utils.disk_drive import get_data_folder, get_images_folder
from src.search.search import preprocess

DB_PATH = get_data_folder() + "\\image_recommender.db"
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

# === PFADE AUS DER DB LADEN ===
def load_paths_from_db(db_path, ids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT id_image, path FROM images WHERE id_image IN ({','.join('?' * len(ids))})"
    cursor.execute(query, list(ids))
    id_to_path = dict(cursor.fetchall())
    conn.close()
    return id_to_path

# === ÄHNLICHSTE BILDER FINDEN MIT FAISS ===
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
        
    print("Lade Embeddings...")
    ids, embeddings = load_embeddings(H5_PATH)

    print("Berechne Embedding für Eingabebild...")
    if len(paths) == 1:
        query_emb = compute_embedding(paths[0])
    if len(paths) == 2:
        a = input("Mit wie viel Prozent soll das erste Bild gewichtet werden?\n")
        a = float(a) / 100
        query_emb = a * compute_embedding(paths[0]) + (1 - a) * compute_embedding(paths[1])
        
    print("Suche Top ähnliche Bilder mit Faiss...")
    top_matches = find_similar_images_faiss(query_emb, embeddings, ids, top_n=TOP_N)

    print("\nTop-ähnliche Bilder:")
    id_to_path = load_paths_from_db(DB_PATH, [img_id for img_id, _ in top_matches])
    for rank, (img_id, score) in enumerate(top_matches, start=1):
        path = id_to_path.get(img_id, "Pfad nicht gefunden")
        print(f"{rank}. ID: {img_id} | Score: {score:.4f} | Pfad: {path}")

if __name__ == "__main__":
    main()
