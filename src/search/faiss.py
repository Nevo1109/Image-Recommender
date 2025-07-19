import sqlite3
import numpy as np
import h5py
from PIL import Image
from torchvision import models, transforms
import torch
import faiss

EXAMPLE_IMAGE_PATH = r"C:\Users\kilic\OneDrive\Desktop\example.jpg"
DB_PATH = r"C:\Users\kilic\OneDrive\Desktop\db\image_recommender.db"
H5_PATH = r"C:\Users\kilic\OneDrive\Desktop\db\vit_b_16_embeddings(no_compr).h5"
TOP_N = 5

# Modell & Transformationen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
model = models.vit_b_16(weights=weights)
model.heads = torch.nn.Identity()
model.eval().to(device)
transform = weights.transforms()

# Eingabebild → Embedding
def compute_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).cpu().numpy().squeeze()
    return embedding.astype("float32")

# Lade Embeddings aus H5-Datei
def load_embeddings(h5_path):
    with h5py.File(h5_path, "r") as f:
        ids = f["ids"][:]
        embeddings = f["embeddings"][:]
    return ids, embeddings.astype("float32")

# Hole Pfade aus SQLite DB
def load_paths_from_db(db_path, ids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT id_image, path FROM images WHERE id_image IN ({','.join('?' * len(ids))})"
    cursor.execute(query, list(ids))
    id_to_path = dict(cursor.fetchall())
    conn.close()
    return id_to_path

# Suche Top-N mit FAISS
def find_similar_images_faiss(query_embedding, all_embeddings, all_ids, top_n=5):
    # Normalisiere Embeddings → Cosine-Similarity
    faiss.normalize_L2(all_embeddings)
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    index = faiss.IndexFlatIP(all_embeddings.shape[1])
    index.add(all_embeddings)

    scores, indices = index.search(query_embedding.reshape(1, -1), top_n)
    results = [(all_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
    return results

def main():
    print("Lade Embeddings...")
    ids, embeddings = load_embeddings(H5_PATH)

    print("Berechne Embedding für Eingabebild...")
    query_emb = compute_embedding(EXAMPLE_IMAGE_PATH)

    print("Suche ähnliche Bilder mit FAISS...")
    top_matches = find_similar_images_faiss(query_emb, embeddings, ids, top_n=TOP_N)

    print("\nTop-ähnliche Bilder:")
    id_to_path = load_paths_from_db(DB_PATH, [id for id, _ in top_matches])
    for idx, (img_id, sim) in enumerate(top_matches, start=1):
        path = id_to_path.get(img_id, "Pfad nicht gefunden")
        print(f"{idx}. ID: {img_id} | Score: {sim:.4f} | Pfad: {path}")

if __name__ == "__main__":
    main()
