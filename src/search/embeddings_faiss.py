import sqlite3
import numpy as np
import h5py
from PIL import Image
from torchvision import models
import torch
import faiss

EXAMPLE_IMAGE_PATH = r"C:\Users\kilic\OneDrive\Desktop\example.jpg"
DB_PATH = r"C:\Users\kilic\OneDrive\Desktop\db\image_recommender.db"
H5_PATH = r"C:\Users\kilic\OneDrive\Desktop\db\vit_b_16_embeddings(no_compr).h5"
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
    print("Lade Embeddings...")
    ids, embeddings = load_embeddings(H5_PATH)

    print("Berechne Embedding für Eingabebild...")
    query_emb = compute_embedding(EXAMPLE_IMAGE_PATH)

    print("Suche Top ähnliche Bilder mit Faiss...")
    top_matches = find_similar_images_faiss(query_emb, embeddings, ids, top_n=TOP_N)

    print("\nTop-ähnliche Bilder:")
    id_to_path = load_paths_from_db(DB_PATH, [img_id for img_id, _ in top_matches])
    for rank, (img_id, score) in enumerate(top_matches, start=1):
        path = id_to_path.get(img_id, "Pfad nicht gefunden")
        print(f"{rank}. ID: {img_id} | Score: {score:.4f} | Pfad: {path}")

if __name__ == "__main__":
    main()
