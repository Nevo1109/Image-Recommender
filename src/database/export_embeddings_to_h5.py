import sqlite3, numpy as np, h5py
from tqdm import tqdm

def main():
    db_path = r"C:\Users\kilic\OneDrive\Desktop\db\image_recommender.db"
    export_path = r"C:\Users\kilic\OneDrive\Desktop\db\vit_b_16_embeddings_no_compr.h5"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id_image, embedding FROM vit_b_16_embeddings")
    rows = cursor.fetchall()
    ids = np.array([row[0] for row in rows], dtype=np.int32)
    embeddings = np.stack([
        np.frombuffer(row[1], dtype='float32') for row in tqdm(rows)
    ])

    with h5py.File(export_path, "w") as f:
        f.create_dataset("ids", data=ids)
        f.create_dataset("embeddings", data=embeddings, compression=None)

    print("Export abgeschlossen.")

if __name__ == "__main__":
    main()
