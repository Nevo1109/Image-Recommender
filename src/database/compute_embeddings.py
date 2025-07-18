import os, sqlite3, torch, numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, set_image_backend
from utils.dataset import ImageDataset, safe_collate
from tqdm import tqdm

set_image_backend("accimage")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_embeddings_to_db(indices, embeddings, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("BEGIN")
    cursor.executemany(
        "INSERT OR REPLACE INTO vit_b_16_embeddings (id_image, embedding) VALUES (?, ?)",
        [(int(i), sqlite3.Binary(emb.tobytes())) for i, emb in zip(indices, embeddings)]
    )
    conn.commit()
    conn.close()

def main(batch_size=32):
    db_path = r"C:\Users\kilic\OneDrive\Desktop\db\image_recommender.db"

    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = models.vit_b_16(weights=weights)
    model.heads = torch.nn.Identity()
    model.eval().to(device)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vit_b_16_embeddings (
            id_image INTEGER PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY (id_image) REFERENCES images(id_image)
        )
    """)
    conn.commit()

    cursor.execute("""
        SELECT id_image, path FROM images
        WHERE id_image NOT IN (
            SELECT id_image FROM vit_b_16_embeddings
        )
    """)
    image_data = cursor.fetchall()
    conn.close()

    dataset = ImageDataset(indexed_paths=image_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate,
        persistent_workers=True
    )

    for indices, images in tqdm(dataloader, total=len(dataloader)):
        if indices is None or images is None:
            continue
        images = images.to(device)
        with torch.no_grad():
            embeddings = model(images).cpu().numpy()
        save_embeddings_to_db(indices, embeddings, db_path)

    print("Embeddings gespeichert.")

if __name__ == "__main__":
    main()
