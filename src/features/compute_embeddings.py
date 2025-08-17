import sqlite3, torch, h5py
from torch.utils.data import DataLoader
from torchvision import models, set_image_backend
from tqdm import tqdm

from src.utils.dataset import ImageDataset, safe_collate
from src.utils.disk_drive import get_data_folder

set_image_backend("accimage")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_embeddings_to_h5(indices, embeddings, h5_path):
    with h5py.File(h5_path, "w-") as f:
        f.create_dataset("ids", data=indices)
        f.create_dataset("embeddings", data=embeddings)

def main(batch_size=32):
    db_path = get_data_folder() + "\\image_recommender.db"
    h5_path = get_data_folder() + "\\vit_b_16_embeddings.h5"

    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = models.vit_b_16(weights=weights)
    model.heads = torch.nn.Identity()
    model.eval().to(device)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT i.id_image, i.path FROM images as i
        INNER JOIN metadata as m ON m.id_image = i.id_image
        WHERE m.size IS NOT NULL
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
        save_embeddings_to_h5(indices, embeddings, h5_path)

    print("Embeddings gespeichert.")

if __name__ == "__main__":
    main()
