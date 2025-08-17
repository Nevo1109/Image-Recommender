import pickle
import h5py
import pandas as pd

import umap.umap_ as umap
from sklearn.decomposition import PCA

from src.utils.disk_drive import get_data_folder

def save_umap(ids, embeddings, folder):
    pca_path = folder + "\\pca_model_vitb16.pkl"
    umap_path = folder + "\\umap_model_vitb16.pkl"
    
    pca = PCA(64).fit(embeddings)
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    pc = pca.transform(embeddings)
    
    umap_model = umap.UMAP(50, 3, verbose=True).fit(pc)
    with open(umap_path, "wb") as f:
        pickle.dump(umap_model, f)
    
    points = umap_model.transform(pc)
    df = pd.DataFrame({
        "ids": ids,
        "umap_x": points[:, 0],
        "umap_y": points[:, 1],
        "umap_z": points[:, 2]
    })
    df.to_csv(folder + "\\umap_vitb16_points.csv", index=False)
        
def main():
    default_folder = get_data_folder()
    with h5py.File(default_folder + "\\vit_b_16_embeddings.h5", "r") as f:
        ids = f["ids"][()]
        embeddings = f["embeddings"][()]
        
    save_umap(ids, embeddings, default_folder)
    
if __name__ == "__main__":
    main()