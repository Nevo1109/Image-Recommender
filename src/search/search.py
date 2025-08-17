import math
import sqlite3
import cv2
import h5py
import faiss
import numba
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance
from tkinter import filedialog as fd
from typing import Literal

from src.features.hashing import phash, color_hash
from src.utils.disk_drive import get_data_folder, get_drive_letter, get_images_folder
from src.search.search import find_nearest_hashes
from src.features.histograms import extract_lab_pair_histogram
from src.features.kmeans import extract_colors_kmeans

def preprocess(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    scale_factor = 65536 / (h * w)
    resized = cv2.resize(img, (int(w * scale_factor)), int(h * scale_factor), interpolation=cv2.INTER_AREA)
    _, compressed = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
    preprocessed = cv2.imdecode(compressed)
    

def find_nearest_hashes(image_path, n=5, mode: Literal["pHash", "cHash"] = "pHash"):
    preprocessed = preprocess(image_path)
    
    if mode == "pHash":
        hash = phash(preprocessed)
        index = faiss.read_index(get_data_folder() + "\\phash_index_1024.index")
    if mode == "cHash":
        hash = color_hash(preprocessed)
        index = faiss.read_index(get_data_folder() + "\\color_index_1024.index")
        
    D, I = index.search(hash, n)
    return D, I


def compare_color_distributions(colors1, counts1, colors2, counts2):
    # Für jede Farbdimension separat
    distances = []
    for dim in range(colors1.shape[1]):  # RGB = 3 Dimensionen
        dist = wasserstein_distance(
            colors1[:, dim], colors2[:, dim],
            u_weights=counts1, v_weights=counts2
        )
        distances.append(dist)
    
    return np.mean(distances)  # oder np.linalg.norm(distances) 
    

def find_nearest_kmeans(image_path, n=5, n_jobs=6):
    preprocessed = preprocess(image_path)
    target_colors, target_counts = extract_colors_kmeans(preprocessed)
    
    with h5py.File(get_data_folder() + "\\kmeans_5_hsv.h5") as f:
        ids = f["ids"][()]
        colors = f["colors"][()]
        counts = f["counts"][()]
    
    def compute_distance(i):
        return compare_color_distributions(
            target_colors, target_counts, colors[i], counts[i]
        )
    
    distances = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(compute_distance)(i) for i in range(len(ids))
    )
    
    distances = np.array(distances)
    top_indices = np.argpartition(distances, n)[:n]
    top_indices = top_indices[np.argsort(distances[top_indices])]
    
    return ids[top_indices], distances[top_indices]
    

@numba.jit(nopython=True, parallel=True)
def bhattacharyya_batch(query, all_hists):
    """Parallelized Bhattacharyya distance calculation with Numba."""
    n_samples = all_hists.shape[0]
    n_features = query.shape[0]
    distances = np.empty(n_samples)
    
    for i in numba.prange(n_samples):  # Parallel loop
        bc = 0.0
        for j in range(n_features):
            bc += math.sqrt(query[j] * all_hists[i, j])
        
        # Avoid log(0)
        bc = max(bc, 1e-10)
        distances[i] = -math.log(bc)
    
    return distances

def find_nearest_histograms(image_path, n=5, mode="accelerated"):
    with h5py.File(get_data_folder() + "\\histograms_32_hsv.h5", "r") as f:
        if mode == "accelerated":
            _, ids = find_nearest_hashes(image_path, 5000, "cHash")
            indices = np.argwhere(np.isin(f["ids"][()], ids)).flatten()
            histograms = f["histograms"][indices]
        else:
            ids = f["ids"][()]
            histograms = f["histograms"][()]
    
    preprocessed = preprocess(image_path)
    _, img_hist = extract_lab_pair_histogram(None, preprocessed)
    bhattacharyya_batch(img_hist, histograms[:4])  # numba warmup
    scores = bhattacharyya_batch(img_hist, histograms)
    
    top_n_idcs = np.argpartition(scores, n)[:n]
    top_n_ids = ids[top_n_idcs]
    top_n_scores = scores[top_n_idcs]
    
    order = np.argsort(top_n_scores)
    return top_n_ids[order], top_n_scores[order]
    

def get_paths(ids):
    db = sqlite3.connect(get_data_folder() + "\\images.db")
    cursor = db.cursor()
    for id in ids:
        cursor.execute(f"SELECT path FROM images WHERE id_image = {id}")
    return [x[0] for x in cursor.fetchall()]

def single_image(path, score_func: callable, n=5, mode=None):
    return score_func(path, n, mode)

def main():
    path = fd.askopenfilenames(title="Select one or two images...", initialdir=get_images_folder())[0]

    print("Available Measures: (1) Perceptual Hash  (2) Color Hash  (3) Histogram  (4) Histogram, cHashs accelerated  (5) KMeans")
    choice = int(input("Enter a number: "))
    
    match choice:
        case 1:
            D, I = find_nearest_hashes(path, mode="pHash")
        case 2:
            D, I = find_nearest_hashes(path, mode="cHash")
        case 3:
            D, I = find_nearest_histograms(path, mode=None)
        case 4:
            D, I = find_nearest_histograms(path, mode="accelerated")
        case 5:
            D, I = find_nearest_kmeans(path)
    
    db = sqlite3.connect(get_data_folder() + "\\images.db")
    cursor = db.cursor()
    for id in I:
        cursor.execute(f"SELECT path FROM images WHERE id_image = {id}")
        
    paths = [get_drive_letter() + x[0] for x in cursor.fetchall()]
    fig, axes = plt.subplots(2, 5)
    
    axes[0,2].imshow(plt.imread(path))
    axes[0,2].set_title("Input")
    axes[0,2].axis('off')
    
    for i in range(1, 5):
        axes[0,i].axis('off')
    
    # Ergebnisse
    for i, (path, score) in enumerate(zip(paths, D)):
        axes[1,i].imshow(plt.imread(path))
        axes[1,i].set_title(f"{score:.3f}")
        axes[1,i].axis('off')
    
    plt.tight_layout()
    plt.show()