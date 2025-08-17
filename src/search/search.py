import os
import math
import cv2
import h5py
import faiss
import numba
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance
from tkinter import filedialog as fd
from typing import Literal, Optional, Tuple

from src.features.hashing import phash, color_hash
from src.utils.disk_drive import get_data_folder, get_drive_letter, get_images_folder
from src.features.histograms import extract_lab_pair_histogram
from src.features.kmeans import extract_colors_kmeans


def preprocess_image(img):
    """Identische Bildvorverarbeitung wie in hashing.py."""
    if img is None:
        return None
    
    h, w = img.shape[:2]
    target_pixels = 65536
    scale_factor = np.sqrt(target_pixels / (h * w))
    
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # Resize with area interpolation for downscaling
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Consistent JPEG compression
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
    _, compressed = cv2.imencode('.jpg', resized, encode_params)
    preprocessed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    
    return preprocessed


def load_and_preprocess_image(img_path: str):
    """Load image and apply consistent preprocessing."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    return preprocess_image(img)


def load_image_for_display(img_path: str):
    """Load image for matplotlib display without preprocessing."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_nearest_hashes(path: str, mode: str = "pHash", n: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Find similar images using hash indices."""
    data_folder = get_data_folder()
    
    if mode == "pHash":
        index_path = os.path.join(data_folder, "phash_index_1024_1024.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"pHash index not found: {index_path}")
        
        index = faiss.read_index_binary(index_path)
        preprocessed = load_and_preprocess_image(path)
        query_hash = phash(preprocessed, phash_bits=1024)
        
    elif mode == "colorHash":
        index_path = os.path.join(data_folder, "color_index_1024_1024.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Color hash index not found: {index_path}")
        
        index = faiss.read_index_binary(index_path)
        preprocessed = load_and_preprocess_image(path)
        query_hash = color_hash(preprocessed, color_bits=1024)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Convert to binary format for FAISS
    query_binary = np.packbits(query_hash.reshape(1, -1), axis=1)
    
    # Search
    distances, indices = index.search(query_binary, n + 1)  # +1 in case query image is in index
    
    # Remove exact match if it exists (distance = 0)
    mask = distances[0] > 0
    if mask.any():
        return distances[0][mask][:n], indices[0][mask][:n]
    else:
        # If no non-zero distances (shouldn't happen), return all
        return distances[0][:n], indices[0][:n]


def compare_color_distributions(colors1: np.ndarray, counts1: np.ndarray, 
                               colors2: np.ndarray, counts2: np.ndarray) -> float:
    """Compare color distributions using Wasserstein distance."""
    distances = []
    
    for dim in range(colors1.shape[1]):  # For each color channel
        try:
            dist = wasserstein_distance(
                colors1[:, dim], colors2[:, dim],
                u_weights=counts1, v_weights=counts2
            )
            distances.append(dist)
        except Exception as e:
            print(f"Error in Wasserstein distance calculation: {e}")
            distances.append(float('inf'))
    
    return np.mean(distances)


def find_nearest_kmeans(image_path: str, n: int = 5, n_jobs: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Find similar images using K-means color clustering."""
    preprocessed = load_and_preprocess_image(image_path)
    pixels = preprocessed.reshape(-1, 3)
    target_colors, target_counts = extract_colors_kmeans(pixels, n=5)
    
    kmeans_file = os.path.join(get_data_folder(), "kmeans_5_hsv.h5")
    if not os.path.exists(kmeans_file):
        raise FileNotFoundError(f"K-means file not found: {kmeans_file}")
    
    with h5py.File(kmeans_file, 'r') as f:
        ids = f["ids"][()]
        colors = f["colors"][()]
        counts = f["counts"][()]
    
    def compute_distance(i: int) -> float:
        try:
            return compare_color_distributions(
                target_colors, target_counts, colors[i], counts[i]
            )
        except Exception:
            return float('inf')
    
    # Parallel distance computation
    distances = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(compute_distance)(i) for i in range(len(ids))
    )
    
    distances = np.array(distances)
    
    # Find top n results
    valid_mask = ~np.isinf(distances)
    if not valid_mask.any():
        raise ValueError("No valid distances computed")
    
    valid_indices = np.where(valid_mask)[0]
    valid_distances = distances[valid_indices]
    
    n_results = min(n, len(valid_distances))
    top_local_indices = np.argpartition(valid_distances, n_results)[:n_results]
    top_indices = valid_indices[top_local_indices]
    
    # Sort by distance
    sorted_order = np.argsort(distances[top_indices])
    final_indices = top_indices[sorted_order]
    final_distances = distances[final_indices]
    
    return ids[final_indices], final_distances


@numba.jit(nopython=True, parallel=True)
def bhattacharyya_batch(query: np.ndarray, all_hists: np.ndarray) -> np.ndarray:
    """Compute Bhattacharyya distances in parallel."""
    n_samples = all_hists.shape[0]
    n_features = query.shape[0]
    distances = np.empty(n_samples, dtype=np.float32)
    
    for i in numba.prange(n_samples):
        bc = 0.0
        for j in range(n_features):
            bc += math.sqrt(max(query[j] * all_hists[i, j], 1e-15))
        
        bc = max(bc, 1e-10)
        distances[i] = -math.log(bc)
    
    return distances


def find_nearest_histograms(image_path: str, n: int = 5, mode: str = "full") -> Tuple[np.ndarray, np.ndarray]:
    """Find similar images using color histograms."""
    hist_file = os.path.join(get_data_folder(), "histograms_32_hsv.h5")
    if not os.path.exists(hist_file):
        raise FileNotFoundError(f"Histogram file not found: {hist_file}")
    
    with h5py.File(hist_file, "r") as f:
        if mode == "accelerated":
            # Use color hash to pre-filter candidates
            try:
                _, candidate_ids = find_nearest_hashes(image_path, "colorHash", 5000)
                
                all_ids = f["ids"][()]
                # Find indices of candidates in histogram file
                indices = np.where(np.isin(all_ids, candidate_ids))[0]
                
                if len(indices) < n:
                    # Fallback to full search if not enough candidates
                    indices = np.arange(len(all_ids))
                    ids = all_ids
                else:
                    ids = all_ids[indices]
                
                histograms = f["histograms"][indices]
                
            except Exception as e:
                print(f"Accelerated search failed, falling back to full search: {e}")
                ids = f["ids"][()]
                histograms = f["histograms"][()]
        else:
            # Full search
            ids = f["ids"][()]
            histograms = f["histograms"][()]
    
    # Compute query histogram
    preprocessed = load_and_preprocess_image(image_path)
    _, query_hist = extract_lab_pair_histogram(None, preprocessed, bins=32)
    
    if query_hist is None:
        raise ValueError("Could not compute histogram for query image")
    
    # Warmup numba
    if len(histograms) > 0:
        _ = bhattacharyya_batch(query_hist, histograms[:4])
    
    # Compute distances
    scores = bhattacharyya_batch(query_hist, histograms)
    
    # Find top n results
    n_results = min(n, len(scores))
    top_indices = np.argpartition(scores, n_results)[:n_results]
    top_ids = ids[top_indices]
    top_scores = scores[top_indices]
    
    # Sort by score (lowest first)
    sort_order = np.argsort(top_scores)
    return top_ids[sort_order], top_scores[sort_order]


def get_paths_from_db(ids: np.ndarray) -> list:
    """Get image paths from database."""
    db_path = os.path.join(get_data_folder(), "images.db")
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return [None] * len(ids)
    
    try:
        with sqlite3.connect(db_path) as db:
            cursor = db.cursor()
            paths = []
            
            for img_id in ids:
                cursor.execute("SELECT path FROM images WHERE id_image = ?", (int(img_id),))
                result = cursor.fetchone()
                if result:
                    paths.append(result[0])
                else:
                    print(f"Path not found for ID: {img_id}")
                    paths.append(None)
            
            return paths
    except Exception as e:
        print(f"Database error: {e}")
        return [None] * len(ids)


def display_results(query_path: str, result_ids: np.ndarray, distances: np.ndarray, method_name: str):
    """Display search results in a matplotlib figure."""
    # Get paths for result images
    result_paths = get_paths_from_db(result_ids)
    drive_letter = get_drive_letter()
    full_paths = [drive_letter + p if p else None for p in result_paths]
    
    n_results = len(result_ids)
    fig = plt.figure(figsize=(n_results * 3, 8))
    fig.suptitle(f"Image Search Results - {method_name}", fontsize=16)
    
    # Display query image (top row, spanning all columns)
    ax_query = plt.subplot2grid((2, n_results), (0, 0), colspan=n_results)
    query_img = load_image_for_display(query_path)
    if query_img is not None:
        ax_query.imshow(query_img)
    ax_query.set_title(f"Query Image\n{os.path.basename(query_path)}", fontsize=14)
    ax_query.axis('off')
    
    # Display result images (bottom row)
    for i in range(n_results):
        ax = plt.subplot2grid((2, n_results), (1, i))
        
        img_path = full_paths[i]
        if img_path and os.path.exists(img_path):
            try:
                result_img = load_image_for_display(img_path)
                if result_img is not None:
                    ax.imshow(result_img)
                else:
                    ax.text(0.5, 0.5, "Load\nError", ha='center', va='center', fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}", ha='center', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, "Image\nNot Found", ha='center', va='center', fontsize=10)
        
        ax.set_title(f"ID: {result_ids[i]}\nDist: {distances[i]:.2f}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main search interface."""
    try:
        # Select query image
        query_path = fd.askopenfilename(
            title="Select query image...", 
            initialdir=get_images_folder(),
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if not query_path:
            print("No image selected.")
            return
        
        if not os.path.exists(query_path):
            print(f"Query image not found: {query_path}")
            return
        
        print(f"Query image: {query_path}")
        print("\nAvailable search methods:")
        print("(1) Perceptual Hash")
        print("(2) Color Hash") 
        print("(3) Color Histogram (full)")
        print("(4) Color Histogram (accelerated)")
        print("(5) K-Means Color Clustering")
        
        try:
            choice = int(input("\nEnter method number (1-5): "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            return
        
        n_results = 5
        
        try:
            if choice == 1:
                distances, ids = find_nearest_hashes(query_path, mode="pHash", n=n_results)
                method_name = "Perceptual Hash"
                
            elif choice == 2:
                distances, ids = find_nearest_hashes(query_path, mode="colorHash", n=n_results)
                method_name = "Color Hash"
                
            elif choice == 3:
                ids, distances = find_nearest_histograms(query_path, n=n_results, mode="full")
                method_name = "Color Histogram (Full)"
                
            elif choice == 4:
                ids, distances = find_nearest_histograms(query_path, n=n_results, mode="accelerated")
                method_name = "Color Histogram (Accelerated)"
                
            elif choice == 5:
                ids, distances = find_nearest_kmeans(query_path, n=n_results)
                method_name = "K-Means Color Clustering"
                
            else:
                print("Invalid choice. Please select 1-5.")
                return
            
            print(f"\nSearch completed using {method_name}")
            print(f"Found {len(ids)} similar images")
            
            # Display results
            display_results(query_path, ids, distances, method_name)
            
        except Exception as e:
            print(f"Search failed: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()