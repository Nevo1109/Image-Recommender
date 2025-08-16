import argparse
import os
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from threading import Lock

import cv2
import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def extract_colors_kmeans(pixels: np.ndarray, n: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Extract dominant colors using K-means."""
    # ensure enough pixels for n clusters
    if len(pixels) < n:
        # pad with zeros if not enough pixels
        centers = np.zeros((n, 3))
        counts = np.zeros(n)
        if len(pixels) > 0:
            centers[0] = pixels[0]
            counts[0] = len(pixels)
        return centers, counts
    
    kmeans = MiniBatchKMeans(
        n_clusters=n, 
        batch_size=8192, 
        n_init=3,
        max_iter=10, 
        random_state=0
    ).fit(pixels)
    
    counts = np.bincount(kmeans.labels_)
    order = np.argsort(counts)[::-1]
    
    centers = kmeans.cluster_centers_[order]
    counts_ordered = counts[order]
    
    # ensure exactly n clusters
    if len(centers) < n:
        padded_centers = np.zeros((n, 3))
        padded_counts = np.zeros(n)
        padded_centers[:len(centers)] = centers
        padded_counts[:len(counts_ordered)] = counts_ordered
        return padded_centers, padded_counts
    
    return centers, counts_ordered


def process_image(img_bytes: np.ndarray, colorspace: int, n_colors: int):
    """Process single image."""
    try:
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        
        img = cv2.cvtColor(img, colorspace)
        pixels = img.reshape(-1, 3)
        
        return extract_colors_kmeans(pixels, n_colors)
    except:
        return None, None


def batch_to_h5(h5_path: str, batch_data: Tuple, start_idx: int, end_idx: int, lock: Lock):
    """Write batch to HDF5 file."""
    colors_batch, counts_batch = batch_data
    
    with lock:
        with h5py.File(h5_path, "r+") as f:
            f['colors'][start_idx:end_idx] = colors_batch
            f['counts'][start_idx:end_idx] = counts_batch
            f.attrs["count"] = end_idx
            f.flush()
    
    del colors_batch, counts_batch


def create_output_h5(output_path: str, total_images: int, n_colors: int, colorspace: str, source_file: str):
    """Create HDF5 file"""
    if not os.path.exists(output_path):
        with h5py.File(output_path, "w") as f:
            f.create_dataset("colors", (total_images, n_colors, 3), dtype=np.float32)
            f.create_dataset("counts", (total_images, n_colors), dtype=np.int32)
            f.attrs.update({
                "count": 0,
                "n_colors": n_colors,
                "colorspace": colorspace,
                "source_file": source_file
            })
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--colorspace", choices=["lab", "hsv", "rgb"], default="hsv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    
    args = parser.parse_args()
    
    colorspaces = {"lab": cv2.COLOR_BGR2LAB, "hsv": cv2.COLOR_BGR2HSV, "rgb": cv2.COLOR_BGR2RGB}
    colorspace = colorspaces[args.colorspace]
    
    # Create output file path
    output_path = os.path.join(
        os.path.dirname(args.h5_path),
        f"kmeans_{args.n}_{args.colorspace}.h5"
    )
    
    # Check for existing progress
    already_processed = 0
    with h5py.File(args.h5_path, 'r') as f:
        total_images = f.attrs["count"]
    
    if os.path.exists(output_path):
        with h5py.File(output_path, 'r') as f:
            already_processed = f.attrs["count"]
    else:
        create_output_h5(output_path, total_images, args.n, args.colorspace, args.h5_path)
        print(f"Created new output file for {total_images} images")
    
    if already_processed >= total_images:
        print("All images already processed!")
        return
    
    # Thread synchronization
    write_lock = Lock()
    
    print(f"Processing remaining {total_images - already_processed} images")
    
    with h5py.File(args.h5_path, 'r') as input_file:
        with ThreadPoolExecutor(max_workers=1) as write_executor:
            write_futures = []
            
            with tqdm(total=total_images - already_processed, 
                     desc="Processing images") as pbar:
                
                for start in range(already_processed, total_images, args.batch_size):
                    end = min(start + args.batch_size, total_images)
                    batch_size = end - start
                    
                    # Load batch
                    images_bytes = input_file["jpeg_bytes"][start:end]
                    
                    # Process batch in parallel
                    batch_centers = []
                    batch_counts = []
                    
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        futures = [
                            executor.submit(process_image, img_bytes, colorspace, args.n)
                            for img_bytes in images_bytes
                        ]
                        
                        for future in futures:
                            centers, counts = future.result()
                            if centers is not None:
                                batch_centers.append(centers)
                                batch_counts.append(counts)
                            else:
                                batch_centers.append(np.zeros((args.n, 3)))
                                batch_counts.append(np.zeros(args.n))
                    
                    # Prepare batch data
                    colors_array = np.stack(batch_centers)
                    counts_array = np.stack(batch_counts)
                    
                    # Submit write task
                    write_future = write_executor.submit(
                        batch_to_h5, 
                        output_path, 
                        (colors_array, counts_array), 
                        start, 
                        end, 
                        write_lock
                    )
                    write_futures.append(write_future)
                    
                    pbar.update(batch_size)
                    
                    # Memory cleanup
                    if (start - already_processed) % (args.batch_size * 10) == 0:
                        # Wait for some writes to complete
                        completed_futures = []
                        for i, future in enumerate(write_futures):
                            if future.done():
                                future.result()  # Get result to handle exceptions
                                completed_futures.append(i)
                        
                        # Remove completed futures
                        for i in reversed(completed_futures):
                            write_futures.pop(i)
                        
                        gc.collect()
            
            # Wait for all writes to complete
            for future in write_futures:
                future.result()
    
    print(f"Completed! Results saved to {output_path}")


if __name__ == "__main__":
    main()