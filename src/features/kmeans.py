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


def extract_colors_kmeans(pixels: np.ndarray, n: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Extract dominant colors using K-means clustering with improved stability."""
    # Validate input
    if pixels is None or len(pixels) == 0:
        return np.zeros((n, 3), dtype=np.float32), np.zeros(n, dtype=np.int32)
    
    # Ensure pixels is 2D array
    if len(pixels.shape) == 3:
        pixels = pixels.reshape(-1, 3)
    
    # Handle case with insufficient pixels
    if len(pixels) < n:
        centers = np.zeros((n, 3), dtype=np.float32)
        counts = np.zeros(n, dtype=np.int32)
        
        # Fill with available pixel data
        unique_pixels = np.unique(pixels, axis=0)
        n_unique = min(len(unique_pixels), n)
        centers[:n_unique] = unique_pixels[:n_unique]
        counts[0] = len(pixels)  # All pixels belong to first cluster
        
        return centers, counts
    
    try:
        # Use MiniBatchKMeans for better performance and stability
        kmeans = MiniBatchKMeans(
            n_clusters=n, 
            batch_size=min(8192, len(pixels) // 4),
            n_init=3,
            max_iter=100,  # Increased for better convergence
            random_state=42,  # Fixed seed for reproducibility
            reassignment_ratio=0.01,  # Better cluster stability
            max_no_improvement=10
        )
        
        # Fit the model
        labels = kmeans.fit_predict(pixels.astype(np.float32))
        
        # Get cluster centers and counts
        centers = kmeans.cluster_centers_
        counts = np.bincount(labels, minlength=n)
        
        # Sort by count (most frequent first)
        order = np.argsort(counts)[::-1]
        centers_sorted = centers[order]
        counts_sorted = counts[order]
        
        # Ensure exactly n clusters (pad if necessary)
        if len(centers_sorted) < n:
            padded_centers = np.zeros((n, 3), dtype=np.float32)
            padded_counts = np.zeros(n, dtype=np.int32)
            padded_centers[:len(centers_sorted)] = centers_sorted
            padded_counts[:len(counts_sorted)] = counts_sorted
            return padded_centers, padded_counts
        
        return centers_sorted.astype(np.float32), counts_sorted.astype(np.int32)
        
    except Exception as e:
        print(f"K-means clustering failed: {e}")
        # Return fallback result
        centers = np.zeros((n, 3), dtype=np.float32)
        counts = np.zeros(n, dtype=np.int32)
        if len(pixels) > 0:
            centers[0] = np.mean(pixels, axis=0)
            counts[0] = len(pixels)
        return centers, counts


def process_image(img_id: int, img_bytes: np.ndarray, colorspace: int, n_colors: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """Process single image with consistent preprocessing."""
    try:
        # Decode image
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return img_id, None, None
        
        # Apply consistent preprocessing
        preprocessed = preprocess_image(img)
        if preprocessed is None:
            return img_id, None, None
        
        # Convert colorspace
        if colorspace != cv2.COLOR_BGR2BGR:  # Skip if already in target colorspace
            img_converted = cv2.cvtColor(preprocessed, colorspace)
        else:
            img_converted = preprocessed
        
        # Extract pixels
        pixels = img_converted.reshape(-1, 3)
        
        # Extract dominant colors
        colors, counts = extract_colors_kmeans(pixels, n_colors)
        
        return img_id, colors, counts
        
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        return img_id, None, None


def batch_to_h5(h5_path: str, batch_data: Tuple, start_idx: int, end_idx: int, lock: Lock):
    """Write batch to HDF5 file with thread safety."""
    colors_batch, counts_batch, ids_batch = batch_data
    
    with lock:
        try:
            with h5py.File(h5_path, "r+") as f:
                f['colors'][start_idx:end_idx] = colors_batch
                f['counts'][start_idx:end_idx] = counts_batch
                f['ids'][start_idx:end_idx] = ids_batch
                f.attrs["count"] = end_idx
                f.flush()
        except Exception as e:
            print(f"Error writing batch {start_idx}-{end_idx}: {e}")
            raise
    
    # Clean up batch data
    del colors_batch, counts_batch, ids_batch


def create_output_h5(output_path: str, total_images: int, n_colors: int, 
                    colorspace: str, source_file: str):
    """Create output HDF5 file with proper structure."""
    if not os.path.exists(output_path):
        try:
            with h5py.File(output_path, "w") as f:
                f.create_dataset("colors", (total_images, n_colors, 3), 
                               dtype=np.float32, compression='gzip', compression_opts=6)
                f.create_dataset("counts", (total_images, n_colors), 
                               dtype=np.int32, compression='gzip', compression_opts=6)
                f.create_dataset("ids", (total_images,), dtype=np.int64)
                f.attrs.update({
                    "count": 0,
                    "n_colors": n_colors,
                    "colorspace": colorspace,
                    "source_file": source_file,
                    "preprocessing": "standardized",
                    "kmeans_algorithm": "MiniBatchKMeans"
                })
                
            print(f"Created output file: {output_path}")
            print(f"Configuration: {n_colors} colors per image, {colorspace} colorspace")
        except Exception as e:
            print(f"Error creating output file: {e}")
            raise
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract dominant colors using K-means with consistent preprocessing")
    parser.add_argument("h5_path", help="Path to input HDF5 with images")
    parser.add_argument("--n", type=int, default=5, 
                       help="Number of dominant colors to extract")
    parser.add_argument("--colorspace", choices=["lab", "hsv", "rgb"], default="rgb",
                       help="Color space for K-means clustering")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for processing")
    parser.add_argument("--workers", type=int, default=12,
                       help="Number of worker threads")
    
    args = parser.parse_args()
    
    # Map colorspace names to OpenCV constants
    colorspace_map = {
        "lab": cv2.COLOR_BGR2LAB, 
        "hsv": cv2.COLOR_BGR2HSV, 
        "rgb": cv2.COLOR_BGR2RGB
    }
    colorspace_cv = colorspace_map[args.colorspace]
    
    # Create output file path
    output_path = os.path.join(
        os.path.dirname(args.h5_path),
        f"kmeans_{args.n}_{args.colorspace}.h5"
    )
    
    # Check input file
    if not os.path.exists(args.h5_path):
        print(f"Input file not found: {args.h5_path}")
        return
    
    # Load metadata
    with h5py.File(args.h5_path, 'r') as f:
        total_images = f.attrs["count"]
        has_ids = 'ids' in f
        print(f"Input file contains {total_images} images")
        print(f"Input file has ID mapping: {has_ids}")
    
    # Check for existing progress
    already_processed = 0
    if os.path.exists(output_path):
        try:
            with h5py.File(output_path, 'r') as f:
                already_processed = f.attrs["count"]
                existing_n = f.attrs.get("n_colors", 0)
                existing_colorspace = f.attrs.get("colorspace", "")
                
                if existing_n != args.n or existing_colorspace != args.colorspace:
                    print(f"Warning: Existing file has different parameters")
                    print(f"Existing: n_colors={existing_n}, colorspace={existing_colorspace}")
                    print(f"Requested: n_colors={args.n}, colorspace={args.colorspace}")
                    response = input("Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        return
            
            print(f"Already processed: {already_processed}/{total_images}")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            already_processed = 0
    else:
        create_output_h5(output_path, total_images, args.n, args.colorspace, args.h5_path)
    
    if already_processed >= total_images:
        print("All images already processed!")
        return
    
    # Processing setup
    write_lock = Lock()
    images_remaining = total_images - already_processed
    
    print(f"Processing {images_remaining} remaining images...")
    print(f"Using {args.workers} workers, batch size {args.batch_size}")
    
    # Process images in batches
    with h5py.File(args.h5_path, 'r') as input_file:
        with ThreadPoolExecutor(max_workers=1) as write_executor:
            write_futures = []
            
            with tqdm(total=images_remaining, desc="Extracting dominant colors") as pbar:
                for start in range(already_processed, total_images, args.batch_size):
                    end = min(start + args.batch_size, total_images)
                    batch_size = end - start
                    
                    # Load batch data
                    images_bytes = input_file["jpeg_bytes"][start:end]
                    
                    # Load or generate IDs
                    if has_ids:
                        ids_batch = input_file["ids"][start:end]
                    else:
                        ids_batch = np.arange(start, end, dtype=np.int64)
                    
                    # Process batch in parallel
                    batch_colors = []
                    batch_counts = []
                    batch_ids_final = []
                    
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        futures = [
                            executor.submit(process_image, img_id, img_bytes, 
                                          colorspace_cv, args.n)
                            for img_id, img_bytes in zip(ids_batch, images_bytes)
                        ]
                        
                        for future in futures:
                            img_id, colors, counts = future.result()
                            batch_ids_final.append(img_id)
                            
                            if colors is not None and counts is not None:
                                batch_colors.append(colors)
                                batch_counts.append(counts)
                            else:
                                # Create zero arrays for failed images
                                batch_colors.append(np.zeros((args.n, 3), dtype=np.float32))
                                batch_counts.append(np.zeros(args.n, dtype=np.int32))
                    
                    # Prepare batch data
                    colors_array = np.stack(batch_colors)
                    counts_array = np.stack(batch_counts)
                    ids_array = np.array(batch_ids_final, dtype=np.int64)
                    
                    # Submit write task
                    write_future = write_executor.submit(
                        batch_to_h5, 
                        output_path, 
                        (colors_array, counts_array, ids_array), 
                        start, 
                        end, 
                        write_lock
                    )
                    write_futures.append(write_future)
                    
                    pbar.update(batch_size)
                    
                    # Memory management
                    if (start - already_processed) % (args.batch_size * 20) == 0:
                        # Wait for some writes to complete
                        completed = []
                        for i, future in enumerate(write_futures):
                            if future.done():
                                try:
                                    future.result()
                                    completed.append(i)
                                except Exception as e:
                                    print(f"Write error: {e}")
                        
                        # Remove completed futures
                        for i in reversed(completed):
                            write_futures.pop(i)
                        
                        gc.collect()
            
            # Wait for all writes to complete
            print("Waiting for remaining writes to complete...")
            for future in write_futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Final write error: {e}")
    
    # Verify final result
    try:
        with h5py.File(output_path, 'r') as f:
            final_count = f.attrs["count"]
            print(f"Processing completed! Final count: {final_count}/{total_images}")
            
            if final_count == total_images:
                print("✓ All images processed successfully")
                
                # Show some statistics
                sample_colors = f['colors'][:min(10, final_count)]
                sample_counts = f['counts'][:min(10, final_count)]
                
                print(f"Sample color statistics:")
                print(f"  Average dominant color: {np.mean(sample_colors[:, 0], axis=0):.1f}")
                print(f"  Average pixel count: {np.mean(sample_counts[:, 0]):.0f}")
            else:
                print(f"⚠ Warning: Expected {total_images}, got {final_count}")
                
    except Exception as e:
        print(f"Error verifying results: {e}")
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()