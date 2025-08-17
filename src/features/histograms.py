import argparse
import os
import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2
import h5py
import numpy as np
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


def extract_lab_pair_histogram(img_id, img_data, bins: int = 32, colorspace: str = "lab"):
    """Extract color histogram from image with consistent preprocessing."""
    try:
        # Handle both image bytes and numpy arrays
        if isinstance(img_data, np.ndarray) and img_data.dtype == np.uint8 and len(img_data.shape) == 1:
            # Image bytes - decode first
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        else:
            # Already decoded image
            img = img_data
        
        if img is None:
            return img_id, None
        
        # Apply consistent preprocessing
        preprocessed = preprocess_image(img)
        if preprocessed is None:
            return img_id, None
        
        # Convert to target colorspace
        colorspace_map = {
            "lab": cv2.COLOR_BGR2LAB,
            "hsv": cv2.COLOR_BGR2HSV, 
            "rgb": cv2.COLOR_BGR2RGB
        }
        
        if colorspace in colorspace_map:
            img_converted = cv2.cvtColor(preprocessed, colorspace_map[colorspace])
        else:
            img_converted = preprocessed
        
        # Calculate pairwise histograms for 3 channels
        if colorspace == "lab":
            ranges = [0, 256, 0, 256, 0, 256]  # L, A, B ranges
            channel_names = ['L', 'A', 'B']
        elif colorspace == "hsv":
            ranges = [0, 180, 0, 256, 0, 256]  # H, S, V ranges
            channel_names = ['H', 'S', 'V']
        else:  # RGB
            ranges = [0, 256, 0, 256, 0, 256]  # R, G, B ranges
            channel_names = ['R', 'G', 'B']
        
        # 2D histograms for channel pairs
        hist_01 = cv2.calcHist([img_converted], [0, 1], None, [bins, bins], 
                              [ranges[0], ranges[1], ranges[2], ranges[3]])
        hist_02 = cv2.calcHist([img_converted], [0, 2], None, [bins, bins], 
                              [ranges[0], ranges[1], ranges[4], ranges[5]])
        hist_12 = cv2.calcHist([img_converted], [1, 2], None, [bins, bins], 
                              [ranges[2], ranges[3], ranges[4], ranges[5]])
        
        # Normalize histograms
        hist_01_norm = hist_01 / (hist_01.sum() + 1e-8)
        hist_02_norm = hist_02 / (hist_02.sum() + 1e-8)
        hist_12_norm = hist_12 / (hist_12.sum() + 1e-8)
        
        # Concatenate and flatten
        combined_hist = np.concatenate([
            hist_01_norm.flatten(),
            hist_02_norm.flatten(), 
            hist_12_norm.flatten()
        ])
        
        return img_id, combined_hist.astype(np.float16)
        
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        return img_id, None


def batch_to_h5(h5_path: str, ids: np.ndarray, batch_data: np.ndarray, 
                start_idx: int, end_idx: int, lock: Lock):
    """Write batch to HDF5 with thread safety."""
    with lock:
        try:
            with h5py.File(h5_path, "r+") as f:
                f['ids'][start_idx:end_idx] = ids
                f['histograms'][start_idx:end_idx] = batch_data
                f.attrs["count"] = end_idx
                f.flush()
        except Exception as e:
            print(f"Error writing batch {start_idx}-{end_idx}: {e}")
            raise


def create_output_h5(output_path: str, total_images: int, bins: int, 
                    colorspace: str, source_file: str):
    """Create output HDF5 file with proper structure."""
    if not os.path.exists(output_path):
        try:
            with h5py.File(output_path, "w") as f:
                # 3 pairs of 2D histograms, each bins x bins
                hist_size = bins * bins * 3
                
                f.create_dataset("ids", (total_images,), dtype=np.uint32)
                f.create_dataset("histograms", (total_images, hist_size), 
                               dtype=np.float16, compression='gzip', compression_opts=6)
                f.attrs.update({
                    "count": 0,
                    "bins": bins,
                    "colorspace": colorspace,
                    "source_file": source_file,
                    "histogram_type": "pairwise_2d",
                    "preprocessing": "standardized"
                })
                
            print(f"Created output file: {output_path}")
            print(f"Histogram size: {hist_size} features per image")
        except Exception as e:
            print(f"Error creating output file: {e}")
            raise
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract color histograms with consistent preprocessing")
    parser.add_argument("h5_path", help="Path to input HDF5 with images")
    parser.add_argument("--bins", type=int, default=32, 
                       help="Number of bins per histogram dimension")
    parser.add_argument("--colorspace", choices=["lab", "hsv", "rgb"], default="hsv",
                       help="Color space for histogram extraction")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for processing")
    parser.add_argument("--workers", type=int, default=16,
                       help="Number of worker threads")
    
    args = parser.parse_args()
    
    # Create output file path
    output_path = os.path.join(
        os.path.dirname(args.h5_path),
        f"histograms_{args.bins}_{args.colorspace}.h5"
    )
    
    # Check input file
    if not os.path.exists(args.h5_path):
        print(f"Input file not found: {args.h5_path}")
        return
    
    # Load metadata from input file
    with h5py.File(args.h5_path, 'r') as f:
        total_images = f.attrs["count"]
        print(f"Input file contains {total_images} images")
    
    # Check existing progress
    already_processed = 0
    if os.path.exists(output_path):
        try:
            with h5py.File(output_path, 'r') as f:
                already_processed = f.attrs["count"]
                existing_bins = f.attrs.get("bins", 0)
                existing_colorspace = f.attrs.get("colorspace", "")
                
                if existing_bins != args.bins or existing_colorspace != args.colorspace:
                    print(f"Warning: Existing file has different parameters")
                    print(f"Existing: bins={existing_bins}, colorspace={existing_colorspace}")
                    print(f"Requested: bins={args.bins}, colorspace={args.colorspace}")
                    response = input("Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        return
            
            print(f"Already processed: {already_processed}/{total_images}")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            already_processed = 0
    else:
        create_output_h5(output_path, total_images, args.bins, args.colorspace, args.h5_path)
    
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
            
            with tqdm(total=images_remaining, desc="Processing histograms") as pbar:
                for start in range(already_processed, total_images, args.batch_size):
                    end = min(start + args.batch_size, total_images)
                    batch_size = end - start
                    
                    # Load batch data
                    ids = input_file["ids"][start:end]
                    images_bytes = input_file["jpeg_bytes"][start:end]
                    
                    batch_ids = []
                    batch_histograms = []
                    
                    # Process batch in parallel
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        futures = [
                            executor.submit(extract_lab_pair_histogram, img_id, img_bytes, 
                                          args.bins, args.colorspace)
                            for img_id, img_bytes in zip(ids, images_bytes)
                        ]
                        
                        for future in futures:
                            img_id, hist = future.result()
                            batch_ids.append(img_id)
                            
                            if hist is not None:
                                batch_histograms.append(hist)
                            else:
                                # Create zero histogram for failed images
                                zero_hist = np.zeros(args.bins**2 * 3, dtype=np.float16)
                                batch_histograms.append(zero_hist)
                    
                    # Prepare batch data
                    histograms_array = np.stack(batch_histograms)
                    ids_array = np.array(batch_ids)
                    
                    # Submit write task
                    write_future = write_executor.submit(
                        batch_to_h5, output_path, ids_array, histograms_array, 
                        start, end, write_lock
                    )
                    write_futures.append(write_future)
                    
                    pbar.update(batch_size)
                    
                    # Memory management
                    if (start - already_processed) % (args.batch_size * 10) == 0:
                        # Wait for some writes to complete
                        completed = []
                        for i, future in enumerate(write_futures):
                            if future.done():
                                try:
                                    future.result()  # Check for exceptions
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
                print(f"✓ All images processed successfully")
            else:
                print(f"⚠ Warning: Expected {total_images}, got {final_count}")
                
    except Exception as e:
        print(f"Error verifying results: {e}")
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()