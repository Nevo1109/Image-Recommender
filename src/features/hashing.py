import argparse
import os
import gc
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import faiss
import numpy as np

from tqdm import tqdm
from scipy.fftpack import dct


def phash(img: np.ndarray, phash_bits: int = 1024) -> np.ndarray:
    """Calculate perceptual hash."""
    hash_size = int(np.sqrt(phash_bits))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img_gray, (hash_size * 4, hash_size * 4))
    
    dct_coeff = dct(dct(img.T, norm='ortho').T, norm='ortho')
    dct_low = dct_coeff[:hash_size, :hash_size]
    median = np.median(dct_low)
    
    result = (dct_low > median).astype(np.uint8).flatten()
    
    # Pad or truncate to exact target size
    if len(result) > phash_bits:
        result = result[:phash_bits]
    elif len(result) < phash_bits:
        result = np.pad(result, (0, phash_bits - len(result)), 'constant', constant_values=0)
    
    return result


def color_hash(img: np.ndarray, color_bits: int = 1024) -> np.ndarray:
    """Calculate color hash using LAB histogram - separate and combined."""
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Calculate optimal bin distribution for target bits
    # Use cube root for 3D bins, rest for separate bins
    combined_bins = int(round((color_bits * 0.9) ** (1/3)))  # 90% for 3D histogram
    remaining_bits = color_bits - (combined_bins ** 3)
    separate_bins = max(1, remaining_bits // 3)  # Divide by 3 LAB channels
    
    # Separate L, A, B histograms
    hist_l = cv2.calcHist([img_lab], [0], None, [separate_bins], [0, 100])
    hist_a = cv2.calcHist([img_lab], [1], None, [separate_bins], [0, 255])
    hist_b = cv2.calcHist([img_lab], [2], None, [separate_bins], [0, 255])
    
    hist_separate = np.concatenate([hist_l, hist_a, hist_b]).flatten()
    hist_separate = hist_separate / (hist_separate.sum() + 1e-8)
    separate_hash = (hist_separate > np.median(hist_separate)).astype(np.uint8)
    
    # Combined 3D LAB histogram
    hist_3d = cv2.calcHist([img_lab], [0, 1, 2], None, 
                          [combined_bins, combined_bins, combined_bins],
                          [0, 100, 0, 255, 0, 255])
    
    hist_combined = hist_3d.flatten()
    hist_combined = hist_combined / (hist_combined.sum() + 1e-8)
    combined_hash = (hist_combined > np.median(hist_combined)).astype(np.uint8)
    
    # Pad or truncate to exact target size
    result = np.concatenate([separate_hash, combined_hash])
    if len(result) > color_bits:
        result = result[:color_bits]
    elif len(result) < color_bits:
        result = np.pad(result, (0, color_bits - len(result)), 'constant', constant_values=0)
    
    return result


def extract_hashes(data: tuple):
    """Extract hashes with image ID."""
    img_id, img_bytes, phash_bits, color_bits = data
    try:
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None
        
        return (img_id, phash(img, phash_bits), color_hash(img, color_bits))
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        return None


def save_results(output_dir: str, ids: np.ndarray, phashes: np.ndarray, colorhashes: np.ndarray, 
                phash_bits: int, color_bits: int):
    """Save hashes to HDF5 and create combined FAISS index."""
    base_name = f"{phash_bits}_{color_bits}"
    
    # Save to HDF5
    with h5py.File(os.path.join(output_dir, f"hashes_{base_name}.h5"), 'w') as f:
        f.create_dataset('ids', data=ids)
        f.create_dataset('phashes', data=phashes)
        f.create_dataset('colorhashes', data=colorhashes)
        f.attrs.update({
            'phash_bits': phash_bits, 
            'color_bits': color_bits,
            'count': len(ids)
        })
    
    # Create combined FAISS index
    total_bits = phash_bits + color_bits
    
    # Kombiniere beide Hash-Typen in einem Index
    combined_hashes = np.concatenate([phashes, colorhashes], axis=1)
    index = faiss.IndexBinaryFlat(total_bits)
    index.add(np.packbits(combined_hashes, axis=1))
    faiss.write_index_binary(index, os.path.join(output_dir, f"combined_index_{base_name}.index"))
    del index
    
    gc.collect()
    print(f"Saved {len(ids)} hashes and combined index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", help="Path to input HDF5 with images")
    parser.add_argument("--phash_bits", type=int, default=1024, help="Number of bits for perceptual hash")
    parser.add_argument("--color_bits", type=int, default=1024, help="Number of bits for color hash")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    
    args = parser.parse_args()
    output_dir = os.path.dirname(args.h5_path)
    
    # Load all data
    with h5py.File(args.h5_path, 'r') as f:
        total_images = f.attrs["count"]
        all_image_ids = f["ids"][:]
        print(f"Loading {total_images} images...")
    
    # Initialize result arrays
    all_ids = []
    all_phashes = []
    all_colorhashes = []
    
    # Process all images in batches
    with h5py.File(args.h5_path, 'r') as input_file:
        for i in tqdm(range(0, total_images, args.batch_size)):
            batch_end = min(i + args.batch_size, total_images)
            batch_indices = list(range(i, batch_end))
            
            # Load batch - alle Bilder auf einmal
            batch_ids = all_image_ids[batch_indices]
            batch_images = input_file["jpeg_bytes"][batch_indices]
            batch_data = [(batch_ids[j], batch_images[j], args.phash_bits, args.color_bits) 
                         for j in range(len(batch_indices))]
            
            # Process in parallel
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = [r for r in executor.map(extract_hashes, batch_data) if r is not None]
            
            # Collect results
            for img_id, p_hash, c_hash in results:
                all_ids.append(img_id)
                all_phashes.append(p_hash)
                all_colorhashes.append(c_hash)
    
    # Save everything at the end
    save_results(output_dir, np.array(all_ids), np.array(all_phashes), np.array(all_colorhashes),
                args.phash_bits, args.color_bits)
    
    print(f"Complete! Processed {len(all_ids)} images")


if __name__ == "__main__":
    main()