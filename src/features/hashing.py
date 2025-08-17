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


def preprocess_image(img):
    """Standardisierte Bildvorverarbeitung für konsistente Hashing."""
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


def phash(input_data, phash_bits: int = 1024) -> np.ndarray:
    """Calculate perceptual hash with improved stability."""
    if isinstance(input_data, str):
        img = cv2.imread(input_data)
        if img is None:
            raise ValueError(f"Could not load image: {input_data}")
        img = preprocess_image(img)
    else:
        img = input_data
        
    if img is None:
        raise ValueError("Image preprocessing failed")
    
    hash_size = int(np.sqrt(phash_bits))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to hash_size * 8 for better DCT precision
    dct_size = hash_size * 8
    img_resized = cv2.resize(img_gray, (dct_size, dct_size), interpolation=cv2.INTER_AREA)
    
    # Apply DCT
    dct_coeff = dct(dct(img_resized.astype(np.float32).T, norm='ortho').T, norm='ortho')
    
    # Take low frequency components
    dct_low = dct_coeff[:hash_size, :hash_size]
    
    # Use mean instead of median for more stability
    threshold = np.mean(dct_low)
    
    result = (dct_low > threshold).astype(np.uint8).flatten()
    
    # Ensure exact target size
    if len(result) > phash_bits:
        result = result[:phash_bits]
    elif len(result) < phash_bits:
        result = np.pad(result, (0, phash_bits - len(result)), 'constant', constant_values=0)
    
    return result


def color_hash(input_data, color_bits: int = 1024) -> np.ndarray:
    """Calculate color hash using LAB histogram with improved consistency."""
    if isinstance(input_data, str):
        img = cv2.imread(input_data)
        if img is None:
            raise ValueError(f"Could not load image: {input_data}")
        img = preprocess_image(img)
    else:
        img = input_data
        
    if img is None:
        raise ValueError("Image preprocessing failed")
        
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Calculate optimal bin distribution
    combined_bins = max(8, int(round((color_bits * 0.7) ** (1/3))))
    remaining_bits = max(0, color_bits - (combined_bins ** 3))
    separate_bins = max(8, remaining_bits // 3)
    
    # Individual channel histograms with fixed ranges
    hist_l = cv2.calcHist([img_lab], [0], None, [separate_bins], [0, 256])
    hist_a = cv2.calcHist([img_lab], [1], None, [separate_bins], [0, 256])
    hist_b = cv2.calcHist([img_lab], [2], None, [separate_bins], [0, 256])
    
    # Normalize histograms
    hist_l = hist_l / (hist_l.sum() + 1e-8)
    hist_a = hist_a / (hist_a.sum() + 1e-8)
    hist_b = hist_b / (hist_b.sum() + 1e-8)
    
    hist_separate = np.concatenate([hist_l, hist_a, hist_b]).flatten()
    
    # Use mean threshold for consistency
    separate_threshold = np.mean(hist_separate)
    separate_hash = (hist_separate > separate_threshold).astype(np.uint8)
    
    # Combined 3D LAB histogram
    hist_3d = cv2.calcHist([img_lab], [0, 1, 2], None, 
                          [combined_bins, combined_bins, combined_bins],
                          [0, 256, 0, 256, 0, 256])
    
    hist_combined = hist_3d.flatten()
    hist_combined = hist_combined / (hist_combined.sum() + 1e-8)
    
    combined_threshold = np.mean(hist_combined)
    combined_hash = (hist_combined > combined_threshold).astype(np.uint8)
    
    # Combine hashes
    result = np.concatenate([separate_hash, combined_hash])
    
    # Ensure exact target size
    if len(result) > color_bits:
        result = result[:color_bits]
    elif len(result) < color_bits:
        result = np.pad(result, (0, color_bits - len(result)), 'constant', constant_values=0)
    
    return result


def extract_hashes(data: tuple):
    """Extract hashes with consistent preprocessing."""
    img_id, img_bytes, phash_bits, color_bits = data
    try:
        # Decode image from bytes
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None
        
        # Apply consistent preprocessing
        preprocessed = preprocess_image(img)
        if preprocessed is None:
            return None
        
        # Calculate hashes on preprocessed image
        p_hash = phash(preprocessed, phash_bits)
        c_hash = color_hash(preprocessed, color_bits)
        
        return (img_id, p_hash, c_hash)
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        return None


def save_results(output_dir: str, ids: np.ndarray, phashes: np.ndarray, colorhashes: np.ndarray, 
                phash_bits: int, color_bits: int):
    """Save hashes to HDF5 and create FAISS indices."""
    base_name = f"{phash_bits}_{color_bits}"
    
    # Save to HDF5
    hdf5_path = os.path.join(output_dir, f"hashes_{base_name}.h5")
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('ids', data=ids)
        f.create_dataset('phashes', data=phashes)
        f.create_dataset('colorhashes', data=colorhashes)
        f.attrs.update({
            'phash_bits': phash_bits, 
            'color_bits': color_bits,
            'count': len(ids),
            'preprocessing': 'standardized'
        })
    
    print(f"Saved hashes to {hdf5_path}")
    
    # Create FAISS indices
    try:
        # Perceptual hash index
        phash_base = faiss.IndexBinaryFlat(phash_bits)
        phash_index = faiss.IndexBinaryIDMap(phash_base)
        
        # Pack bits for FAISS
        phash_packed = np.packbits(phashes, axis=1)
        phash_index.add_with_ids(phash_packed, ids.astype(np.int64))
        
        phash_index_path = os.path.join(output_dir, f"phash_index_{base_name}.index")
        faiss.write_index_binary(phash_index, phash_index_path)
        print(f"Saved pHash index to {phash_index_path}")
        
        # Color hash index
        color_base = faiss.IndexBinaryFlat(color_bits)
        color_index = faiss.IndexBinaryIDMap(color_base)
        
        color_packed = np.packbits(colorhashes, axis=1)
        color_index.add_with_ids(color_packed, ids.astype(np.int64))
        
        color_index_path = os.path.join(output_dir, f"color_index_{base_name}.index")
        faiss.write_index_binary(color_index, color_index_path)
        print(f"Saved color hash index to {color_index_path}")
        
    except Exception as e:
        print(f"Error creating FAISS indices: {e}")
    
    # Cleanup
    del phashes, colorhashes
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Generate consistent image hashes")
    parser.add_argument("h5_path", help="Path to input HDF5 with images")
    parser.add_argument("--phash_bits", type=int, default=1024, 
                       help="Number of bits for perceptual hash")
    parser.add_argument("--color_bits", type=int, default=1024, 
                       help="Number of bits for color hash")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for processing")
    parser.add_argument("--workers", type=int, default=12,
                       help="Number of worker threads")
    
    args = parser.parse_args()
    output_dir = os.path.dirname(args.h5_path)
    
    # Load metadata
    with h5py.File(args.h5_path, 'r') as f:
        total_images = f.attrs["count"]
        
    print(f"Processing {total_images} images with {args.workers} workers")
    print(f"Hash sizes: pHash={args.phash_bits} bits, colorHash={args.color_bits} bits")
    
    # Process images
    all_ids = []
    all_phashes = []
    all_colorhashes = []
    
    with h5py.File(args.h5_path, 'r') as input_file:
        image_ids = input_file["ids"][:]
        
        with tqdm(total=total_images, desc="Extracting hashes") as pbar:
            for i in range(0, total_images, args.batch_size):
                batch_end = min(i + args.batch_size, total_images)
                batch_indices = range(i, batch_end)
                
                # Load batch data
                batch_ids = image_ids[batch_indices]
                batch_images = input_file["jpeg_bytes"][batch_indices]
                
                # Prepare data for processing
                batch_data = [
                    (batch_ids[j], batch_images[j], args.phash_bits, args.color_bits)
                    for j in range(len(batch_indices))
                ]
                
                # Process batch
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    results = list(executor.map(extract_hashes, batch_data))
                
                # Collect successful results
                for result in results:
                    if result is not None:
                        img_id, p_hash, c_hash = result
                        all_ids.append(img_id)
                        all_phashes.append(p_hash)
                        all_colorhashes.append(c_hash)
                
                pbar.update(batch_end - i)
                
                # Periodic cleanup
                if i % (args.batch_size * 20) == 0:
                    gc.collect()
    
    # Convert to numpy arrays and save
    if all_ids:
        print(f"Successfully processed {len(all_ids)}/{total_images} images")
        save_results(
            output_dir, 
            np.array(all_ids), 
            np.array(all_phashes), 
            np.array(all_colorhashes),
            args.phash_bits, 
            args.color_bits
        )
    else:
        print("No images were successfully processed!")


if __name__ == "__main__":
    main()