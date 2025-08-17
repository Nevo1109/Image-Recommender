import argparse
import os
import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def extract_lab_pair_histogram(id, img_bytes: np.ndarray, bins: int = 32):
    """Extract color histogram from image. Pairs of two."""
    try:
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Calculate histogram for each channel
        hist_la = cv2.calcHist([img], [0, 1], None, [bins, bins], [0, 256, 0, 256])
        hist_lb = cv2.calcHist([img], [0, 2], None, [bins, bins], [0, 256, 0, 256])
        hist_ab = cv2.calcHist([img], [1, 2], None, [bins, bins], [0, 256, 0, 256])
        
        # Normalize and flatten
        norm_hist_la = hist_la / hist_la.sum()
        norm_hist_lb = hist_lb / hist_lb.sum()
        norm_hist_ab = hist_ab / hist_ab.sum()
        hist = np.concatenate([norm_hist_la, norm_hist_lb, norm_hist_ab]).flatten()
        
        return id, hist.astype(np.float16)
    except Exception as e:
        print("Error at image: ", e)
        return id, None


def batch_to_h5(h5_path: str, ids: np.ndarray, batch_data: np.ndarray, start_idx: int, end_idx: int, lock: Lock):
    """Write batch to HDF5."""
    with lock:
        with h5py.File(h5_path, "r+") as f:
            f['ids'][start_idx:end_idx] = ids
            f['histograms'][start_idx:end_idx] = batch_data
            f.attrs["count"] = end_idx
            f.flush()


def create_output_h5(output_path: str, total_images: int, bins: int, colorspace: str, source_file: str):
    """Create output HDF5 file."""
    if not os.path.exists(output_path):
        with h5py.File(output_path, "w") as f:
            f.create_dataset("ids", (total_images,), dtype="uint32")
            f.create_dataset("histograms", (total_images, bins**2 * 3), dtype=np.float16)
            f.attrs.update({
                "count": 0,
                "bins": bins,
                "colorspace": colorspace,
                "source_file": source_file
            })
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--colorspace", choices=["lab", "hsv", "rgb"], default="hsv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    
    args = parser.parse_args()
    
    colorspaces = {"lab": cv2.COLOR_BGR2LAB, "hsv": cv2.COLOR_BGR2HSV, "rgb": cv2.COLOR_BGR2RGB}
    colorspace = colorspaces[args.colorspace]
    
    output_path = os.path.join(
        os.path.dirname(args.h5_path),
        f"histograms_{args.bins}_{args.colorspace}.h5"
    )
    
    # Check existing progress
    with h5py.File(args.h5_path, 'r') as f:
        total_images = f.attrs["count"]
    
    already_processed = 0
    if os.path.exists(output_path):
        with h5py.File(output_path, 'r') as f:
            already_processed = f.attrs["count"]
        print(f"Already processed: {already_processed}/{total_images}")
    else:
        create_output_h5(output_path, total_images, args.bins, args.colorspace, args.h5_path)
    
    if already_processed >= total_images:
        print("All done!")
        return
    
    write_lock = Lock()
    
    with h5py.File(args.h5_path, 'r') as input_file:
        with ThreadPoolExecutor(max_workers=1) as write_executor:
            write_futures = []
            
            with tqdm(total=total_images - already_processed) as pbar:
                for start in range(already_processed, total_images, args.batch_size):
                    end = min(start + args.batch_size, total_images)
                    
                    ids = input_file["ids"][start:end]
                    images_bytes = input_file["jpeg_bytes"][start:end]
                    batch_ids = []
                    batch_histograms = []
                    
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        futures = [
                            executor.submit(extract_lab_pair_histogram, id, img_bytes, args.bins)
                            for id, img_bytes in zip(ids, images_bytes)
                        ]
                        
                        for future in futures:
                            id, hist = future.result()
                            if hist is not None:
                                batch_ids.append(id)
                                batch_histograms.append(hist)
                            else:
                                batch_ids.append(id)
                                batch_histograms.append(np.zeros(args.bins**2 * 3, dtype=np.float16))
                    
                    histograms_array = np.stack(batch_histograms)
                    ids_array = np.array(batch_ids)
                    
                    write_future = write_executor.submit(
                        batch_to_h5, output_path, ids_array, histograms_array, start, end, write_lock
                    )
                    write_futures.append(write_future)
                    
                    pbar.update(end - start)
                    
                    if (start - already_processed) % (args.batch_size * 10) == 0:
                        gc.collect()
            
            for future in write_futures:
                future.result()
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()