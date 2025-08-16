import os
import gc
import h5py
import sqlite3
import numpy as np

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.utils.dataset import create_fixed_sized_images_dataloader
from src.utils.disk_drive import get_db_path, get_drive_letter


def create_h5(folder, image_count, pixel_per_image=65536, quality=80):
    path = folder + f"\\resized_{pixel_per_image}_{quality}%.h5"
    
    if not os.path.exists(path):
        with h5py.File(path, "w") as hdf:
            jpeg_dt = h5py.vlen_dtype(np.dtype("uint8"))
            hdf.create_dataset("ids", (image_count,), "uint32")
            hdf.create_dataset("jpeg_bytes", (image_count,), dtype=jpeg_dt)
            hdf.create_dataset("dimensions", (image_count, 2), dtype="uint16")
            hdf.attrs["count"] = 0
    return path

def batch_to_h5(h5_path, batch, start_idx, end_idx):
    ids, bytes, dimensions = batch
    with h5py.File(h5_path, "r+") as f:
        f['ids'][start_idx:end_idx] = ids
        f['jpeg_bytes'][start_idx:end_idx] = bytes
        f['dimensions'][start_idx:end_idx] = dimensions
        f.attrs["count"] = end_idx
        f.flush()
        
    del ids, bytes, dimensions



if __name__ == "__main__":
    db = sqlite3.connect(get_db_path())
    cursor = db.cursor()

    folder = r"C:\Users\blue_\Desktop"
    file_path = folder + "\\resized_65536_80%.h5"
    
    if os.path.exists(file_path):
        with h5py.File(file_path) as f:
            already_written = f.attrs["count"]
            ids = f["ids"][:already_written].tolist()
            print(already_written)
    else:
        ids = ["-1"]
        already_written = 0

    # small images
    cursor.execute(f"""
        SELECT i.id_image, i.path FROM images AS i
        INNER JOIN metadata as m ON m.id_image = i.id_image
        WHERE size < {1_000_000}
        AND i.id_image NOT IN ({','.join([str(id) for id in ids])})
    """)
    small_images = cursor.fetchall()
    print(len(small_images), "small images remaining")
    config_small = {"batch_size": 128, "num_worker": 6, "prefetch_factor": 3}

    # medium images
    cursor.execute(f"""
        SELECT i.id_image, i.path FROM images AS i
        INNER JOIN metadata as m ON m.id_image = i.id_image
        WHERE size BETWEEN {1_000_000} AND {25_000_000}
        AND i.id_image NOT IN ({','.join([str(id) for id in ids])})
    """)
    medium_images = cursor.fetchall()
    print(len(medium_images), "medium images remaining")
    config_medium = {"batch_size": 32, "num_worker": 6, "prefetch_factor": 2}

    # large images
    cursor.execute(f"""
        SELECT i.id_image, i.path FROM images AS i
        INNER JOIN metadata as m ON m.id_image = i.id_image
        WHERE size BETWEEN {25_000_001} AND {50_000_000}
        AND i.id_image NOT IN ({','.join([str(id) for id in ids])})
    """)
    large_images = cursor.fetchall()
    print(len(large_images), "large images remaining")
    config_large = {"batch_size": 32, "num_worker": 6, "prefetch_factor": 2}

    # very large images
    cursor.execute(f"""
        SELECT i.id_image, i.path FROM images AS i
        INNER JOIN metadata as m ON m.id_image = i.id_image
        WHERE size > {50_000_000}
        AND i.id_image NOT IN ({','.join([str(id) for id in ids])})
    """)
    very_large_images = cursor.fetchall()
    print(len(very_large_images), "very large images remaining")
    config_very_large = {"batch_size": 8, "num_worker": 2, "prefetch_factor": 1}

    all_images = [small_images, medium_images, large_images, very_large_images]
    all_configs = [config_small, config_medium, config_large, config_very_large]
    names = [
        "Small Images",
        "Medium Images",
        "Large Images",
        "Very Large Images",
    ]

    pixel_count = 65536
    jpeg_quality = 80

    image_count = cursor.execute("SELECT COUNT(size) FROM metadata").fetchall()[0][0]  # images that could be loaded
    db.close()
    hdf5_path = create_h5(folder, image_count, pixel_count, jpeg_quality)
    current_write_position = already_written 
    
    print("Initializing Dataloader")
    for images, config, name in list(zip(all_images, all_configs, names))[2:]:
        if len(list(images)) == 0:
            continue
        indices, paths = zip(*images)
        paths = [get_drive_letter() + p for p in paths]

        batch_size = config["batch_size"]
        num_worker = config["num_worker"]
        prefetch_factor = config["prefetch_factor"]
        
        dataloader = create_fixed_sized_images_dataloader(zip(indices, paths), pixel_count, jpeg_quality, batch_size=batch_size, num_workers=num_worker, prefetch_factor=prefetch_factor)

        executor = ThreadPoolExecutor(1)
        futures = []

        for batch_idx, (ids, bytes_batch, dimensions) in tqdm(
            enumerate(dataloader),
            smoothing=0.1,
            total=len(dataloader),
            desc=name,
            bar_format="{desc}{percentage:5.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ):
            start_idx = current_write_position
            end_idx = current_write_position + len(ids)

            futures.append(executor.submit(batch_to_h5, file_path, (ids, bytes_batch, dimensions), start_idx, end_idx))

            current_write_position += len(ids) 

            if batch_idx % 32 == 0:
                del ids, bytes_batch, dimensions
                gc.collect()

        for future in futures:
            future.result()
