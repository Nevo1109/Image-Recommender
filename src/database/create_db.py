import os
import sqlite3
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.utils.disk_drive import get_db_path, get_image_size, get_images_folder


def init_db(path = None):
    """Create database file and tables for image paths und pixel dimensions."""
    if path is None:
        path = get_db_path()
        
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images(
            id_image INTEGER PRIMARY KEY, 
            path TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata(
            id_image INTEGER,
            width INTEGER,
            height INTEGER,
            size INTEGER,
            FOREIGN KEY (id_image) REFERENCES images(id_image)
        )
    """)
    conn.commit()
    conn.close()
    print("Database created at", get_db_path())


def insert_paths(db_path=None, root_images_folder=None):
    """Reads paths of images from given or default root_images_folder and inserts them into given or default database."""
    if db_path is None:
        db_path = get_db_path()
    if root_images_folder is None:
        root_images_folder = get_images_folder()

    # check if paths have already been inserted
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if cursor.execute("SELECT COUNT(*) FROM images").fetchone()[0] > 0:
        print("Paths already inserted into database")
        return

    # collect image paths
    root = get_images_folder()
    path_gen = Path(root).rglob("*")
    only_images = [
        str(path)
        for path in path_gen
        if path.suffix.lower() in Image.registered_extensions().keys()
    ]

    # insert paths
    cursor.executemany(
        "INSERT INTO images VALUES (?, ?)",
        [(i, path[1:]) for i, path in enumerate(only_images)],  # cut off drive letter
    )
    conn.commit()
    conn.close()
    print("Image paths inserted into database")


def insert_sizes(db_path=None, chunck_size=10_000):
    """
    Reads paths from given or default database and inserts image sizes into it. 
    
    Because it takes some time the process is chunked to avoid going through all images again.
    """
    if db_path is None:
        db_path = get_db_path()
    
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    if cursor.execute("SELECT COUNT(*) FROM metadata").fetchone()[0] > 0:
        cursor.execute("""
            SELECT id_image, path FROM images
            WHERE id_image > (
                SELECT MAX(id_image) FROM metadata
            )
        """)
    else:
        cursor.execute("SELECT id_image, path FROM images")

    remaining = cursor.fetchall()
    if len(remaining) == 0:
        print("Images already inserted into database")
        return
    remaining_indices, remaining_paths = zip(*remaining)
    chunks = len(remaining_paths) // chunck_size + 1
    print("Starting at Image", remaining_indices[0], "with", len(remaining_paths), "images to collect")

    for i in range(chunks):
        idcs = remaining_indices[i * chunck_size : (i + 1) * chunck_size]
        paths = remaining_paths[i * chunck_size : (i + 1) * chunck_size]
        with ThreadPoolExecutor(max_workers=24) as executor:
            sizes = list(
                tqdm(
                    executor.map(get_image_size, paths),
                    total=len(paths),
                    desc=f"Collecting image sizes (Chunk {i + 1}/{chunks} {i / chunks * 100:.2f}%)",
                    smoothing=0.1,
                )
            )
            cursor.executemany(
                "INSERT INTO metadata VALUES (?, ?, ?, ?)",
                [(i, w, h, size) for i, (w, h, size, _) in zip(idcs, sizes)],
            )
            db.commit()

    db.close()
    print("Image sizes inserted into database")


def main(db_path=None, root_image_folder=None, get_image_sizes=True):
    if db_path is None and not os.path.exists(get_db_path()):
        init_db()
        db_path = get_db_path()
        
    insert_paths(db_path, root_image_folder)
    
    if get_image_sizes:
        insert_sizes(db_path)
    print("Completed")


if __name__ == "__main__":
    main(get_image_sizes=True)
