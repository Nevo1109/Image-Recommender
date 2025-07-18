import sqlite3
from pathlib import Path
from PIL.Image import registered_extensions
from tqdm import tqdm

def walk(root: str):
    file_extensions = set(registered_extensions().keys())
    path_generator = Path(root).rglob("*")
    for path in path_generator:
        if path.suffix.lower() in file_extensions:
            yield str(path)

def main():
    db_path = r"C:\Users\kilic\OneDrive\Desktop\db\image_recommender.db"
    image_root = r"E:\data\image_data"  # Pfad zu deinem Bildordner

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS images("
        "id_image INTEGER PRIMARY KEY,"
        "path TEXT)"
    )
    conn.commit()

    for image_path in tqdm(walk(image_root)):
        cursor.execute("INSERT INTO images VALUES (NULL, (?))", (image_path,))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
