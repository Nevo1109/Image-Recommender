import os
import json
import base64
import sqlite3
import random
import tkinter as tk
from tkinter import filedialog
from config import PROGRESS_DIR, DATA_DIR, FILE_MAPPINGS


def open_native_file_dialog():
    """Open native file dialog for image selection."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filenames = filedialog.askopenfilenames(
        initialdir=r"E:\data\image_data",
        title="Select Images (max 3)",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"),
            ("All files", "*.*"),
        ],
        multiple=True,
    )
    root.destroy()
    return list(filenames)


def get_random_image_from_db():
    """Get random image from SQLite database."""
    db_path = os.path.join(DATA_DIR, "images.db")

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get random image - expects table with 'path' column
        # You can adjust table/column names as needed
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("❌ No tables found in database")
            conn.close()
            return None

        # Try various possible table names
        query = "SELECT path FROM images ORDER BY RANDOM() LIMIT 1;"

        cursor.execute(query)
        result = cursor.fetchone()
        image_path = result[0] if isinstance(result, tuple) else result

        conn.close()

        if result and image_path:
            # Check if file path exists
            if os.path.exists(image_path):
                return image_path
            else:
                print(f"❌ Image file not found: {image_path}")
                return None
        else:
            print("❌ No images found in database")
            return None

    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        return None


def file_to_base64(filepath):
    """Convert file to Base64 string for web display."""
    try:
        with open(filepath, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        ext = filepath.split(".")[-1].lower()
        return f"data:image/{ext};base64,{encoded}"
    except Exception as e:
        print(f"❌ Error converting file to base64: {str(e)}")
        return None


# Progress File Operations
def progress_file_path(op_id: str):
    """Get the file path for a progress file."""
    return os.path.join(PROGRESS_DIR, f"progress_{op_id}.json")


def write_progress_file(op_id: str, value: int, running: bool = True, message: str = ""):
    """Write progress file atomically to avoid race conditions."""
    path = progress_file_path(op_id)
    data = {
        "value": int(value),
        "running": bool(running),
        "operation": op_id,
        "message": str(message),
    }

    # Write to temporary file first, then rename for atomic operation
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(path + ".tmp", path)


def read_progress_file(op_id: str):
    """Read progress file safely with error handling."""
    path = progress_file_path(op_id)
    if not os.path.exists(path):
        return {"value": 0, "running": False, "operation": op_id, "message": ""}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"value": 0, "running": False, "operation": op_id, "message": ""}


def check_file_exists(filename):
    """Check if a file exists in the DATA_DIR."""
    filepath = os.path.join(DATA_DIR, filename)
    return os.path.exists(filepath)


def get_file_dropdown_options(section_key):
    """Create dropdown options - shows only available files."""
    files = FILE_MAPPINGS.get(section_key, [])
    options = []
    available_files = []

    # Collect only available files
    for filename in files:
        if check_file_exists(filename):
            available_files.append(filename)
            # Clean display without status icons (since all are available)
            options.append({"label": filename, "value": filename})

    # Fallback if no files are available
    if not options:
        return [{"label": "No files available", "value": "none", "disabled": True}], "none"

    # Default: first available file
    default_file = available_files[0] if available_files else "none"
    return options, default_file


def update_display_from_selected_images(selected_images):
    """Create display for selected images with responsive layout."""
    from dash import html

    if len(selected_images) == 0:
        return [], 0

    num_images = len(selected_images)
    max_width = f"calc({100/num_images}% - 10px)" if num_images > 1 else "400px"

    image_elements = [
        html.Img(
            src=content,
            id={"type": "img-click", "index": name},
            className="img-hover img-responsive",
            style={
                "maxWidth": max_width,
                "maxHeight": "270px",
                "objectFit": "contain",
                "border": "2px solid #495057",
                "borderRadius": "8px",
                "cursor": "pointer",
                "margin": "0 5px",
                "flex": "0 0 auto",
            },
        )
        for name, content in selected_images.items()
    ]

    return (
        html.Div(
            image_elements,
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "width": "100%",
                "height": "100%",
                "minHeight": "250px",
                "padding": "0 20px",
                "margin": "0",
                "background": "transparent",
                "boxSizing": "border-box",
            },
        ),
        num_images,
    )