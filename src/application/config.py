import os

# Global configuration and constants
HEADER_HEIGHT = 80
BODY_HEIGHT = f"calc(100vh - {HEADER_HEIGHT}px)"
PROGRESS_COLOR = "#c181fd"
PROGRESS_DIR = os.path.join(os.getcwd(), "progress")
os.makedirs(PROGRESS_DIR, exist_ok=True)

# Data directory path for image data
DATA_DIR = r"E:\data\recommender_data"

# File mappings for each processing section
FILE_MAPPINGS = {
    "embeddings": [
        "embeddings_clip.pkl",
        "embeddings_resnet.pkl",
        "embeddings_vgg.pkl",
        "embeddings_inception.pkl",
    ],
    "histograms": [
        "histograms_lab.pkl",
        "histograms_hsv.pkl",
        "histograms_rgb.pkl",
    ],
    "clustering": [
        "kmeans_clusters.pkl",
        "cluster_centers.pkl",
        "cluster_labels.pkl",
    ],
    "hashing": [
        "hashes_ahash.pkl",
        "hashes_phash.pkl",
        "hashes_dhash.pkl",
        "similar_groups.pkl",
    ],
}

# Application color scheme
COLORS = {
    "primary": "#c181fd",
    "secondary": "#555555",
    "background": "#212529",
    "card_bg": "#495057",
    "text_light": "#eeccff",
    "text_purple": "#dab2ff",
}

# Default slider configurations for various controls
SLIDER_CONFIGS = {
    "batch_size": {
        "min": 4,
        "max": 13,
        "step": 1,
        "value": 7,
        "marks": {str(i): str(2**i) for i in range(4, 14)},
    },
    "num_worker": {
        "min": 1,
        "max": 16,
        "step": 1,
        "value": 8,
        "marks": {str(i): str(i) if i % 2 == 0 else "" for i in range(1, 17)},
    },
    "prefetch_factor": {
        "min": 0,
        "max": 16,
        "step": 1,
        "value": 2,
        "marks": {str(i): str(i) if i % 2 == 0 else "" for i in range(0, 17)},
    },
    "hist_bins": {"min": 8, "max": 256, "step": 8, "value": 32},
    "kmeans_clusters": {"min": 2, "max": 50, "step": 1, "value": 8},
    "kmeans_maxiter": {"min": 50, "max": 1000, "step": 50, "value": 300},
    "hash_size": {"min": 8, "max": 64, "step": 8, "value": 32},
    "hash_threshold": {"min": 1, "max": 100, "step": 1, "value": 10},
}

# Dropdown option configurations
DROPDOWN_OPTIONS = {
    "hist_colorspace": ["LAB", "HSV", "RGB"],
    "kmeans_init": ["k-means++", "random"],
    "hash_method": ["ahash", "phash", "dhash"],
}

# Operation labels for UI display
OPERATION_LABELS = {
    "search": "🔍 Search",
    "embeddings": "Start",
    "histograms": "Start",
    "clustering": "Start",
    "hashing": "Start",
}