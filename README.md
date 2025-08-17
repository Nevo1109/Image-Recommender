# Image Recommender

## Overview
This project implements a complete image recommendation and similarity search system. It combines multiple approaches for feature extraction, clustering, and efficient retrieval. The system is designed to handle large image collections and provides several methods to measure similarity between images.

The project includes:

- Backend for image processing, feature extraction, clustering, and similarity search
- Persistence using SQLite and HDF5 files
- Algorithms such as K-Means clustering, LAB color histograms, perceptual hashing (pHash), color hashing, deep learning–based embeddings, and FAISS-based approximate nearest neighbor search
- (Optional) A web-based frontend using Plotly Dash for interactive exploration and visualization

---

## Features

- Preprocessing of all images (resizing, normalization)
- Storage of image paths and metadata in a relational database
- Feature extraction via:
  - LAB color histograms (3D)
  - Perceptual hash (pHash)
  - Color hash
  - Vision Transformer (ViT) embeddings
- Combination of features for robust similarity search
- Clustering with K-Means to group visually similar images
- Efficient nearest-neighbor search with FAISS
- (Optional) UMAP projection of embeddings into 3D for visualization
- (Optional) Web interface with:
  - Accordion panel for weighting histogram scores via slider
  - Method selection between pHash and Color Hash
  - Cluster-based prefiltering checkbox
  - Results visualization with similarity contributions:
    - blue: embedding
    - green: color
    - yellow: structural similarity


## Image-Recommender/
  - Src/
    - application/ (Optional web frontend)
      - assets/ (Static assets: CSS, JS)
      - callbacks.py (Dash callbacks)
      - components.py (UI components)
      - config.py (App configuration)
      - main.py (Entry point for the web app)
      - styles.py (Style definitions)
      - utils.py (Helper functions for the app)
    - database/ (Database handling)
      - create_db.py (Initialize and populate SQLite DB)
    - features/ (Feature extraction methods)
      - compute_embeddings.py (Embedding extraction, e.g., ViT)
      - hashing.py (Perceptual & color hashing)
      - histograms.py (LAB color histograms)
      - kmeans.py (K-Means clustering)
    - search/ (Similarity search)
      - embeddings_faiss.py (FAISS for embeddings)
      - ... 
    - utils/ (Utility scripts)
      - dataset.py (Dataset loading)
      - disk_drive.py (File I/O helpers)
      - preprocess_resize.py (Preprocessing & resizing)
  - requirements.txt (Python dependencies)
  - README.md (Project documentation)





<img width="729" height="779" alt="Untitled (1)" src="https://github.com/user-attachments/assets/c44eaff9-3cec-43a2-89d4-a09171be9a79" />
Installation

Clone the repository:

git clone https://github.com/Nevo1109/Image-Recommender.git
cd Image-Recommender


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate    # on Linux/Mac
venv\Scripts\activate       # on Windows


Install dependencies:

pip install -r requirements.txt

Usage
1. Create database and preprocess images
python database/create_db.py --input data/images --output db/images.db
python utils/preprocess_resize.py --input data/images --output data/resized

2. Compute features
python features/hashing.py
python features/histograms.py
python features/compute_embeddings.py
python features/kmeans.py

3. Run similarity search
python search/embeddings_faiss.py

4. (Optional) Start the web application
python application/main.py

Future Work

Label generation for UMAP visualization

Extended weighting strategies for feature fusion

Support for additional embedding models

License

This project is provided for academic and research purposes. See LICENSE for details.
