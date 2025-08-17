# Image-Recommender
An intelligent image recommendation system based on visual similarity.

Project Objective

The goal of this project is to develop an intelligent image recommendation system that identifies and returns visually similar images from a dataset of approximately 500,000 images. The system is designed to support multiple similarity measures, ensure fast search performance, and provide a modular, well-documented Python implementation.

Features

- **Database management**: Store image paths and metadata in a SQLite database.  
- **Preprocessing**: Resize images to a uniform resolution (65,536 px) for consistent input.  
- **Perceptual hashing (pHash)**: Robust similarity measure based on frequency domain features.  
- **Color hashing**: LAB color space histograms with both separate and combined (3D) bins.  
- **Embeddings**: Use pre-trained ViT-B/16 (Vision Transformer) to extract feature embeddings.  
- **Similarity search**: Approximate nearest neighbor search using FAISS for large-scale datasets.  
- **Visualization**: Dimensionality reduction with UMAP to project embeddings into 3D.  
- **Optional labeling**: Generate semantic labels with a model to enrich visualization.

Usage
1. Database creation

Create a SQLite database and store image paths along with metadata (e.g., size).

2. Preprocessing

Resize all images to 65,536 px resolution for consistent input.

3. Hash extraction

Run the hashing pipeline to compute perceptual and color hashes:
bash:

python hash_pipeline.py path/to/images.h5 --phash_bits 1024 --color_bits 1024

This will save results in .h5 format and create a combined FAISS index.

4. Embeddings

Extract feature embeddings using ViT-B/16 and store them in .h5 files.

5. Similarity search

Use FAISS to find the top-N most similar images for a given query:
bash:

python recommend.py --query path/to/example.jpg --top_n 5

6. Visualization

Apply UMAP on embeddings to reduce dimensionality and visualize relationships in 3D space.

