"""
InboxIQ - Unsupervised Clustering (K-Means + PCA)
====================================================
This module demonstrates UNSUPERVISED learning, distinct from the
supervised category/priority classifiers trained in train_models.py.

- TF-IDF turns each email's text into a high-dimensional numeric vector.
- PCA (Principal Component Analysis) reduces that vector to 2 dimensions
  purely for visualization (it does not "know" about categories/priority).
- K-Means then groups emails into clusters based only on text similarity,
  with no labels involved at all.

Used by the "Email Clusters" page in app.py.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import preprocess_for_vectorizer


def run_clustering(subjects, bodies, n_clusters=5, random_state=42):
    """
    Vectorize the given emails with a fresh TF-IDF vectorizer, reduce to 2D
    with PCA, and group them with K-Means.

    Returns a dict with:
        - pca_coords: list of [x, y] for each email
        - cluster_labels: list of cluster ids (ints)
        - explained_variance: fraction of variance captured by the 2 PCA components
    Returns None if there are too few emails to cluster meaningfully.
    """
    n_samples = len(subjects)
    if n_samples < max(n_clusters, 3):
        return None

    texts = [preprocess_for_vectorizer(s, b) for s, b in zip(subjects, bodies)]

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)

    n_components = 2
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(tfidf_matrix.toarray())

    k = min(n_clusters, n_samples)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    return {
        "pca_coords": coords.tolist(),
        "cluster_labels": cluster_labels.tolist(),
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "n_clusters": k,
    }
