###
## cluster_maker
## Will Avery - University of Bath
## Task 5) New module `agglomerative.py` created
###

from __future__ import annotations

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


def agglomerative_clustering(
    X: np.ndarray,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform agglomerative hierarchical clustering.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_clusters : int or None, default None
        Number of clusters to find. If None, distance_threshold must be set.
    distance_threshold : float or None, default None
        The linkage distance threshold above which clusters will not be merged.
        If not None, n_clusters must be None.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
        Linkage criterion. "ward" minimizes variance, others minimize maximum,
        average, or minimum distances.
    metric : str, default "euclidean"
        Metric used to compute distance. For "ward", only "euclidean" is accepted.
    
    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    centroids : ndarray of shape (n_clusters, n_features)
        Centroids of each cluster (mean of points).
    
    Raises
    ------
    ValueError
        If neither n_clusters nor distance_threshold is set,
        or if both are set.
    TypeError
        If X is not a NumPy array.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    
    if n_clusters is None and distance_threshold is None:
        raise ValueError(
            "Either n_clusters or distance_threshold must be set."
        )
    if n_clusters is not None and distance_threshold is not None:
        raise ValueError(
            "Only one of n_clusters or distance_threshold can be set."
        )
    
    if linkage == "ward" and metric != "euclidean":
        raise ValueError(
            'For linkage="ward", metric must be "euclidean".'
        )
    
    # Perform agglomerative clustering
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage=linkage,
        metric=metric,
    )
    labels = model.fit_predict(X)
    
    # Calculate centroids
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    n_features = X.shape[1]
    centroids = np.zeros((n_clusters, n_features), dtype=float)
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if np.any(mask):
            centroids[cluster_id] = X[mask].mean(axis=0)
        else:
            # Empty cluster (should not happen with agglomerative)
            centroids[cluster_id] = np.zeros(n_features)
    
    return labels, centroids


def plot_dendrogram(
    X: np.ndarray,
    linkage_method: str = "ward",
    metric: str = "euclidean",
    truncate_mode: str = "lastp",
    p: int = 12,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a dendrogram for hierarchical clustering.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    linkage_method : str, default "ward"
    metric : str, default "euclidean"
    truncate_mode : str, default "lastp"
        How to truncate dendrogram. Options: "lastp", "level", "none".
    p : int, default 12
        Number of last merges to show if truncate_mode="lastp".
    title : str or None, default None
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    
    # Compute linkage matrix
    Z = linkage(X, method=linkage_method, metric=metric)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot dendrogram
    dendrogram(
        Z,
        truncate_mode=truncate_mode,
        p=p,
        ax=ax,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True,
    )
    
    ax.set_xlabel("Sample index or (cluster size)")
    ax.set_ylabel("Distance")
    
    if title:
        ax.set_title(title)
    else:
        linkage_display = linkage_method.title()
        ax.set_title(f"Dendrogram ({linkage_display} linkage)")
    
    fig.tight_layout()
    return fig, ax


def find_optimal_clusters_dendrogram(
    X: np.ndarray,
    linkage_method: str = "ward",
    metric: str = "euclidean",
    max_clusters: int = 10,
) -> Dict[str, Any]:
    """
    Suggest optimal number of clusters by analyzing dendrogram distances.
    
    Parameters
    ----------
    X : ndarray
    linkage_method : str, default "ward"
    metric : str, default "euclidean"
    max_clusters : int, default 10
    
    Returns
    -------
    results : dict
        Contains:
        - "linkage_matrix": The computed linkage matrix
        - "distances": List of merge distances
        - "suggested_clusters": Suggested number of clusters
        - "distance_threshold": Suggested distance threshold
        - "distance_differences": Differences between consecutive merges
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    
    # Compute linkage matrix
    Z = linkage(X, method=linkage_method, metric=metric)
    
    # Extract merge distances (last column of linkage matrix)
    distances = Z[:, 2]
    
    # Find large jumps in distance (potential optimal cuts)
    if len(distances) > 1:
        distance_diffs = np.diff(distances)
        # Normalize differences
        normalized_diffs = distance_diffs / distances[:-1] if distances[0] > 0 else distance_diffs
        
        # Find the largest normalized jump
        if len(normalized_diffs) > 0:
            max_jump_idx = np.argmax(normalized_diffs)
            suggested_clusters = len(X) - max_jump_idx - 1
            suggested_clusters = max(2, min(suggested_clusters, max_clusters))
        else:
            suggested_clusters = 2
    else:
        suggested_clusters = 2
    
    # Ensure suggested clusters is reasonable
    suggested_clusters = max(2, min(suggested_clusters, max_clusters))
    
    # Suggested distance threshold (distance at which to cut)
    if len(distances) >= suggested_clusters:
        distance_threshold = distances[-suggested_clusters]
    else:
        distance_threshold = distances[0] if len(distances) > 0 else 0.0
    
    return {
        "linkage_matrix": Z,
        "distances": distances,
        "suggested_clusters": suggested_clusters,
        "distance_threshold": distance_threshold,
        "distance_differences": distance_diffs if len(distances) > 1 else np.array([]),
    }


def silhouette_agglomerative(
    X: np.ndarray,
    k_range: List[int],
    linkage: str = "ward",
    metric: str = "euclidean",
) -> Dict[int, float]:
    """
    Compute silhouette scores for agglomerative clustering across k values.
    
    Parameters
    ----------
    X : ndarray
    k_range : list of int
        Range of k values to test.
    linkage : str, default "ward"
    metric : str, default "euclidean"
    
    Returns
    -------
    scores : dict
        Mapping from k to silhouette score.
    """
    from sklearn.metrics import silhouette_score
    
    scores = {}
    
    for k in k_range:
        if k < 2 or k > len(X):
            scores[k] = np.nan
            continue
        
        try:
            labels, _ = agglomerative_clustering(
                X, n_clusters=k, linkage=linkage, metric=metric
            )
            score = silhouette_score(X, labels)
            scores[k] = float(score)
        except Exception:
            scores[k] = np.nan
    
    return scores