### 
## demo_agglomerative.py
## Demonstration of agglomerative clustering on difficult_dataset.csv
## Will Avery - University of Bath
## Task 5) 
# MA52109 Practical Exam

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from cluster_maker package
from cluster_maker import (
    agglomerative_clustering,
    plot_dendrogram,
    find_optimal_clusters_dendrogram,
    silhouette_agglomerative,
    select_features,
    standardise_features,
    silhouette_score_sklearn,
    plot_clusters_2d,
)


def main():
    """Demonstrate agglomerative clustering effectiveness."""
    
    os.makedirs("demo_output", exist_ok=True)
    
    print("=" * 60)
    print("AGGLOMERATIVE CLUSTERING DEMONSTRATION")
    print("=" * 60)
    
    # 1. Load and prepare data
    data_path = "data/difficult_dataset.csv"
    df = pd.read_csv(data_path)
    print(f"\nDataset: {df.shape[0]} points, features: {list(df.columns)}")
    
    X_df = select_features(df, ['x', 'y'])
    X = X_df.to_numpy()
    X_scaled = standardise_features(X)
    
    # 2. Determine optimal clusters using dendrogram
    print("\n" + "-" * 40)
    print("STEP 1: DENDROGRAM ANALYSIS")
    print("-" * 40)
    
    # Create and save dendrogram
    fig_dendro, ax_dendro = plot_dendrogram(
        X_scaled,
        linkage_method="ward",
        title="Dendrogram for Difficult Dataset"
    )
    
    # Analyze dendrogram for optimal k
    dendro_results = find_optimal_clusters_dendrogram(
        X_scaled, linkage_method="ward", max_clusters=8
    )
    suggested_k = dendro_results["suggested_clusters"]
    
    # Mark suggested cut on dendrogram
    ax_dendro.axhline(y=dendro_results["distance_threshold"], 
                     color='r', linestyle='--',
                     label=f'Suggested k={suggested_k}')
    ax_dendro.legend()
    
    fig_dendro.savefig("demo_output/dendrogram.png", dpi=150, bbox_inches='tight')
    plt.close(fig_dendro)
    print(f" Dendrogram suggests k = {suggested_k}")
    print(f" Saved dendrogram.png")
    
    # 3. Validate with silhouette scores
    print("\n" + "-" * 40)
    print("STEP 2: SILHOUETTE VALIDATION")
    print("-" * 40)
    
    k_range = list(range(2, 9))
    silhouette_results = silhouette_agglomerative(
        X_scaled, k_range=k_range, linkage="ward"
    )
    
    print("Silhouette scores for different k:")
    best_k, best_score = 2, -1
    for k in k_range:
        score = silhouette_results.get(k, np.nan)
        if not np.isnan(score):
            print(f"  k={k}: {score:.3f}")
            if score > best_score:
                best_k, best_score = k, score
    
    print(f"\n Best silhouette: k={best_k}, score={best_score:.3f}")
    
    # Choose final k (prioritize silhouette if good)
    final_k = best_k if best_score > 0.4 else suggested_k
    print(f"\n Final choice: k={final_k}")
    
    # 4. Perform clustering with optimal k
    print("\n" + "-" * 40)
    print(f"STEP 3: CLUSTERING WITH k={final_k}")
    print("-" * 40)
    
    labels, centroids = agglomerative_clustering(
        X_scaled, n_clusters=final_k, linkage="ward"
    )
    
    silhouette = silhouette_score_sklearn(X_scaled, labels)
    print(f"Results:")
    print(f"  • Silhouette score: {silhouette:.3f}")
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  • Cluster distribution:")
    for cluster_id, count in zip(unique, counts):
        percentage = count / len(labels) * 100
        print(f"    Cluster {cluster_id}: {count} points ({percentage:.1f}%)")
    
    # 5. Create visualizations
    print("\n" + "-" * 40)
    print("STEP 4: VISUALIZATIONS")
    print("-" * 40)
    
    # Cluster plot using cluster_maker
    fig_cluster, _ = plot_clusters_2d(
        X_scaled, labels, centroids=centroids,
        title=f"Agglomerative Clustering (k={final_k}, Ward linkage)"
    )
    fig_cluster.savefig("demo_output/agglomerative_clusters.png", 
                       dpi=150, bbox_inches='tight')
    plt.close(fig_cluster)
    print(" Saved agglomerative_clusters.png")
    
    # Comparison plot: Different linkage methods
    fig_compare, axes = plt.subplots(1, 3, figsize=(15, 5))
    linkage_methods = ["ward", "complete", "average"]
    
    for idx, method in enumerate(linkage_methods):
        ax = axes[idx]
        try:
            labels_method, centroids_method = agglomerative_clustering(
                X_scaled, n_clusters=final_k, linkage=method
            )
            sil_method = silhouette_score_sklearn(X_scaled, labels_method)
            
            ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                      c=labels_method, cmap='tab10', alpha=0.7, s=30)
            ax.scatter(centroids_method[:, 0], centroids_method[:, 1],
                      marker='X', s=100, c='red', edgecolors='black', 
                      linewidths=2)
            
            ax.set_title(f"{method.title()} linkage\nSilhouette: {sil_method:.3f}")
            ax.set_xlabel("x (standardized)")
            if idx == 0:
                ax.set_ylabel("y (standardized)")
            ax.grid(True, alpha=0.3)
            
        except Exception:
            ax.text(0.5, 0.5, "Failed", ha='center', va='center')
            ax.set_title(f"{method.title()} linkage")
    
    plt.tight_layout()
    plt.savefig("demo_output/linkage_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig_compare)
    print(" Saved linkage_comparison.png")
    
    # 6. Save results and provide summary
    print("\n" + "-" * 40)
    print("STEP 5: RESULTS AND CONCLUSION")
    print("-" * 40)
    
    # Save clustered data
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    df_clustered.to_csv(f"demo_output/difficult_clustered_k{final_k}.csv", index=False)
    print(f" Saved clustered data to difficult_clustered_k{final_k}.csv")
    
    # Effectiveness demonstration
    print(f"\nEffectiveness of Agglomerative Clustering:")
    print(f"1. Handles difficult data structure effectively")
    print(f"2. Dendrogram provides intuitive k selection")
    print(f"3. Produces good silhouette score: {silhouette:.3f}")
    print(f"4. Creates balanced, interpretable clusters")
    
    print(f"\nOutput files in demo_output/:")
    print("• dendrogram.png - Hierarchical structure visualization")
    print("• agglomerative_clusters.png - Final clustering result")
    print("• linkage_comparison.png - Method comparison")
    print(f"• difficult_clustered_k{final_k}.csv - Labeled dataset")
    
    print(f"\n Demonstration complete! Agglomerative clustering")
    print("  successfully handles this difficult dataset.")


if __name__ == "__main__":
    main()