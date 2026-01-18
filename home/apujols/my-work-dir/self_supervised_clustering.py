import os 
import numpy as np 
import pandas as pd 
import faiss 
import argparse

from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

version = "1.1"

parser = argparse.ArgumentParser()
parser.add_argument("--backbone_dim", type=int, default=64)
parser.add_argument("--best_clustering", type=int, default=0)
args = parser.parse_args()

BACKBONE_DIM = args.backbone_dim
BEST_CLUSTERING = args.best_clustering

image_dir = "../../../data/upftfg26/apujols/processed/sum_image_cropped"
csv_file = "../../../data/upftfg26/apujols/processed/dataset_temp.csv"

def save_scatter_plot(K, features_pca, labels):
    plt.figure(figsize=(8, 6)) 
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap="tab20", s=10) 
    plt.title(f"PCA Scatter (K={K})") 
    plt.xlabel("PCA 1") 
    plt.ylabel("PCA 2") 
    plt.colorbar(label="Cluster") 
    plt.tight_layout() 
    plt.savefig(f"logs/clustering/scatter_K{K}_v{version}_{BACKBONE_DIM}.png", dpi=200) 
    plt.close()

def save_evaluation_curves(K_values, elbow_sse, silhouette_scores, dbi_scores):
    plt.figure(figsize=(8, 6)) 
    plt.plot(K_values, elbow_sse, marker="o") 
    plt.title("Elbow Curve (SSE vs K)") 
    plt.xlabel("K") 
    plt.ylabel("SSE") 
    plt.grid(True) 
    plt.savefig(f"logs/clustering/elbow_v{version}_{BACKBONE_DIM}.png", dpi=200) 
    plt.close() 
    
    plt.figure(figsize=(8, 6)) 
    plt.plot(K_values, silhouette_scores, marker="o") 
    plt.title("Silhouette Score vs K") 
    plt.xlabel("K") 
    plt.ylabel("Silhouette Score") 
    plt.grid(True) 
    plt.savefig(f"logs/clustering/silhouette_v{version}_{BACKBONE_DIM}.png", dpi=200) 
    plt.close() 
    
    plt.figure(figsize=(8, 6)) 
    plt.plot(K_values, dbi_scores, marker="o") 
    plt.title("Davies-Bouldin Index vs K") 
    plt.xlabel("K") 
    plt.ylabel("DBI") 
    plt.grid(True) 
    plt.savefig(f"logs/clustering/dbi_v{version}_{BACKBONE_DIM}.png", dpi=200) 
    plt.close()

def save_silhouette_diagram_bestK(features, best_labels, best_K):

    # Compute silhouette scores for each sample
    sil_samples = silhouette_samples(features, best_labels)
    avg_score = np.mean(sil_samples)

    plt.figure(figsize=(10, 6))
    y_lower = 10

    for c in range(best_K):
        cluster_vals = sil_samples[best_labels == c]
        cluster_vals.sort()

        size = len(cluster_vals)
        y_upper = y_lower + size

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_vals,
            alpha=0.7,
            label=f"Cluster {c}"
        )

        plt.text(-0.05, y_lower + size / 2, str(c))
        y_lower = y_upper + 10

    plt.axvline(avg_score, color="red", linestyle="--", label=f"Avg = {avg_score:.3f}")
    plt.title(f"Silhouette Plot for K={best_K}")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"logs/clustering/silhouette_diagram_K{best_K}_v{version}_{BACKBONE_DIM}.png", dpi=200)
    plt.close()


if __name__ == "__main__":

    # Load self-supervised output
    features = np.load(f"ssl_features_v{version}_{BACKBONE_DIM}.npy").astype("float32")
    filenames = np.load(f"ssl_filenames_v{version}_{BACKBONE_DIM}.npy")

    # Load the csv with the samples information
    df_labels = pd.read_csv(csv_file, sep=";")

    # Filter to only the files we have the image from
    valid_files = set(os.path.splitext(f)[0].replace("_CROP_SUMIMG", "") for f in os.listdir(image_dir) if f.lower().endswith(".png"))

    df_labels = df_labels[df_labels["filename"].isin(valid_files)].copy()
    df_labels = df_labels.sort_values("filename").reset_index(drop=True)

    # Align SSL filenames with the CSV
    df_ssl = pd.DataFrame({"filename": filenames})
    df_merged = df_ssl.merge(df_labels, on="filename", how="inner")

    # Mask features to match filtered filenames
    mask = df_ssl["filename"].isin(df_merged["filename"])
    features = features[mask.values]
    filenames = df_merged["filename"].values
    classes = df_merged["class"].values

    print(f"Loaded {len(features)} aligned samples")
    d = features.shape[1]
    print(f"Features dimension: {d}")

    # PCA for visualization + metrics
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    if BEST_CLUSTERING == 0:

        print(f"Computing clustering for ALL K")
        K_values = list(range(2, 21))
        elbow_sse = []
        silhouette_scores = []
        dbi_scores = []
        all_cluster_stats = []
        
        for K in K_values:
            print(f"\n=== Running FAISS KMeans for K={K} ===")
            
            kmeans = faiss.Kmeans(d, K, niter=50, verbose=True, gpu=True, seed=42)

            kmeans.train(features)
            _, labels = kmeans.index.search(features, 1)
            labels = labels.astype(int).flatten()

            # Stats
            df_stats = pd.DataFrame({"class": classes, "cluster": labels})
            cluster_stats = (df_stats.groupby("cluster")["class"].value_counts(normalize=True).rename("percentage").mul(100).reset_index())
            cluster_stats["K"] = K
            all_cluster_stats.append(cluster_stats)

            print("Cluster composition (%):")
            print(cluster_stats)

            # Elbow
            centroids = kmeans.centroids
            sse = np.sum((features - centroids[labels]) ** 2)
            elbow_sse.append(sse)

            # Silhouette score
            sil = silhouette_score(features, labels)
            silhouette_scores.append(sil)

            # David-Bouldin Index
            dbi = davies_bouldin_score(features, labels)
            dbi_scores.append(dbi)

            print(f"K={K} | SSE={sse:.2f} | Silhouette={sil:.4f} | DBI={dbi:.4f}")

            df_temp = pd.DataFrame({
                "filename": filenames,
                "class": classes,
                "cluster": labels
            })
            df_temp.to_csv(f"logs/clustering/clusters_K{K}_v{version}_{BACKBONE_DIM}.csv", sep=";", index=False)

            save_scatter_plot(K, features_pca, labels)

        # Save stats
        df_all_stats = pd.concat(all_cluster_stats, ignore_index=True)
        df_all_stats.to_csv(f"logs/clustering/cluster_stats_ALLK_v{version}_{BACKBONE_DIM}.csv", sep=";", index=False)

        # Save plots
        save_evaluation_curves(K_values, elbow_sse, silhouette_scores, dbi_scores)

        # Save full merged dataframe with the cluster for best K
        best_K = K_values[np.argmax(silhouette_scores)] 
        print(f"\nBest K by silhouette: {best_K}")

    else:

        print(f"Recomputing best clustering with K={BEST_CLUSTERING}")
        best_K = BEST_CLUSTERING
        kmeans_best = faiss.Kmeans(d, best_K, niter=100, verbose=False, gpu=True, seed=42) 
        kmeans_best.train(features) 
        _, best_labels = kmeans_best.index.search(features, 1) 
        best_labels = best_labels.flatten()

        df = pd.DataFrame({ 
            "filename": filenames, 
            "class": df_labels["class"].values, 
            "cluster": best_labels })

        feature_cols = [f"f{i}" for i in range(features.shape[1])] 
        df_features = pd.DataFrame(features, columns=feature_cols) 
        df_full = pd.concat([df, df_features], axis=1)

        print("Plotting silhouette diagram...")
        save_silhouette_diagram_bestK(features, best_labels, best_K)

        df_full.to_csv(f"logs/clustering/ssl_clustering_bestK{best_K}_v{version}_{BACKBONE_DIM}.csv", index=False)