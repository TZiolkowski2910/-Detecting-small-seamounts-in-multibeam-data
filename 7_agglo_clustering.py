import os
import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import cv2

# Liste der Modelle
models = ['VGG16', 'ResNet50', 'InceptionV3', 'MobileNetV2']


# Funktion zum Anzeigen aller Bilder eines Clusters
def plot_all_images_in_cluster(images, labels, cluster, algorithm_name, model_name):
    cluster_images = np.where(labels == cluster)[0]
    n_images = len(cluster_images)

    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 2, 2))  # Adjust figsize as needed
    if n_images == 1:
        axes = [axes]  # Make it iterable

    for i, img_idx in enumerate(cluster_images):
        img = cv2.imread(images[img_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = axes[i]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(
        rf'C:\Users\Tobi\Desktop\Seamounts\UNet_Input\MSM88\all_images_cluster_{cluster}_{algorithm_name}_{model_name}.png')
    plt.close(fig)


# Ergebnisse visualisieren
def visualize_clusters(images, labels, algorithm_name, model_name, n_clusters=5):
    fig, axes = plt.subplots(n_clusters, 10, figsize=(15, n_clusters * 3))
    axes = axes.flatten()  # Ensure axes is always a flat array

    for cluster in range(n_clusters):
        cluster_images = np.where(labels == cluster)[0]
        n_images = min(10, len(cluster_images))
        selected_images = np.random.choice(cluster_images, n_images, replace=False)

        for i, img_idx in enumerate(selected_images):
            img = cv2.imread(images[img_idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = axes[cluster * 10 + i]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused axes in this row
        for j in range(n_images, 10):
            ax = axes[cluster * 10 + j]
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(rf'C:\Users\Tobi\Desktop\Seamounts\UNet_Input\MSM88\cluster_{algorithm_name}_{model_name}.png')
    plt.close(fig)


# Clustering und Visualisierung für jedes Modell
for model_name in models:
    # Laden der Features und Bildpfade
    with open(rf'C:\Users\Tobi\Desktop\Seamounts\UNet_Input\MSM88\features_{model_name}.pkl', 'rb') as f:
        features, input_img_paths = pickle.load(f)

    # Clustering mit AgglomerativeClustering
    agglo = AgglomerativeClustering(n_clusters=5)
    labels = agglo.fit_predict(features)

    # Anzahl der Bilder in jedem Cluster ausgeben
    unique, counts = np.unique(labels, return_counts=True)
    print(f"AgglomerativeClustering Clusterverteilung für {model_name}: {dict(zip(unique, counts))}")

    visualize_clusters(input_img_paths, labels, 'AgglomerativeClustering', model_name)

    # Visualisiere alle Bilder in jedem Cluster
    #for cluster in range(5):
    #    plot_all_images_in_cluster(input_img_paths, labels, cluster, 'AgglomerativeClustering', model_name)
    #    print(f"Alle Bilder im Cluster {cluster} für {model_name} wurden erfolgreich visualisiert und gespeichert.")
