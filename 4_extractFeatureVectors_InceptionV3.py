import os
import numpy as np
import cv2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import shutil
import matplotlib.pyplot as plt

# Verzeichnisse der Bilder
input_dir = r'C:\Users\Tobi\Desktop\Seamounts\UNet_Input\Chunk24\Alle Bilder'
manual_sorted_dir = r'C:\Users\Tobi\Desktop\Seamounts\UNet_Input\Chunk24\Manuelle sortiere Bilder'
output_cluster_dir = r'C:\Users\Tobi\Desktop\Seamounts\UNet_Input\Chunk24\Clustered Bilder InceptionV3'
selected_cluster_dir = r'C:\Users\Tobi\Desktop\Seamounts\UNet_Input\Chunk24\Ausgewählte Cluster InceptionV3'

# Output-Verzeichnisse erstellen
os.makedirs(output_cluster_dir, exist_ok=True)
os.makedirs(selected_cluster_dir, exist_ok=True)

# Bildgröße anpassen
IMG_HEIGHT, IMG_WIDTH = 256, 256
IMG_CHANNELS = 3

# Alle Bilder einlesen
input_img_paths = sorted([
    os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith('.png')
])

# InceptionV3 Modell initialisieren
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
x = base_model(input_layer)
x = Flatten()(x)
model = Model(inputs=input_layer, outputs=x)

# Feature-Vektoren extrahieren
features = []
for path in input_img_paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalisieren
    feature = model.predict(img)
    features.append(feature.flatten())

features = np.array(features)
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Clustering mit K-Means
num_clusters = 15  # Anzahl der Cluster
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(features)
labels = kmeans.labels_

# Bilder nach Clustern sortieren
for cluster_id in range(num_clusters):
    cluster_folder = os.path.join(output_cluster_dir, f'Cluster_{cluster_id}')
    os.makedirs(cluster_folder, exist_ok=True)

for img_path, cluster_id in zip(input_img_paths, labels):
    shutil.copy(img_path, os.path.join(output_cluster_dir, f'Cluster_{cluster_id}', os.path.basename(img_path)))

print(f"Bilder wurden in {num_clusters} Cluster aufgeteilt und gespeichert.")

# Cluster-Bilder anzeigen in einem einzigen Plot mit Bilderanzahl
fig, axes = plt.subplots(num_clusters, 10, figsize=(20, 2 * num_clusters))  # Raster mit 10 Spalten
fig.suptitle("InceptionV3", fontsize=20)

for cluster_id in range(num_clusters):
    cluster_folder = os.path.join(output_cluster_dir, f'Cluster_{cluster_id}')
    img_paths = [os.path.join(cluster_folder, fname) for fname in os.listdir(cluster_folder)]
    num_images_in_cluster = len(img_paths)  # Anzahl der Bilder im Cluster
    img_paths = img_paths[:10]  # Maximal 10 Bilder pro Cluster

    for col, ax in enumerate(axes[cluster_id]):
        if col < len(img_paths):
            img = cv2.imread(img_paths[col])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.axis('off')  # Leere Achsen deaktivieren
        ax.axis('off')  # Achsen deaktivieren

    # Cluster-Titel mit Anzahl der Bilder
    axes[cluster_id][0].set_title(f"Cluster {cluster_id} ({num_images_in_cluster} Bilder)", fontsize=8, loc='left')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Layout anpassen
plt.show()

# Benutzerabfrage für Cluster-Auswahl
print("\nInception - Verfügbare Cluster: 0 bis", num_clusters - 1)
selected_clusters = input("Geben Sie die Nummern der gewünschten Cluster ein (z. B. 0,2,4): ")
selected_clusters = [int(c.strip()) for c in selected_clusters.split(',')]

# Kopiere ausgewählte Cluster
for cluster_id in selected_clusters:
    cluster_folder = os.path.join(output_cluster_dir, f'Cluster_{cluster_id}')
    for fname in os.listdir(cluster_folder):
        shutil.copy(
            os.path.join(cluster_folder, fname),
            os.path.join(selected_cluster_dir, fname)
        )

print("Ausgewählte Cluster wurden kopiert.")

# Vergleich mit manuell sortierten Bildern
manual_sorted_images = set(os.listdir(manual_sorted_dir))
selected_cluster_images = set(os.listdir(selected_cluster_dir))

common_images = manual_sorted_images.intersection(selected_cluster_images)

print(f"\nÜbereinstimmende Bilder zwischen manueller Sortierung und ausgewählten Clustern: {len(common_images)}")
print(f"Gesamtanzahl manuell sortierter Bilder: {len(manual_sorted_images)}")
print(f"Prozentsatz der Übereinstimmungen: {len(common_images) / len(manual_sorted_images) * 100:.2f}%")
