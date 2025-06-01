import os
import numpy as np
import cv2
import rasterio

# Pfade zu den Eingabedateien
#google_earth_image = r'C:\Users\Tobias\Arbeitsaufgaben\Paper schreiben\Seamounts\input_image.jpg'
google_earth_image = r'C:\Users\Tobias\Arbeitsaufgaben\Paper schreiben\Seamounts\Abbildungen\tiff.png'
#google_earth_image = r'C:\Users\Tobias\Arbeitsaufgaben\Paper schreiben\Seamounts\masked_input_image.jpg'
#binary_mask = '/data6/tziolkowski/seamount/sourceimages/output_mask.tif'

# Ausgabeverzeichnisse für die Kacheln
#output_image_path = r'C:\Users\Tobias\Arbeitsaufgaben\Paper schreiben\Seamounts\Segmented Images\128'
output_image_path = r'C:\Users\Tobias\Arbeitsaufgaben\Paper schreiben\Seamounts\Segmented Images Abbildung'
os.makedirs(output_image_path, exist_ok=True)

# Kachelgröße definieren (64x64 Pixel)
tile_size = 128

# Eingabebild laden
with rasterio.open(google_earth_image) as img_src:
    img = img_src.read(
        out_shape=(
            img_src.count,
            int(img_src.height),
            int(img_src.width)
        )
    )
    img = np.moveaxis(img, 0, -1)  # Kanäle an die letzte Achse verschieben

# Funktion zum Speichern der Kacheln
def save_tile(tile, path, filename):
    cv2.imwrite(os.path.join(path, filename), tile)

# Kacheln erstellen und speichern
def create_tiles(image, tile_size, img_output_path):
    img_height, img_width = image.shape[:2]

    tile_id = 0
    for i in range(0, img_height, tile_size):
        for j in range(0, img_width, tile_size):
            img_tile = image[i:i+tile_size, j:j+tile_size]

            if img_tile.shape[0] == tile_size and img_tile.shape[1] == tile_size:
                img_filename = f'image_tile_{tile_id:04d}.png'
                save_tile(img_tile, img_output_path, img_filename)
                tile_id += 1

create_tiles(img, tile_size, output_image_path)
print(f"Kacheln wurden erfolgreich gespeichert in {output_image_path}")
