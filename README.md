# -Detecting-small-seamounts-in-multibeam-data

This repository contains all code and scripts used to implement the two-step deep-learning framework for small seamount detection in multibeam bathymetry:

- Image Filtering: Preselect tiles likely to contain seamounts using CNN feature extraction + clustering
- Seamount Detection: Train and apply a U-Net segmentation model to those candidate tiles

Scripts Overview:

filtering/
0_run_all.py: Orchestrates the full filtering workflow
1_segment_images.py: Reads raw XYZ/CSV, segments into chunks
2–5_extractFeatureVectors_*.py: Generate feature vectors with four CNN backbones
6_kmeans_clustering.py: Cluster tiles using K-means (k=10)
7_agglo_clustering.py: (Optional) Hierarchical clustering for comparison

seamount_detection/
1_recalculate_coordinates.py: Convert raw UTM coordinates to lon/lat
2_image_creation.py: Interpolate each chunk, normalize, tile, and apply augmentation
3_train_unet.py: Define, tune, and train the U-Net segmentation model
4_apply_model.py: Load trained model and generate seamount segmentation on new data
