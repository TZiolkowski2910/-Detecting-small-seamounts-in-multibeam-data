# -Detecting-small-seamounts-in-multibeam-data

This repository contains all code and scripts used to implement the two-step deep-learning framework for small seamount detection in multibeam bathymetry:

- Image Filtering: Preselect tiles likely to contain seamounts using CNN feature extraction + clustering
- Seamount Detection: Train and apply a U-Net segmentation model to those candidate tiles


├── filtering/
│   ├── 0_run_all.py
│   ├── 1_segment_images.py
│   ├── 2_extractFeatureVectors_VGG16.py
│   ├── 3_extractFeatureVectors_ResNet50.py
│   ├── 4_extractFeatureVectors_InceptionV3.py
│   ├── 5_extractFeatureVectors_MobileNetV2.py
│   ├── 6_kmeans_clustering.py
│   └── 7_agglo_clustering.py
│
└── seamount_detection/
    ├── 1_recalculate_coordinates.py
    ├── 2_image_creation.py
    ├── 3_train_unet.py
    ├── 4_apply_model.py

