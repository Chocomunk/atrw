# Unsupervised Background Clustering

In this project, we use a pre-trained VGG 16 model to generate image feature-vectors that contain background information. The resulting feature vectors are reduced to 512 dimensions using UMAP, and clustered using spectral clustering.

We use our clustering to generate train/val/split tests for a YOLOv5 detection model trained on the Amur Tiger Re-Identification in the Wild (ATRW) to form bounding boxes on tigers in images. These background-separated splits tell us whether the model is overfitting to the backgrounds.

## Training/Testing YOLOv5

Use the notebooks under `models/yolov5` to train/test YOLOv5 on SageMaker. Or, use the `train.py`, `test.py` and `detect.py` scripts.

Our YOLOv5 model is built on the implementation provided from https://github.com/ultralytics/yolov5 (with minimal changes to allow for SageMaker training).

## Clustering

Use the notebooks under `clustering/` to generate and evaluate clusters, and to generate train/val/test splits. Or, use the corresponding python scripts.

The VGG16 Places model is pre-trained by GKalliatakis at https://github.com/GKalliatakis/Keras-VGG16-places365.
