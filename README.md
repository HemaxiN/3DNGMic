# 3DNGMic
Official Implementation of "3DNGMic: A Dataset of Real and GAN-Generated 3D Microscopy Images for Nucleus-Golgi Detection, Segmentation and Pairing"

## Abstract

Endothelial cells (ECs), which line the inner surface of blood vessels, play a crucial role in vascular homeostasis and disease progression. Understanding EC polarity, typically computed as a vector between the nucleus and the Golgi complex, is vital for studying EC migration patterns. However, the nucleus-Golgi vectors are often manually annotated on 2D projections of 3D images, which is a laborious process, and the scarcity of large annotated datasets hinders the development of automated methods to compute 3D EC polarity. To address this challenge, we present 3DNGMic, a novel dataset of real and synthetic 3D microscopy images with nucleus-Golgi paired centroid annotations and segmentation masks. The centroids of real images are manually annotated, while synthetic images and their annotations are created automatically. Specifically, we generate coarse 3D segmentation masks from real centroid annotations and train the Vox2Vox generative adversarial network (GAN) to translate these masks into synthetic images. To introduce additional variability, we generate synthetic images with new configurations of nuclei and Golgi by automatically creating nucleus-Golgi centroids within subvolumes, converting these centroids into segmentation masks, and feeding them into the trained Vox2Vox model. We detail the 3DNGPair dataset generation process, including the real nucleus-Golgi annotation methodology and the GAN architecture. By making 3DNGMic publicly available, along with the code for generating additional synthetic data, we aim to advance research in automated EC polarity estimation and consequently improve our understanding of EC migration in both health and disease.

![](https://github.com/HemaxiN/3DNGMic/blob/main/images/example_dataset.JPG)


## Dataset Structure and Contents

Our dataset, named 3DNGMic, is structured into four main folders: `train`, `val`, `test`, `domain shift test`, and `synthetic`. Each of these folders contains four subdirectories:

- **inputs/** - contains segmentation subvolumes of nuclei and Golgi (128×128×64×2); inputs of the Vox2Vox model.

- **outputs/** - contains microscopy image subvolumes (128×128×64×2); real for `domain shift test`, `train`, `val`, and `test`, and generated with a trained Vox2Vox model for `synthetic`.

- **vectors/** - contains the coordinates of the paired nucleus-Golgi centroids (N×6); manually annotated for `train`, `val`, `test`, and `domain shift test`, and automatically generated for `synthetic`.

- **heatmaps/** - contains the Gaussian heatmaps of nuclei and Golgi centroids (128×128×64×2); these heatmaps were created automatically by placing a Gaussian kernel on top of each annotated centroid (Fig. \ref{fig:heatmaps}).


## Instructions

All instructions for visualizing the dataset (in 2D and 3D), training and evaluating the Vox2Vox model for synthetic image generation, generating new data using the trained Vox2Vox model are provided in the notebook [main.ipynb](./main.ipynb).

The centroid detection convolutional neural network (CNN) is presented in [this repository](https://github.com/HemaxiN/Polarity-Vectors).
