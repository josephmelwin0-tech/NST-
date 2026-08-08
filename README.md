---
title: StyleForge AI
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.9.0
app_file: app.py
pinned: false
---

# Real-Time Arbitrary Style Transfer using Adaptive Instance Normalization


This repository contains a PyTorch implementation and a Flask-based web application for real-time arbitrary style transfer. The system is based on the Adaptive Instance Normalization (AdaIN) framework, which enables transferring arbitrary visual styles onto content images in real-time.

## Project Overview

Neural Style Transfer (NST) traditionally requires optimization over individual target style and content images, or training a network for a specific, single style. This project implements an arbitrary style transfer network that runs in real-time. It achieves this by aligning the channel-wise mean and variance of the content image features with those of the style image features at a specific bottleneck layer.

A pre-trained, frozen VGG-19 network acts as the Encoder to extract multi-scale features. A trainable Decoder network is then optimized to invert the stylized AdaIN features back into raw image pixels, producing the final stylized output.

---

## Directory Structure

*   `NST_Code/` - Core source directory containing the web application and training scripts.
    *   `app.py` - Flask web application handling user uploads, configuring inference parameters (such as style strength), and rendering results.
    *   `train.py` - Training pipeline script for optimizing the Decoder network on a custom dataset.
    *   `vgg_normalised.pth` - Normalized weights for the pre-trained VGG-19 Encoder network.
    *   `utils/` - Shared utility scripts.
        *   `models.py` - PyTorch module definitions for the `VGGEncoder` and `Decoder` architectures.
        *   `utils.py` - Core algorithmic utilities, including the `adaptive_instance_normalization` operator, style loss calculations, and dataset classes.
    *   `templates/` - HTML layout templates for the Flask front-end.
    *   `static/` - Static assets including CSS, UI animations, and user file uploads.
    *   `experiment/final_exp/` - Directory containing the final trained model weights (`decoder_final.pth`).
*   `Demo_IO_Images/` - Directory containing sample input/output images demonstrating the performance of the system.
*   `requirements.txt` - File specifying python dependencies for execution.
*   `Procfile.txt` - Deployment configuration file for cloud hosting services (e.g., Heroku).

---

## Technical Details

### 1. Adaptive Instance Normalization (AdaIN)
The core transfer mechanism is the AdaIN layer. Given a content feature map $F_c$ and a style feature map $F_s$, AdaIN normalizes the channel-wise mean and standard deviation of $F_c$ to match those of $F_s$:

$$\text{AdaIN}(F_c, F_s) = \sigma(F_s) \left( \frac{F_c - \mu(F_c)}{\sigma(F_c)} \right) + \mu(F_s)$$

This operation effectively strips the content image of its original style (represented by its mean and variance) and paints it with the style of the reference image.

### 2. Loss Functions
The Decoder is trained using a weighted sum of content loss and style loss:

*   **Content Loss**: The Mean Squared Error (MSE) between the output image features (from the Encoder) and the stylized bottleneck features:
    $$\mathcal{L}_c = \|f(g(t)) - t\|_2$$
    where $t = \text{AdaIN}(f(c), f(s))$, $f$ is the Encoder, and $g$ is the Decoder.
*   **Style Loss**: The sum of MSE differences in channel-wise mean and standard deviation across multiple intermediate layers of the Encoder:
    $$\mathcal{L}_s = \sum_{i=1}^{L} \|\mu(\phi_i(g(t))) - \mu(\phi_i(s))\|_2 + \sum_{i=1}^{L} \|\sigma(\phi_i(g(t))) - \sigma(\phi_i(s))\|_2$$
    where $\phi_i$ denotes the output of layer $i$ in the Encoder.

---

## Getting Started

### Prerequisites
*   Python 3.12 or higher
*   PyTorch (2.2.2 or higher)
*   Torchvision

### Installation
1.  Clone this repository or navigate to its directory.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Web Application
Start the local development server:
```bash
python NST_Code/app.py
```
After the model finishes initializing, open your browser and navigate to `http://localhost:5000`. You can upload a custom content image, pick a style reference, adjust the style strength (alpha slider), and perform the transfer.

### Training the Decoder
To train the decoder network from scratch:
```bash
python NST_Code/train.py --content_dir /path/to/content/dataset --style_dir /path/to/style/dataset --device cuda
```
Available flags:
*   `--content_dir`: Path to the content image dataset directory.
*   `--style_dir`: Path to the style image dataset directory.
*   `--epochs`: Number of training epochs (default is 10).
*   `--batch_size`: Batch size for loading training inputs (default is 4).
*   `--lr`: Optimizer learning rate (default is 1e-4).
*   `--device`: Device to run training on (`cuda` or `cpu`).