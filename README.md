# DCGAN Implementation

## Overview
This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** using PyTorch. DCGAN is a type of GAN that leverages convolutional and convolutional-transpose layers, making it particularly effective for generating high-quality images. The network has two components:
- A **Generator** model that learns to produce realistic images.
- A **Discriminator** model that evaluates the authenticity of generated images.

## Key Features
- **Generator Network**: Takes a random noise vector and generates an image.
- **Discriminator Network**: Takes an image (real or generated) and classifies it as real or fake.
- **Loss Function**: Binary Cross-Entropy (BCE) loss used for both generator and discriminator training.
- **Optimization**: Uses Adam optimizer for both networks with custom learning rate and betas.

## Steps in the Notebook

### 1. **Imports and Configuration**
- The necessary libraries such as PyTorch, torchvision, and others are imported.
- Hyperparameters such as batch size, learning rate, number of epochs, and latent space dimensions are defined.
- CUDA is enabled for GPU-based training if available.

### 2. **Dataset Preparation**
- The image dataset is loaded using `torchvision.datasets.ImageFolder`.
- Images are preprocessed (resized, normalized) using `torchvision.transforms`.
- A DataLoader is created to feed the images in batches to the model during training.

### 3. **Model Definition**
- **Generator**: 
  - A transposed convolutional network that upsamples random noise into an image.
  - Batch normalization and ReLU activations are used in intermediate layers.
  - The final layer uses Tanh to scale the output between -1 and 1.
- **Discriminator**:
  - A convolutional neural network that downsamples the input image and classifies it as real or fake.
  - Leaky ReLU activations are used for better gradient flow in the discriminator.
  - The output is a probability score using a sigmoid activation function.

### 4. **Weight Initialization**
- Custom weight initialization using a normal distribution is applied to the models to stabilize training.

### 5. **Training Loop**
- For each epoch:
  - The discriminator is trained on both real and fake images, computing real and fake losses.
  - The generator is trained to generate better fake images by attempting to "fool" the discriminator.
  - Both networks' gradients are updated using the Adam optimizer.
  
---

## Instructions for Running the Code

1. **Prerequisites**:
   - Install PyTorch, torchvision, and other dependencies. Ensure you have CUDA installed if using a GPU.

   ```bash
   pip install torch torchvision
   ```

2. **Prepare Dataset**:
   - Place your image dataset in a directory with subfolders for each class (if applicable).
   - Modify the path in the dataset loading section to point to your dataset.


# ---------------------------  GAN Architecture ------------------------------ #

## Project Overview

This notebook implements an image generation model using a Generative Adversarial Network (GAN). The GAN consists of two components: a Generator and a Discriminator. The model was trained to generate images based on random noise input.

## Step-by-Step Process

### 1. **Setup and Imports**
The necessary libraries, such as PyTorch, NumPy, and torchvision, are imported to facilitate data loading, image transformations, model building, and training.

### 2. **Defining Hyperparameters**
Key hyperparameters like the number of epochs, learning rate, batch size, and latent dimension (size of the noise vector input) are defined to control the training process.

### 3. **Data Preprocessing**
- **Dataset Loading**: The dataset of images is loaded using the `ImageFolder` class from the torchvision library.
- **Image Transformation**: Images are resized and normalized to prepare them for the model. The transformations include resizing images to 64x64 and normalizing them to a [-1, 1] scale.
- **DataLoader**: A `DataLoader` is created to handle batching and shuffling of the dataset, ensuring efficient training.

### 4. **Model Architecture**
- **Generator**: The Generator is a neural network that takes random noise as input and produces an image as output. It uses transposed convolutional layers to upsample the noise into a larger image.
- **Discriminator**: The Discriminator is a neural network that evaluates whether an image is real or generated. It uses convolutional layers to downsample the image and outputs a probability score.

### 5. **Weight Initialization**
Both the Generator and Discriminator models are initialized using custom weight initialization techniques to ensure stable training.

### 6. **Loss Function and Optimizer**
- **Loss Function**: Binary Cross-Entropy loss is used for both the Generator and Discriminator, where the Generator tries to minimize the loss of fooling the Discriminator, and the Discriminator tries to maximize its accuracy in distinguishing real from fake images.
- **Optimizers**: The Adam optimizer is applied to both models with custom learning rates and beta values for smooth gradient updates.

### 7. **Training Process**
- The training loop runs for the predefined number of epochs:
  - **Discriminator Training**: The Discriminator is trained on both real images from the dataset and fake images generated by the Generator. It calculates the loss based on how well it distinguishes between the two.
  - **Generator Training**: The Generator is trained to produce more realistic images by trying to fool the Discriminator. Its loss is based on how well the Discriminator is tricked.
  
### 8. **Model Checkpointing**
After each epoch, the Generator and Discriminator models are saved as `.pth` files. These files contain the weights of the trained models and can be used later for fine-tuning or generating new images.

### 9. **Results Visualization**
At various points in training, the generated images are visualized to assess the progress of the Generator. Images are produced by passing random noise vectors through the Generator and displaying the output.

### 10. **Saving the Final Model**
At the end of the training process, the final Generator and Discriminator models are saved for future use, here named as `generator.pth`, `discriminator.pth`.

---

## Conclusion

This notebook successfully builds and trains a GAN model to generate images. The process includes loading and preprocessing a dataset, defining the Generator and Discriminator architectures, training the models, and saving the final weights.

3. **Run the Notebook**:
   - Start training by running all cells in the notebook.
   - Training logs, including generator and discriminator losses, will be displayed for each epoch.
   
---
