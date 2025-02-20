# Image Colourization with Autoencoders and Conditional GANs

This project implements an image colourization task using both convolutional autoencoders and conditional generative adversarial networks (cGANs). Given a grayscale image, the goal is to predict the colour at each pixel. The dataset used is CIFAR-10, consisting of 32x32 pixel images, which makes it easier to manage training time.

## Project Overview
The project is divided into two major parts:

1. **Autoencoder Approach:**
   - A convolutional autoencoder is used to colourize grayscale images.
   - The architecture is gradually improved by experimenting with different configurations.
   
2. **Conditional GAN Approach:**
   - A conditional generative adversarial network (cGAN) is compared to the autoencoder.
   - The goal is to generate more realistic colourized images by leveraging adversarial training.

### Setup Instructions
1. **Dependencies**:
   - PyTorch
   - NumPy
   - Matplotlib
   - SciPy
   - PIL
   - CUDA (for GPU support)

2. **Data Loading**:
   - CIFAR-10 dataset is used and automatically downloaded the first time the code is run.

### Key Components

#### Part A - Autoencoder
In this part, we build and train a convolutional autoencoder for image colourization.

- **Model Overview**:
  The autoencoder consists of an encoder-decoder architecture where the encoder extracts features and the decoder reconstructs the image.
  
- **Data Preparation**:
  - We select only the "horse" category from CIFAR-10 for simplicity.
  - Images are converted into grayscale to serve as input, with the original RGB images serving as the target output for training.

- **Model Implementation**:
  The autoencoder's architecture includes convolutional layers for both encoding and decoding the images. 

#### Part B - Conditional GAN (cGAN)
In this section, we implement a conditional GAN for image colourization.

- **Generator**:
  The generator is a deep convolutional network that learns to convert grayscale images to colour images, conditioned on the grayscale input.

- **Discriminator**:
  The discriminator is a binary classifier that distinguishes between real and fake colour images, providing feedback to the generator.

- **Training**:
  - We use **Binary Cross-Entropy Loss** for the discriminator and **L1 Loss + BCE Loss** for the generator.
  - The network is trained for multiple epochs, where the generator and discriminator are trained alternately.

### Results
- The autoencoder approach produces reasonable results but struggles to generate highly accurate colourized images.
- The cGAN model significantly improves the colourization quality, generating more realistic and vibrant images. In particular, the generator is able to preserve better spatial relationships and textures.

### Model Architectures
1. **Autoencoder**:
   - A simple convolutional neural network (CNN) with an encoder-decoder structure.
   
2. **cGAN**:
   - A more complex generator (with multiple convolutional layers and skip connections) and discriminator that work adversarially to improve the quality of the generated images.

### Results Comparison
- **Autoencoder**:
  The autoencoder produces good approximations of the colours but lacks the fine details and variety of the cGAN-generated images.

- **Conditional GAN**:
  The cGAN produces highly detailed and accurate colourization, closely resembling the original images. The addition of skip connections in the UNet architecture of the generator improves the model's performance by retaining spatial information.

### Future Work
- Experimenting with different datasets to test generalization.
- Improving model performance using techniques like transfer learning or fine-tuning on specific types of images.
- Incorporating advanced GAN models like CycleGAN for unpaired image-to-image translation tasks.

### Acknowledgments
- CIFAR-10 dataset, available at [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).
- PyTorch for its excellent deep learning framework.
