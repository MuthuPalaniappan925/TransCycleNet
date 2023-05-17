
# TransCycleNet - CycleGAN

This readme documentation provides an overview of the code repository that implements the CycleGAN model for image translation. The CycleGAN is a deep learning architecture capable of learning to translate images from one domain to another without the need for paired training data. This document describes the purpose, functionality, and usage of the code.


## Table of Contents

- Introduction
- Installation
- Getting Started
- Code Structure
- Usage
- Examples
- Conclusion
- References



## Introduction

The CycleGAN Image Translation repository contains code for training and utilizing a CycleGAN model for image translation tasks. The CycleGAN architecture consists of two generators and two discriminators that work together to learn the mapping between two different image domains. It has been widely used for various image-to-image translation tasks, such as converting Monet-style paintings to realistic photographs.

#### Paired VS Unpaired Data

![paired VS unpaired Data](https://th.bing.com/th/id/R.37c14685c12543b3be22793efd6860e5?rik=dOtsSSXUzccCWg&riu=http%3a%2f%2felastic-ai.com%2fwp-content%2fuploads%2f2019%2f01%2fpaired_unpaired_training_data.png&ehk=U%2fEsvjNzhswFS7n8atA8W2EyrOAyopJ1RGSdFG1%2fLPY%3d&risl=&pid=ImgRaw&r=0)

#### Model - Two ways

![CycleGAN architecture](https://machine-learning-note.readthedocs.io/en/latest/_images/cycle-consistency_loss.png)

The code in this repository is implemented in Python, utilizing the TensorFlow and Keras libraries for deep learning operations. It provides a flexible framework for training and using CycleGAN models on custom datasets.


## Installation

To use the code in this repository, follow these steps

- clone the repository

```bash
gh repo clone MuthuPalaniappan925/Intelligent-Neural-Network-Optimization-with-Evolutionary-Algorithms
```

- Navigate to the repository directory

```bash
cd cycleGAN-image-translation 
```

- Install the required dependencies

```bash
pip install -r requirements.txt
```

- Ensure you have Python 3.7 or above installed.

## Getting Started

Before using the CycleGAN model for image translation, some initial setup is required.

- **Dataset Preparation**: Prepare your own dataset or use the provided Monet-to-Photo dataset (used in the following code). Ensure that the dataset follows the required directory structure and image format.

- **Training Configuration**: Adjust the training configuration parameters in ```monopfo.py``` to match your dataset and training preferences. You can specify the dataset paths, model hyperparameters, and training options.

- **GPU Acceleration (Optional)**: If you have a compatible GPU, make sure to install the necessary drivers and CUDA toolkit to leverage GPU acceleration during training NOTE: Not used in the given code.



## Code Structure



| script | Description                |
| :-------- | :------------------------- |
| `monopho.py` |The main script for dataset loading, model training, and image translation using the CycleGAN.  |
| `instancenormalization.py` |Module implementing the Instance Normalization layer used in the generator and discriminator models. - Used in the base research paper |
| `cycleGAN.py` |Script for defining the discriminator and generator model architecture|




## Usage

The main functionality of the CycleGAN model is accessible through the ```monopfo.py``` script. Here are the main commands

- Training the CycleGAN model

```bash
python monopfo.py --mode Training
```

- Translating the CycleGAN model

```bash
python monopfo.py --model translate
```

Additional options and parameters are available for customization, such as specifying the dataset path, the number of training epochs, and the batch size.


## Conclusion

The CycleGAN Image Translation repository provides a comprehensive implementation of the CycleGAN model for image translation tasks. It allows users to train their own models on custom datasets or use pre-trained models for image translation. The code is modular and flexible, making it easy to adapt for different image domains and applications.

## References

- CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. Jun-Yan Zhu et al. (2017). https://arxiv.org/abs/1703.10593
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- Keras contribution team: https://github.com/keras-team/keras-contrib
