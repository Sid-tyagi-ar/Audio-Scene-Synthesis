# Refined Audio-LDM for Audio Synthesis

## Overview

This project explores a refined Audio-LDM approach for audio synthesis, focusing on generating high-quality audio representations through improved techniques in latent space. The refined model incorporates advanced components like HT-SAT and GPT-2 for text encoding, aiming to improve scene classification and overall semantic representation of audio spectrograms.

## Motivation

The standard Audio-LDM (LDMs) faced limitations due to their architecture, primarily regarding the ability to generate semantically meaningful audio representations. Our goal was to enhance the Audio-LDM by integrating better pre-trained models and architectures to improve performance and efficiency.

## Key Components

### 1. Audio Representation
Audio signals are represented in the frequency domain using Mel-Spectrograms, which provide a differentiable scale for both high and low-frequency components. This representation is then used for generating embeddings.

### 2. Refined Audio-LDM
The refined Audio-LDM utilizes:
- **HT-SAT (Hierarchical Transformer - Self-Attention Transformer):** Pre-trained on 22 different audio tasks, this model efficiently captures and represents audio spectrogram features with lower GPU consumption.
- **GPT-2 Text Encoder:** Provides a more robust text encoding mechanism, better suited for mapping text embeddings to the semantic information of spectrograms. 
- **LDM**
- **HIFI-GAN**
  
## Training Strategies:
The refined model was trained from scratch, accommodating the larger input size required by the new components, resulting in a lower loss at 50k iterations compared to the original Audio-LDM, which required 500k iterations without achieving similar improvement.

## Results

The refined LDM achieved:
- **Reduced Training Time:** The refined LDM achieved lower loss in fewer iterations (50k vs. 500k).
- **Improved Audio Quality:** Better semantic representations were generated for the audio spectrograms, leading to higher quality audio synthesis.
 ![Loss Function Graph](https://github.com/Sid-tyagi-ar/Audio-Scene-Synthesis/blob/main/Training%20Loss%20Graph.png)


## Usage 
- **For training the model use the below script** 

```shell
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_custom.yaml
```


