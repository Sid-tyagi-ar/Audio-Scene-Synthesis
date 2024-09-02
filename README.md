# Refined Audio-LDM for Audio Synthesis

## Overview

This project explores a refined Audio-LDM approach for audio synthesis, focusing on generating high-quality audio representations through improved techniques in latent space. The refined model incorporates advanced components like HT-SAT and GPT-2 for text encoding, aiming to improve scene classification and overall semantic representation of audio spectrograms.
The drive link which encompasses our entire project details :- [drive_link](https://drive.google.com/drive/u/2/folders/1Qni5M2-nzsVK1boZuPVowiq97E1qQ4mW)

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
python3 latent_diffusion.py -c audioldm_custom.yaml
```

## For inference you can use the below command

```shell
python3 infer.py --config_yaml audioldm_custom.yaml --list_inference inference_test.lst --reload_from_ckpt checkpoint.ckpt
```
- The checkpoint file can be found here -> [Drive Link](https://drive.google.com/file/d/1-zWIR3CiNpr75yrP4cByd2KD7lfWSUU5/view?usp=drive_link)


PS: While running inference on CUDA, there is some driver error for the time being which we are trying to fix ASAP. So for the time being inference can taken out from the validation step of the training. To take out custom audios, once can change the textual description in the data folder.

