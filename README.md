# Refined Audio-LDM for Audio Scene Synthesis

## Overview

This project explores a refined Audio-LDM approach for audio synthesis, focusing on generating high-quality audio representations through improved techniques in latent space. The refined model incorporates advanced components like HT-SAT and GPT-2 for text encoding, aiming to improve scene classification and overall semantic representation of audio spectrograms.

The repository is a fork of [haoheliu/AudioLDM-training-finetuning](https://github.com/haoheliu/AudioLDM-training-finetuning) with the conditioning stack replaced by [Microsoft CLAP (MS-CLAP)](https://github.com/microsoft/CLAP). See [What we changed](#what-we-changed) for the exact diff against upstream.

## Motivation

The standard Audio-LDM (LDMs) faced limitations due to their architecture, primarily regarding the ability to generate semantically meaningful audio representations. Our goal was to enhance the Audio-LDM by integrating better pre-trained models and architectures to improve performance and efficiency.

## Key Components

### 1. Audio Representation
Audio signals are represented in the frequency domain using Mel-Spectrograms, which provide a differentiable scale for both high and low-frequency components. This representation is then used for generating embeddings.

### 2. Refined Audio-LDM
The refined Audio-LDM utilizes:
- **HT-SAT (Hierarchical Token-Semantic Audio Transformer):** Pre-trained on 22 different audio tasks, this model efficiently captures and represents audio spectrogram features with lower GPU consumption.
- **GPT-2 Text Encoder:** Provides a more robust text encoding mechanism, better suited for mapping text embeddings to the semantic information of spectrograms.
- **LDM** — the latent diffusion U-Net, conditioned via FiLM on a 1024-d MS-CLAP embedding.
- **HiFi-GAN** — 16 kHz / 64-bin vocoder that turns generated mel-spectrograms back into waveforms.

## Training Strategies

The refined model was trained from scratch, accommodating the larger input size required by the new components, resulting in a lower loss at 50k iterations compared to the original Audio-LDM, which required 500k iterations without achieving similar improvement.

## Results

The refined LDM achieved:
- **Reduced Training Time:** The refined LDM achieved lower loss in fewer iterations (50k vs. 500k).
- **Improved Audio Quality:** Better semantic representations were generated for the audio spectrograms, leading to higher quality audio synthesis.

![Loss Function Graph](Training%20Loss%20Graph.png)

The full Weights & Biases training report is included at [docs/wandb-training-report.pdf](docs/wandb-training-report.pdf).

## Audio Samples

Some samples generated using our model: [Samples](https://drive.google.com/drive/folders/1aVqhOB6fJRE1neTBj3mI0BYo2O7TTA-t)

---

## Repository layout

```
audioldm_train/            # model, training, inference and evaluation code
├── config/                # experiment YAMLs (audioldm_custom.yaml is ours)
├── modules/
│   ├── MSCLAP/            # vendored Microsoft CLAP (our conditioning encoder)
│   ├── clap/              # LAION CLAP, used by the AudioLDM baseline
│   ├── diffusionmodules/  # U-Net, attention, samplers
│   ├── latent_diffusion/  # DDPM / DDIM / PLMS / DPM-Solver
│   ├── latent_encoder/    # mel VAE
│   ├── hifigan/           # vocoder
│   └── audiomae/          # AudioMAE (unused by our config)
├── utilities/             # dataloaders, STFT/mel, misc helpers
├── train/                 # latent_diffusion.py, autoencoder.py
├── infer.py               # inference entry point
└── eval.py                # FAD / KL / IS evaluation
data/dataset/metadata/     # AudioCaps caption + label metadata (tracked in git)
experiments/               # side explorations: DCASE Task 7 cVAE/VQ-VAE, PixelSNAIL
tests/                     # caption lists and a dataset/checkpoint validator
scripts/setup_data.sh      # downloads the datasets and weights (see below)
docs/                      # upstream README, model architecture dumps, W&B report
notebooks/                 # Colab training notebook, CLAP autoencoder experiment
```

Model weights and audio are **not** in git — they are fetched by `scripts/setup_data.sh`.

## Setup

### 1. Python environment

```shell
conda create -n audioldm_train python=3.10
conda activate audioldm_train

git clone https://github.com/Sukhvansh2004/Audio-Scene-Synthesis.git
cd Audio-Scene-Synthesis

pip install poetry
poetry install          # or: pip install -r requirements.txt
```

### 2. Datasets and pretrained weights

`scripts/setup_data.sh` downloads every large asset from its original source and
puts it where the configs expect it. Flags can be combined, anything already
present is skipped, and interrupted downloads resume, so it is safe to re-run.

```shell
# Just enough to run inference with our trained model (~13 GB)
./scripts/setup_data.sh --minimal

# Everything, including the AudioCaps audio needed for training (~56 GB)
./scripts/setup_data.sh --all

# Check what is present without downloading anything
./scripts/setup_data.sh --verify
```

| Flag | What it fetches | Size | Original source |
|---|---|---|---|
| `--checkpoints` | Pretrained VAE, AudioMAE, CLAP, 16 kHz + 48 kHz HiFi-GAN | 7.8 GB | `checkpoints.tar` from [AudioLDM-training-finetuning](https://github.com/haoheliu/AudioLDM-training-finetuning#download-checkpoints-and-dataset) |
| `--dataset` | Preprocessed AudioCaps audio (a subset of AudioSet `unbalanced_train_segments`) | 32 GB | `dataset.tar` from the same upstream repo |
| `--trained-ckpt` | **Our refined model**, 70k steps | 5.5 GB | [🤗 Hub](https://huggingface.co/Sukhvansh/audio-scene-synthesis) |
| `--baseline-ckpt` | Our AudioLDM baseline, 500k steps | 4.6 GB | [🤗 Hub](https://huggingface.co/Sukhvansh/audio-scene-synthesis) |
| `--clap-htsat-tiny` | LAION CLAP HTSAT-tiny weights, needed only by `audioldm_original.yaml` | 1.7 GB | [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) |
| `--clap-autoencoder` | CLAP text-embedding autoencoder weights, needed only by `custom_audioldm.yaml` | 256 MB | [🤗 Hub](https://huggingface.co/Sukhvansh/audio-scene-synthesis) |
| `--finetune-ckpts` | Official `audioldm-m-full` / `audioldm-s-full` | ~4 GB | [Zenodo record 7884686](https://zenodo.org/records/7884686) |
| `--checkpoints-alt` | Same as `--checkpoints`, downloaded file by file instead of as one tarball | 7.4 GB | mirror on this project's Drive |

Our own trained checkpoints live on the Hugging Face Hub at
[**Sukhvansh/audio-scene-synthesis**](https://huggingface.co/Sukhvansh/audio-scene-synthesis).
The script uses the `hf` CLI when available and falls back to plain `curl`
against the public resolve endpoint, so no Hugging Face account is needed to
download them. To grab one directly:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("Sukhvansh/audio-scene-synthesis",
                       "audioldm/checkpoint-fad-133.00-global_step=69999.ckpt")
```

MS-CLAP weights (`CLAP_weights_2023.pth`) are not downloaded by the script — the
model pulls them from the [`microsoft/msclap`](https://huggingface.co/microsoft/msclap)
Hugging Face repo on first use.

The AudioCaps caption metadata under `data/dataset/metadata/` is tracked in git,
so it does not need downloading. Its original source is the
[AudioCaps dataset](https://audiocaps.github.io/) (Kim et al., NAACL 2019); the
audio itself comes from [AudioSet](https://research.google.com/audioset/).

> **Important — the tracked split files are debug-sized.** `dataset_root.json`
> as committed points every split at a one-entry index, which is what we were
> using for quick debug runs at the end of the project. Before training on real
> data, edit `data/dataset/metadata/dataset_root.json` to point at the full
> indices:
>
> | split | committed default | entries | full index | entries |
> |---|---|---|---|---|
> | train | `datafiles/audiocaps_train_label.json` | 1 | `datafiles/audiocaps_train_label_orig.json` | 49,502 |
> | val | `testset_subset/audiocaps_test_nonrepeat_subset_0.json` | 1 | `datafiles/audiocaps_val_label_orig.json` | 2,475 |
> | test | `testset_subset/audiocaps_test_nonrepeat_subset_0.json` | 1 | `testset_subset/audiocaps_test_nonrepeat.json` | 964 |
>
> `datafiles/audiocaps_test_orig.json` (4,820 entries) is the full test split with
> all captions. `audiocaps_test_nonrepeat_subset_{1..4}.json` each hold the same
> 964 non-repeated test clips paired with a *different* one of AudioCaps' five
> reference captions — `subset_0` was that set too before we shrank it for
> debugging. `audiocaps_test_nonrepeat_subset_tiny.json` (25 clips) is handy for
> smoke tests.

## Usage

### Training

```shell
python3 audioldm_train/train/latent_diffusion.py \
  -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_custom.yaml
```

`audioldm_custom.yaml` is the refined configuration (MS-CLAP + GPT-2 conditioning).
For reference, `audioldm_original.yaml` reproduces the AudioLDM baseline and
`audioldm_crossattn_flant5.yaml` swaps in FLAN-T5 cross-attention conditioning.
The VAE can be retrained separately with
`audioldm_train/train/autoencoder.py -c audioldm_train/config/2023_11_13_vae_autoencoder/16k_64.yaml`.

Training generates audio for the evaluation set every `validation_every_n_epochs`
epochs into `log/latent_diffusion/<group>/<exp>/val_<step>_.../`.

### Inference

```shell
python3 audioldm_train/infer.py \
  --config_yaml audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_custom.yaml \
  --list_inference tests/captionlist/inference_test.lst \
  --reload_from_ckpt "log/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_custom/checkpoints/checkpoint-fad-133.00-global_step=69999.ckpt"
```

That checkpoint path is where `./scripts/setup_data.sh --trained-ckpt` puts our
model; it is also downloadable directly [here](https://drive.google.com/file/d/1-zWIR3CiNpr75yrP4cByd2KD7lfWSUU5/view?usp=drive_link).
Generated audio is written next to the checkpoint's log folder and named after
the caption. `tests/captionlist/inference_test_with_filename.lst` shows the
format for choosing output filenames yourself, and
`tests/captionlist/inference_test_custom.lst` holds the prompts we used for the
samples linked above. `test.sh` is the SLURM batch version of the same command.

> **Known issue:** inference on CUDA hits a driver error we did not get to the
> bottom of. As a workaround, generate through the validation step of training
> instead (set `step.validation_every_n_epochs` to 1 and edit the textual
> descriptions in the test metadata).

### Evaluation

```shell
python3 audioldm_train/eval.py --log_path all               # every generated folder
python3 audioldm_train/eval.py --log_path <experiment-dir>  # one experiment
```

Results are written as JSON alongside the audio folder. Evaluation needs
[audioldm_eval](https://github.com/haoheliu/audioldm_eval), which `poetry install`
pulls in.

## What we changed

Relative to upstream `haoheliu/AudioLDM-training-finetuning`:

- **Vendored MS-CLAP** at `audioldm_train/modules/MSCLAP/`. `msclap/models/clap3.py`
  is the variant our final config instantiates: an HTSAT audio encoder with a GPT-2
  text encoder, projected to a 1024-d joint embedding, with weights pulled from
  Hugging Face. `clap2.py` (which loads a locally trained embedding autoencoder)
  and `clap4.py` are earlier variants we tried.
- **Two new configs**, both replacing upstream's
  `conditional_models.CLAPAudioEmbeddingClassifierFreev2` conditioning stage:
  - `audioldm_custom.yaml` — the refined model reported above. Uses
    `MSCLAP...clap3.CLAP` and widens `extra_film_condition_dim` from 512 to 1024
    to match MS-CLAP's projection.
  - `custom_audioldm.yaml` — an earlier attempt using `MSCLAP...clap2.CLAP`;
    otherwise it is `audioldm_original.yaml` with `batchsize` 8. Note it still
    declares `extra_film_condition_dim: 512` while asking MS-CLAP for a 1024-d
    projection, so expect to reconcile the two before it runs.
- **Modified upstream files:** `conditional_models.py`, `infer.py`,
  `train/latent_diffusion.py`, `utilities/data/dataset.py`,
  `utilities/audio/stft.py`, `modules/latent_diffusion/ddpm.py`,
  `modules/latent_diffusion/ddim.py`, `modules/audiomae/sequence_gen/model.py`,
  `modules/clap/open_clip/{bert,factory}.py`.
- **Extras:** `tests/captionlist/inference_test_custom.lst`, the AudioCaps
  metadata under `data/dataset/metadata/`, the notebooks in `notebooks/`, and the
  architecture dumps in `docs/`.

The upstream README, which documents the original codebase and its configuration
format in more detail, is kept at [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

## Other approaches we tried

Before settling on the refined Audio-LDM above, we explored three other routes.
They live under [experiments/](experiments/), independent of `audioldm_train/`,
with their own setup instructions in [experiments/README.md](experiments/README.md):

- **[experiments/dcase2023_task7/](experiments/dcase2023_task7/)** — our fork of
  the DCASE 2023 Task 7 Foley sound synthesis baseline (VQ-VAE + PixelSNAIL +
  HiFi-GAN), to which we added a conditional VAE, a class-conditional latent
  diffusion model with classifier-free guidance, and a transformer variant of
  the autoregressive prior.
- **[experiments/pixelsnail/](experiments/pixelsnail/)** — PixelCNN / PixelCNN++ /
  PixelSNAIL trained directly on mel spectrograms rendered as images.
- **[experiments/swin_embeddings/](experiments/swin_embeddings/)** — the
  spectrogram-image and annotation preprocessing for a Swin Transformer scene
  embedding we did not finish.

```shell
./scripts/setup_data.sh --experiments   # data + weights for the above, ~4.2 GB
```

## Project archive

The original working directory for this project, including training logs, W&B
runs and generated audio, lives in Google Drive:
[CS671-DL](https://drive.google.com/drive/folders/1Qni5M2-nzsVK1boZuPVowiq97E1qQ4mW).
Everything needed to reproduce the work has been extracted into this repository.

## Licensing and attribution

This work builds on:

- [AudioLDM](https://github.com/haoheliu/AudioLDM) and
  [AudioLDM-training-finetuning](https://github.com/haoheliu/AudioLDM-training-finetuning)
  (MIT, see [LICENSE](LICENSE)) — Liu et al., ICML 2023
- [Microsoft CLAP](https://github.com/microsoft/CLAP)
  (MIT, see [audioldm_train/modules/MSCLAP/LICENSE](audioldm_train/modules/MSCLAP/LICENSE))
- [LAION CLAP](https://github.com/LAION-AI/CLAP), [HiFi-GAN](https://github.com/jik876/hifi-gan),
  and [stable-diffusion](https://github.com/CompVis/stable-diffusion)

The side experiments additionally build on the
[DCASE 2023 Task 7 baseline](https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline)
and [liuxubo717/sound_generation](https://github.com/liuxubo717/sound_generation);
see [experiments/README.md](experiments/README.md) for the full attribution and
for the research-only restriction on part of the DCASE dataset.

The pretrained AudioLDM checkpoints fetched by `--finetune-ckpts` are released
under CC-BY-NC 4.0 and are not licensed for commercial use.

```bibtex
@article{liu2023audioldm,
  title={{AudioLDM}: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo
          and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={Proceedings of the International Conference on Machine Learning},
  year={2023}
}
```
