# Experiments

Side explorations from this project, kept for completeness. Each directory is
self-contained and independent of the main `audioldm_train/` model — they were
the routes we tried before settling on the refined Audio-LDM described in the
[top-level README](../README.md).

| Directory | What it is | Generative approach |
|---|---|---|
| [dcase2023_task7/](dcase2023_task7/) | Our fork of the DCASE 2023 Task 7 Foley sound synthesis baseline | VQ-VAE + PixelSNAIL, plus a conditional VAE and a latent DDPM we added |
| [pixelsnail/](pixelsnail/) | Standalone PixelCNN / PixelCNN++ / PixelSNAIL trained directly on mel-spectrogram *images* | Autoregressive over pixels |
| [swin_embeddings/](swin_embeddings/) | Audio → spectrogram-image preprocessing used to prepare data for Swin/HTSAT embedding work | (preprocessing only) |

None of these are importable from `audioldm_train`; they are run from inside
their own directory, since they use relative paths for data and checkpoints.

## Getting the data and weights

Both model directories need assets that are not in git:

```shell
# From the repository root
./scripts/setup_data.sh --experiments        # everything below, ~4.2 GB

./scripts/setup_data.sh --dcase-dataset      # DCASE'23 Task 7 Foley dev set   (~4 GB)
./scripts/setup_data.sh --dcase-checkpoints  # cVAE / VQ-VAE / PixelSNAIL / vocoder (~190 MB)
./scripts/setup_data.sh --pixelsnail-ckpt    # standalone PixelSNAIL, epoch 41 (~17 MB)
```

The checkpoints we trained come from
[**Sukhvansh/audio-scene-synthesis**](https://huggingface.co/Sukhvansh/audio-scene-synthesis)
on the Hugging Face Hub; the DCASE HiFi-GAN vocoder comes from the upstream
baseline repository, and the dataset from Zenodo.

> **Dataset licence.** The DCASE 2023 Task 7 development set
> ([Zenodo record 8091972](https://zenodo.org/record/8091972)) mixes UrbanSound8K,
> FSD50K and BBC Sound Effects material. The BBC portion was provided for the
> DCASE challenge under the condition that it is used for **research purposes
> only**. Check `DevMeta.csv` in the archive for per-clip provenance before
> reusing any of it.

---

## dcase2023_task7

A fork of
[DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline](https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline),
which is itself derived from
[liuxubo717/sound_generation](https://github.com/liuxubo717/sound_generation)
(Liu et al., "Conditional Sound Generation Using Neural Discrete Time-Frequency
Representation Learning", 2021) with the HiFi-GAN generator from
[jik876/hifi-gan](https://github.com/jik876/hifi-gan).

The baseline is a two-stage pipeline: a multi-scale VQ-VAE learns a discrete
time-frequency representation (DTFR) of 4-second clips, then a class-conditional
PixelSNAIL models the prior over those codes, and HiFi-GAN vocodes the decoded
mel back to audio. `README.md` in that directory is the upstream document and
still describes the correct four-step workflow.

**What we added on top of the baseline:**

- `cVAE_improved.py` — a conditional VAE (`CVAE`) over the DTFR codes, replacing
  the discrete VQ-VAE prior with a continuous class-conditioned latent.
- `LDM.py`, `LDM2.py` — class-conditional latent diffusion over the same codes,
  with classifier-free guidance (`sample(..., guide_w=...)`). The two differ in
  backbone: `LDM.py` uses an Inception-style block, `LDM2.py` a residual conv
  U-Net.
- `pixelsnail_trans.py` + `train_pixelsnail_trans.py` — a transformer variant
  (`PixelSNAILTransformer`) of the stage-2 prior.
- `inference_1.py`, `inference_without_pixelsnail.py`, `train_pixelsnail_1.py` —
  inference/training variants used while iterating; the second decodes VQ-VAE
  codes directly, skipping the autoregressive prior.

**What we changed in the baseline files** (everything else differs only in line
endings):

| File | Change |
|---|---|
| `datasets.py` | lower-cased the class dictionary, split paths on `\` as well as `/` for Windows, and point the walk at `DCASEFoleySoundSynthesisDevSet(1)` |
| `pixelsnail.py`, `inference.py` | halved the stage-2 capacity: 256 → 128 channels, 4 → 2 residual/conditioning blocks |
| `extract_code.py` | default VQ-VAE checkpoint `vqvae_046.pt`, LMDB `map_size` reduced from 100 GB to 800 MB |
| `train_pixelsnail.py` | derive `class_id` from the containing folder name, batch size 8 → 1 |

**Before running it, be aware:**

- `cVAE_improved.py` loads its checkpoint with a hard-coded Windows path
  (`checkpoint\cvae_improved\...`, line 125) and calls `train_epoch()` at import
  time, so importing the module starts training. Change the separator to `/` on
  Linux and macOS.
- `LDM.py` and `LDM2.py` resume from a hard-coded epoch (`range(45, n_epoch)` and
  `range(520, n_epoch)`); set those back to `0` for a fresh run.
- The stage-2 scripts read the extracted codes from `vqvae-code/`, an LMDB
  produced by `extract_code.py`. It is derived data, so it is neither in git nor
  in the setup script — regenerate it with step 2 of the upstream workflow.
- `datasets.py` also references `UrbanSound8K/metadata/UrbanSound8K.csv` in
  `get_salience()`; that path is inherited from the original
  `liuxubo717/sound_generation` code, and its only call site (in `audio2mel.py`)
  is commented out, so the file is not needed.

`test.ipynb` and `model_graph.pdf` are a scratch notebook and a rendered model
graph from the same work.

---

## pixelsnail

An unconditional/class-conditional autoregressive image model applied to mel
spectrograms rendered as JPEGs, rather than to audio or to a learned latent.
See [pixelsnail/README.md](pixelsnail/README.md).

---

## swin_embeddings

Four preprocessing utilities for turning audio folders into spectrogram image
datasets and the annotation files that go with them. See
[swin_embeddings/README.md](swin_embeddings/README.md).
