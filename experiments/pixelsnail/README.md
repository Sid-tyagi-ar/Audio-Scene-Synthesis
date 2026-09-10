# PixelSNAIL on mel-spectrogram images

An autoregressive image model applied to mel spectrograms rendered as JPEGs.
Instead of generating audio (or a learned latent) directly, this experiment
renders each clip to a log-mel spectrogram picture, trains a PixelCNN family
model over the pixels, and samples new spectrogram images from it.

The model files follow the reference implementations they cite in their own
headers — [openai/pixel-cnn](https://github.com/openai/pixel-cnn) for PixelCNN++
and [neocxi/pixelsnail-public](https://github.com/neocxi/pixelsnail-public) for
PixelSNAIL. The mel rendering and the spectrogram dataset path are ours.

## Files

| File | Purpose |
|---|---|
| `main.py` | Training / evaluation / sampling entry point. Three subcommands select the architecture: `pixelcnn`, `pixelcnnpp`, `pixelsnail`. |
| `pixelcnn.py`, `pixelcnnpp.py`, `pixelsnail.py` | The three model definitions. |
| `optim.py` | Learning-rate schedule and Polyak/EMA parameter averaging. |
| `generate.py` | Standalone sampling from a saved checkpoint. |
| `MelSpectro_multifile.py` | Renders a folder tree of audio into log-mel spectrogram JPEGs — the dataset builder for everything above. |
| `Melspecto.py` | Single-folder predecessor of the above. **Does not parse**: the Windows paths on lines 21 and 23 are plain (not raw) strings, so `\U` in `\Users` raises a `SyntaxError`. Use `MelSpectro_multifile.py` instead, or add the `r` prefix. |
| `pixelsnail.ipynb` | One cell recording the training invocation that was used. |

## Running it

```shell
# From inside this directory
python main.py --train --dataset abstract --batch_size 16 --lr 35e-5 \
    pixelsnail --n_channels 160 --attn_dv 96
```

`--evaluate` and `--generate` replace `--train`; `--restore_file` resumes from a
checkpoint. `./scripts/setup_data.sh --pixelsnail-ckpt` (run from the repository
root) fetches our final checkpoint to
`results/pixelsnail/64-40-epochs/checkpoint_41_model2.pt`.

**Paths need editing first.** The data location is hard-coded rather than taken
from `--data_path`:

- `main.py` lines 149–150 — `ImageFolder(...)` for the train and validation
  splits, pointing at a local Windows directory.
- `main.py` lines 367–369 — optimiser and scheduler state paths on resume.
- `generate.py` line 110 — `model_path` for the checkpoint to sample from.
- `MelSpectro_multifile.py` lines 36–37 — `input_path` / `output_path`.

Point these at your own spectrogram folder (an `ImageFolder` layout: one
subdirectory per class).

`fetch_dataloaders()` also keeps `mnist`, `cifar10` and `colored-mnist` branches
from the reference implementation, but they are unreachable as written: line 68
declares `--dataset` as `choices='abstract'`, and because `choices` is a *string*
argparse accepts only substrings of `"abstract"`. Widen it to a list to use them.

## Capacity presets

`main.py` ships the largest, latest set of `pixelsnail` argparse defaults.
Three earlier capacity presets were used during the project; they differed
*only* in those defaults, so rather than carry near-duplicate files they are
recorded below and can be reproduced with command-line flags.

| Preset | `n_channels` | `n_res_layers` | `attn_n_layers` | `attn_nh` | `attn_dv` | `attn_drop_rate` |
|---|---|---|---|---|---|---|
| **`main.py` (kept)** | 256 | 5 | 12 | 4 | 64 | 20 |
| `pixelsnail_full` | 256 | 4 | 12 | 1 | 128 | 0 |
| `pixelsnail_semifull` | 192 | 4 | 12 | 1 | 128 | 0 |
| `pixelsnail` (smallest) | 128 | 4 | 10 | 1 | 64 | 0 |

`attn_dq` (16) and `n_logistic_mix` (10) are the same in all four. To train the
smallest preset, for example:

```shell
python main.py --train --dataset abstract pixelsnail \
    --n_channels 128 --n_res_layers 4 --attn_n_layers 10 --attn_nh 1 \
    --attn_dv 64 --attn_drop_rate 0
```
