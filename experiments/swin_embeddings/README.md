# Swin embedding preprocessing

Utilities for turning a folder tree of scene recordings into the
spectrogram-image dataset and annotation files needed to train a Swin
Transformer for acoustic scene embeddings. This was the "Swin embedding
generation" strand of the project, which we did not finish — only the data
preparation stage is here, there is no training script.

The ten scene classes the scripts iterate over (`Airport`, `Bus`, `Metro`,
`Metro_Station`, `Park`, `Public_Square`, `Shopping_Mall`, `Street_Pedestrian`,
`Street_Traffic`, `Tram`) are those of
[TAU Urban Acoustic Scenes](https://dcase.community/challenge2020/task-acoustic-scene-classification),
the DCASE acoustic scene classification dataset.

## Files

| File | Purpose |
|---|---|
| `Audio2Mel.py` | Renders each clip to a log-mel spectrogram image (`n_fft=2048`, `hop_length=512`, `n_mels=40`) via `librosa` + `matplotlib`, saved with axes stripped. |
| `Audio2Gamma.py` | The same, but through a hand-built gammatone filterbank (`custom_gammatone_filterbank`, 40 filters over 0–16 kHz at 44.1 kHz). Also has `Freq_max_min_Class` for inspecting the per-class frequency range. |
| `Image_Resizer.py` | Pads and resizes the rendered images to a fixed square (default 1200×1200) while preserving aspect ratio, using transparent padding. |
| `Annotation File Creator.py` | Walks each class folder and writes `train.txt` / `val.txt` / `test.txt` splits (900 / 200 / 100 files per class, shuffled with `random.seed(69)`). |

Intended order: `Audio2Mel.py` or `Audio2Gamma.py` → `Image_Resizer.py` →
`Annotation File Creator.py`.

## Running them

Each script is a straight-line program with its input and output directories
set as module-level variables at the bottom of the file, all pointing at a local
Windows path. Edit those before running:

| File | Lines to change |
|---|---|
| `Audio2Mel.py` | `output_path` (line 94), and the input path passed to `File_Path_extractor` |
| `Audio2Gamma.py` | `output_path` (line 147), `Range` and filterbank settings (lines 146–148) |
| `Image_Resizer.py` | `output_folder` (line 61) and the folder handed to `file_path_extractor` |
| `Annotation File Creator.py` | the ten `folder_path` assignments (lines 36–63) and the three output paths (lines 72–74) |

They need `librosa`, `matplotlib`, `numpy` and `Pillow`, all four of which are
already resolved in the project's `poetry.lock`, so the environment from
`poetry install` at the repository root is enough.

The split sizes in `Annotation File Creator.py` are absolute counts, not ratios
— a class folder with fewer than 1,200 files will silently produce short splits.
