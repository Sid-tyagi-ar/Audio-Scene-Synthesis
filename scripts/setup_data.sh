#!/usr/bin/env bash
#
# setup_data.sh -- fetch the large assets that are not tracked in git and lay
# them out in the directory structure the training/inference code expects.
#
# Everything downloaded here comes from the upstream projects this work builds
# on (AudioLDM-training-finetuning, AudioCaps/AudioSet, Zenodo) or from the
# project's own Google Drive for the checkpoints we trained ourselves.
#
# Usage:
#   scripts/setup_data.sh --checkpoints        # pretrained VAE/HiFi-GAN/CLAP/AudioMAE (~7.8 GB)
#   scripts/setup_data.sh --dataset            # preprocessed AudioCaps audio       (~32 GB)
#   scripts/setup_data.sh --trained-ckpt       # our refined model, 70k steps       (~5.5 GB)
#   scripts/setup_data.sh --baseline-ckpt      # our AudioLDM baseline, 500k steps  (~4.6 GB)
#   scripts/setup_data.sh --clap-htsat-tiny    # LAION CLAP HTSAT-tiny weights      (~1.7 GB)
#   scripts/setup_data.sh --clap-autoencoder   # CLAP embedding autoencoder weights (~256 MB)
#   scripts/setup_data.sh --finetune-ckpts     # official audioldm-{s,m}-full       (~4 GB)
#   scripts/setup_data.sh --checkpoints-alt    # same as --checkpoints, file by file
#   scripts/setup_data.sh --minimal            # checkpoints + trained-ckpt (inference only)
#   scripts/setup_data.sh --all                # everything for the main model
#   scripts/setup_data.sh --verify             # check the layout without downloading
#
# For the side experiments under experiments/ (see experiments/README.md):
#   scripts/setup_data.sh --dcase-dataset      # DCASE'23 Task 7 Foley dev set      (~4 GB)
#   scripts/setup_data.sh --dcase-checkpoints  # cVAE / VQ-VAE / PixelSNAIL / vocoder (~190 MB)
#   scripts/setup_data.sh --pixelsnail-ckpt    # standalone PixelSNAIL, epoch 41    (~17 MB)
#   scripts/setup_data.sh --experiments        # all three of the above
#
# Flags can be combined. Anything already present is skipped, so the script is
# safe to re-run after an interrupted download.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT_DIR="data/checkpoints"
DATASET_DIR="data/dataset"
LOG_DIR="log/latent_diffusion/2023_08_23_reproduce_audioldm"

# ---------------------------------------------------------------------------
# Asset sources
# ---------------------------------------------------------------------------

# Pretrained component checkpoints (VAE, AudioMAE, CLAP, 16k + 48k HiFi-GAN).
# Original source: https://github.com/haoheliu/AudioLDM-training-finetuning
CHECKPOINTS_TAR_ID="1T6EnuAHIc8ioeZ9kB1OZ_WGgwXAVGOZS"   # checkpoints.tar, 7.8 GB

# The same checkpoints as individual files, mirrored on this project's Drive.
# Used by --checkpoints-alt as a fallback if the upstream tarball is unavailable,
# and by --clap-htsat-tiny, which the tarball does not ship.
CKPT_FILES=(
  "vae_mel_16k_64bins.ckpt:1Um8ECAMkeicsZbaOr37YDfzau1SYxWgl"
  "hifigan_16k_64bins.ckpt:1cyeIag5nnX-vp_2yDvFJ9mUu9MRtPNjT"
  "hifigan_16k_64bins.json:1CcvPX7oWHwqL2nO9p-NLAVq-oKvvbST8"
  "hifigan_48k_256bins.ckpt:1Z-N4ktpUhAcKleh66XIPixiSJc068HwN"
  "hifigan_48k_256bins.json:144iea-zrj2xgBwccJe-j_eS_kKXTtBBH"
  "audiomae_16k_128bins.ckpt:1y3Fx6eIjbLtnP38d9l3PyQ8qtoYybSDi"
  "clap_music_speech_audioset_epoch_15_esc_89.98.pt:1kEbOQFIfln8f10BsEAx3AUHLtUXf8bPR"
)

# LAION CLAP HTSAT-tiny weights, required only by audioldm_original.yaml (the
# AudioLDM baseline config). Upstream project: https://github.com/LAION-AI/CLAP
CLAP_HTSAT_TINY_ID="1OKVfQQlKYX1yOD0T9EHXDZo2OIoOTfvh"    # clap_htsat_tiny.pt, 1.7 GB

# Preprocessed AudioCaps audio (a subset of AudioSet unbalanced_train_segments,
# resampled and segmented). Original source: same upstream repo. The caption
# metadata itself is tracked in git under data/dataset/metadata/.
DATASET_TAR_ID="16J1CVu7EZPD_22FxitZ0TpOd__FwzOmx"       # dataset.tar, 32 GB

# Checkpoints we trained for this project.
TRAINED_CKPT_ID="1-zWIR3CiNpr75yrP4cByd2KD7lfWSUU5"      # refined model,  70k steps
TRAINED_CKPT_NAME="checkpoint-fad-133.00-global_step=69999.ckpt"
BASELINE_CKPT_ID="1VgOPpvNBhBqKi210HrQ2c1GtIkMYCHGN"     # AudioLDM baseline, 500k steps
BASELINE_CKPT_NAME="checkpoint-fad-133.00-global_step=499999.ckpt"

# Weights for the CLAP text-embedding autoencoder experiment. Loaded by
# audioldm_train/modules/MSCLAP/msclap/models/clap2.py, which is used by
# custom_audioldm.yaml only -- audioldm_custom.yaml (our final model) uses clap3.
CLAP_AUTOENCODER_ID="115dIHOuW0X_e-nngNiESEopMIU_Au5pn"  # encoder_model.pth, 256 MB
CLAP_AUTOENCODER_PATH="audioldm_train/modules/MSCLAP/msclap/models/encoder_model.pth"

# Official AudioLDM checkpoints for finetuning, from Zenodo record 7884686.
FINETUNE_URLS=(
  "https://zenodo.org/records/7884686/files/audioldm-m-full.ckpt"
  "https://zenodo.org/records/7884686/files/audioldm-s-full"
)

# --- Side experiments under experiments/ ------------------------------------

DCASE_DIR="experiments/dcase2023_task7"
PIXELSNAIL_DIR="experiments/pixelsnail"

# DCASE 2023 Challenge Task 7 (Foley Sound Synthesis) development set, ~4 GB.
# Original source: https://zenodo.org/record/8091972 -- file names are resolved
# from the Zenodo API at run time so a re-upload does not break this script.
# NOTE: part of this data is BBC Sound Effects material, provided for the DCASE
# challenge and research use only. Check DevMeta.csv for per-clip provenance.
DCASE_ZENODO_RECORD="8091972"

# The HiFi-GAN vocoder for the DCASE baseline ships in the upstream repo itself.
DCASE_HIFIGAN_BASE="https://raw.githubusercontent.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline/main/checkpoint/hifigan"

# Checkpoints we trained for the DCASE experiments, on this project's Drive.
DCASE_CKPTS=(
  "checkpoint/cvae_improved/checkpoint_cvae_improved_no fl 500_6000.pt:1-SZQ6qoYz_vicsweEtiIHgETE2GKb5AX"
  "checkpoint/pixelsnail-final/bottom_050.pt:1qZOn1gDcozOQrCqJRLkxTsSQEZxCxg2K"
  "vqvae_046.pt:1CVUwBZdLSv-223oS6-fsmDFtFAhnlQkS"
)

# Standalone PixelSNAIL run on mel-spectrogram images, final epoch.
PIXELSNAIL_CKPT_ID="1udK8hNemircljzRFKd3qjzq_8Ycdy8XD"   # checkpoint_41_model2.pt, 16 MB
PIXELSNAIL_CKPT_NAME="results/pixelsnail/64-40-epochs/checkpoint_41_model2.pt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed. $2"
}

# Download a public Google Drive file to $2, handling the interstitial
# "virus scan" confirmation page that Drive serves for large files.
# Prefers gdown; falls back to curl if gdown is unavailable or fails.
gdrive_download() {
  local file_id="$1" out="$2"
  mkdir -p "$(dirname "$out")"

  if command -v gdown >/dev/null 2>&1; then
    log "gdown $file_id -> $out"
    if gdown --no-cookies --continue "$file_id" -O "$out" && [ -s "$out" ]; then
      return 0
    fi
    warn "gdown failed for $file_id; falling back to curl"
  fi

  need_cmd curl "Install curl, or 'pip install gdown'."
  need_cmd python3 "Install python3."

  local tmp="$out.part"
  log "curl $file_id -> $out"
  curl -fL --retry 3 --retry-delay 5 -C - \
    "https://drive.usercontent.google.com/download?id=${file_id}&export=download" \
    -o "$tmp"

  # If Drive returned the confirmation page instead of the file, resubmit its form.
  if head -c 4096 "$tmp" | grep -q "Virus scan warning"; then
    log "resolving Drive confirmation token"
    local confirm_url
    confirm_url="$(python3 - "$tmp" <<'PY'
import re, sys, urllib.parse
html = open(sys.argv[1], encoding="utf-8", errors="replace").read()
m = re.search(r'<form[^>]+action="([^"]+)"', html)
action = m.group(1).replace("&amp;", "&") if m else ""
fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', html))
print(f"{action}?{urllib.parse.urlencode(fields)}" if action else "")
PY
)"
    [ -n "$confirm_url" ] || die "could not parse Drive confirmation page for $file_id"
    curl -fL --retry 3 --retry-delay 5 "$confirm_url" -o "$tmp"
  fi

  [ -s "$tmp" ] || die "download produced an empty file for $file_id"
  mv "$tmp" "$out"
}

# ---------------------------------------------------------------------------
# Install steps
# ---------------------------------------------------------------------------

install_checkpoints() {
  if [ -f "$CKPT_DIR/vae_mel_16k_64bins.ckpt" ] && [ -f "$CKPT_DIR/hifigan_16k_64bins.ckpt" ]; then
    log "pretrained checkpoints already present, skipping"
    return
  fi
  log "downloading pretrained component checkpoints (~7.8 GB)"
  mkdir -p "$CKPT_DIR"
  local tar_path="$CKPT_DIR/checkpoints.tar"
  [ -f "$tar_path" ] || gdrive_download "$CHECKPOINTS_TAR_ID" "$tar_path"

  log "extracting into $CKPT_DIR/"
  tar -xf "$tar_path" -C "$CKPT_DIR"
  # The tarball may unpack into a nested checkpoints/ directory; flatten it.
  if [ -d "$CKPT_DIR/checkpoints" ]; then
    mv "$CKPT_DIR/checkpoints/"* "$CKPT_DIR/"
    rmdir "$CKPT_DIR/checkpoints"
  fi
  rm -f "$tar_path"
  log "pretrained checkpoints installed"
}

install_checkpoints_alt() {
  log "downloading pretrained checkpoints file by file (~7.4 GB)"
  mkdir -p "$CKPT_DIR"
  local entry name id
  for entry in "${CKPT_FILES[@]}"; do
    name="${entry%%:*}"; id="${entry##*:}"
    if [ -s "$CKPT_DIR/$name" ]; then
      log "$name already present, skipping"
      continue
    fi
    gdrive_download "$id" "$CKPT_DIR/$name"
  done
  log "pretrained checkpoints installed"
}

install_clap_htsat_tiny() {
  if [ -s "$CKPT_DIR/clap_htsat_tiny.pt" ]; then
    log "clap_htsat_tiny.pt already present, skipping"
    return
  fi
  log "downloading LAION CLAP HTSAT-tiny weights (~1.7 GB)"
  gdrive_download "$CLAP_HTSAT_TINY_ID" "$CKPT_DIR/clap_htsat_tiny.pt"
}

install_dataset() {
  if [ -d "$DATASET_DIR/audioset/zip_audios" ]; then
    log "AudioCaps audio already present, skipping"
    return
  fi
  log "downloading preprocessed AudioCaps audio (~32 GB, this takes a while)"
  mkdir -p "$DATASET_DIR"
  local tar_path="$DATASET_DIR/dataset.tar"
  [ -f "$tar_path" ] || gdrive_download "$DATASET_TAR_ID" "$tar_path"

  log "extracting into $DATASET_DIR/"
  tar -xf "$tar_path" -C "$DATASET_DIR"
  # Keep the metadata that ships with this repo; the tarball also carries a copy.
  if [ -d "$DATASET_DIR/dataset" ]; then
    cp -rn "$DATASET_DIR/dataset/"* "$DATASET_DIR/" 2>/dev/null || true
    rm -rf "$DATASET_DIR/dataset"
  fi
  rm -f "$tar_path"
  log "dataset installed"
}

install_trained_ckpt() {
  local dest="$LOG_DIR/audioldm_custom/checkpoints/$TRAINED_CKPT_NAME"
  if [ -f "$dest" ]; then
    log "refined model checkpoint already present, skipping"
    return
  fi
  log "downloading our refined model checkpoint, 70k steps (~5.5 GB)"
  gdrive_download "$TRAINED_CKPT_ID" "$dest"
  log "checkpoint at $dest"
}

install_baseline_ckpt() {
  local dest="$LOG_DIR/audioldm_original/checkpoints/$BASELINE_CKPT_NAME"
  if [ -f "$dest" ]; then
    log "baseline checkpoint already present, skipping"
    return
  fi
  log "downloading our AudioLDM baseline checkpoint, 500k steps (~4.6 GB)"
  gdrive_download "$BASELINE_CKPT_ID" "$dest"
  log "checkpoint at $dest"
}

install_clap_autoencoder() {
  if [ -f "$CLAP_AUTOENCODER_PATH" ]; then
    log "CLAP autoencoder weights already present, skipping"
    return
  fi
  log "downloading CLAP embedding autoencoder weights (~256 MB)"
  gdrive_download "$CLAP_AUTOENCODER_ID" "$CLAP_AUTOENCODER_PATH"
}

install_finetune_ckpts() {
  need_cmd wget "Install wget."
  mkdir -p "$CKPT_DIR"
  for url in "${FINETUNE_URLS[@]}"; do
    local name="${url##*/}"
    if [ -f "$CKPT_DIR/$name" ]; then
      log "$name already present, skipping"
      continue
    fi
    log "downloading $name from Zenodo"
    wget -c -q --show-progress -O "$CKPT_DIR/$name" "$url"
  done
}

install_dcase_dataset() {
  local dest="$DCASE_DIR/DCASEFoleySoundSynthesisDevSet"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    log "DCASE Foley dev set already present, skipping"
    return
  fi
  need_cmd curl "Install curl."
  need_cmd python3 "Install python3."
  log "resolving DCASE dev set files from Zenodo record $DCASE_ZENODO_RECORD"

  local urls
  urls="$(curl -fsL --retry 3 --retry-delay 5 \
      "https://zenodo.org/api/records/${DCASE_ZENODO_RECORD}" \
    | python3 -c 'import json,sys
rec = json.load(sys.stdin)
for f in rec.get("files", []):
    print(f["links"]["self"], f["key"], sep="\t")' )" || {
    warn "could not reach the Zenodo API."
    warn "Download the development set manually from https://zenodo.org/record/${DCASE_ZENODO_RECORD}"
    warn "and unpack it to $dest"
    return 1
  }
  [ -n "$urls" ] || die "Zenodo record ${DCASE_ZENODO_RECORD} listed no files"

  mkdir -p "$DCASE_DIR/_zenodo"
  local url name
  while IFS=$'\t' read -r url name; do
    [ -n "$name" ] || continue
    if [ -s "$DCASE_DIR/_zenodo/$name" ]; then
      log "$name already downloaded, skipping"
    else
      log "downloading $name"
      curl -fL --retry 3 --retry-delay 5 -C - "$url" -o "$DCASE_DIR/_zenodo/$name"
    fi
  done <<< "$urls"

  log "unpacking into $DCASE_DIR/"
  local archive
  for archive in "$DCASE_DIR/_zenodo"/*; do
    case "$archive" in
      *.zip) need_cmd unzip "Install unzip."; unzip -q -o "$archive" -d "$DCASE_DIR" ;;
      *.tar|*.tar.gz|*.tgz) tar -xf "$archive" -C "$DCASE_DIR" ;;
      *) cp -n "$archive" "$DCASE_DIR/" ;;
    esac
  done

  # datasets.py walks ./DCASEFoleySoundSynthesisDevSet; accept either that name
  # or the "(1)"-suffixed one the archive may unpack to.
  if [ ! -d "$dest" ] && [ -d "$DCASE_DIR/DCASEFoleySoundSynthesisDevSet(1)" ]; then
    ln -s "DCASEFoleySoundSynthesisDevSet(1)" "$dest"
  fi
  log "DCASE dev set installed under $DCASE_DIR/ (archives kept in _zenodo/)"
}

install_dcase_checkpoints() {
  mkdir -p "$DCASE_DIR/checkpoint/hifigan"
  local f
  for f in g_00935000 hifigan_config.json; do
    if [ -s "$DCASE_DIR/checkpoint/hifigan/$f" ]; then
      log "$f already present, skipping"
    else
      log "downloading HiFi-GAN $f from the upstream baseline repo"
      need_cmd curl "Install curl."
      curl -fL --retry 3 --retry-delay 5 "$DCASE_HIFIGAN_BASE/$f" \
        -o "$DCASE_DIR/checkpoint/hifigan/$f"
    fi
  done

  local entry name id
  for entry in "${DCASE_CKPTS[@]}"; do
    name="${entry%:*}"; id="${entry##*:}"
    if [ -s "$DCASE_DIR/$name" ]; then
      log "$(basename "$name") already present, skipping"
      continue
    fi
    gdrive_download "$id" "$DCASE_DIR/$name"
  done
  log "DCASE checkpoints installed"
}

install_pixelsnail_ckpt() {
  if [ -s "$PIXELSNAIL_DIR/$PIXELSNAIL_CKPT_NAME" ]; then
    log "PixelSNAIL checkpoint already present, skipping"
    return
  fi
  log "downloading standalone PixelSNAIL checkpoint, epoch 41 (~17 MB)"
  gdrive_download "$PIXELSNAIL_CKPT_ID" "$PIXELSNAIL_DIR/$PIXELSNAIL_CKPT_NAME"
}

verify() {
  local ok=1
  log "verifying repository layout"

  for f in vae_mel_16k_64bins.ckpt hifigan_16k_64bins.ckpt hifigan_16k_64bins.json; do
    if [ -f "$CKPT_DIR/$f" ]; then
      printf '  [ok]      %s\n' "$CKPT_DIR/$f"
    else
      printf '  [missing] %s  (run --checkpoints)\n' "$CKPT_DIR/$f"; ok=0
    fi
  done

  # Only the AudioLDM baseline config needs this one.
  if [ -f "$CKPT_DIR/clap_htsat_tiny.pt" ]; then
    printf '  [ok]      %s\n' "$CKPT_DIR/clap_htsat_tiny.pt"
  else
    printf '  [absent]  %s  (run --clap-htsat-tiny; only for audioldm_original.yaml)\n' \
      "$CKPT_DIR/clap_htsat_tiny.pt"
  fi

  if [ -d "$DATASET_DIR/audioset/zip_audios" ]; then
    printf '  [ok]      %s  (%s wav files)\n' "$DATASET_DIR/audioset/zip_audios" \
      "$(find "$DATASET_DIR/audioset/zip_audios" -name '*.wav' 2>/dev/null | wc -l)"
  else
    printf '  [missing] %s  (run --dataset)\n' "$DATASET_DIR/audioset/zip_audios"; ok=0
  fi

  if [ -f "$DATASET_DIR/metadata/dataset_root.json" ]; then
    printf '  [ok]      %s\n' "$DATASET_DIR/metadata/dataset_root.json"
  else
    printf '  [missing] %s  (tracked in git; re-clone the repo)\n' "$DATASET_DIR/metadata/dataset_root.json"; ok=0
  fi

  local trained="$LOG_DIR/audioldm_custom/checkpoints/$TRAINED_CKPT_NAME"
  if [ -f "$trained" ]; then
    printf '  [ok]      %s\n' "$trained"
  else
    printf '  [missing] %s  (run --trained-ckpt; only needed for inference)\n' "$trained"
  fi

  # Upstream's own structural check, when its one dependency is installed.
  if python3 -c "import tqdm" >/dev/null 2>&1; then
    python3 tests/validate_dataset_checkpoint.py || warn "upstream validation reported problems"
  else
    warn "tqdm is not installed; skipped tests/validate_dataset_checkpoint.py"
  fi

  if [ "$ok" = 1 ]; then
    log "layout looks good"
  else
    warn "some assets are still missing"
  fi
}

usage() {
  sed -n '10,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------

[ $# -gt 0 ] || usage 1

do_checkpoints=0 do_checkpoints_alt=0 do_dataset=0 do_trained=0 do_baseline=0
do_clap_tiny=0 do_clap_ae=0 do_finetune=0 do_verify=0
do_dcase_data=0 do_dcase_ckpt=0 do_ps_ckpt=0

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoints)       do_checkpoints=1 ;;
    --checkpoints-alt)   do_checkpoints_alt=1 ;;
    --dataset)           do_dataset=1 ;;
    --trained-ckpt)      do_trained=1 ;;
    --baseline-ckpt)     do_baseline=1 ;;
    --clap-htsat-tiny)   do_clap_tiny=1 ;;
    --clap-autoencoder)  do_clap_ae=1 ;;
    --finetune-ckpts)    do_finetune=1 ;;
    --dcase-dataset)     do_dcase_data=1 ;;
    --dcase-checkpoints) do_dcase_ckpt=1 ;;
    --pixelsnail-ckpt)   do_ps_ckpt=1 ;;
    --experiments)       do_dcase_data=1; do_dcase_ckpt=1; do_ps_ckpt=1 ;;
    --minimal)           do_checkpoints=1; do_trained=1 ;;
    --all)               do_checkpoints=1; do_dataset=1; do_trained=1
                         do_baseline=1; do_clap_tiny=1; do_clap_ae=1; do_finetune=1 ;;
    --verify)            do_verify=1 ;;
    -h|--help)           usage 0 ;;
    *)                   die "unknown option '$1' (try --help)" ;;
  esac
  shift
done

[ "$do_checkpoints"     = 1 ] && install_checkpoints
[ "$do_checkpoints_alt" = 1 ] && install_checkpoints_alt
[ "$do_dataset"         = 1 ] && install_dataset
[ "$do_trained"         = 1 ] && install_trained_ckpt
[ "$do_baseline"        = 1 ] && install_baseline_ckpt
[ "$do_clap_tiny"       = 1 ] && install_clap_htsat_tiny
[ "$do_clap_ae"         = 1 ] && install_clap_autoencoder
[ "$do_finetune"        = 1 ] && install_finetune_ckpts
[ "$do_dcase_data"      = 1 ] && install_dcase_dataset
[ "$do_dcase_ckpt"      = 1 ] && install_dcase_checkpoints
[ "$do_ps_ckpt"         = 1 ] && install_pixelsnail_ckpt
[ "$do_verify"          = 1 ] && verify

log "done"
