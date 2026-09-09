cd /content/drive/MyDrive/CS671-DL/AudioLDM/
pip install poetry
poetry install
source ~/.cache/pypoetry/virtualenvs/audioldm-train-eOy3ZhPZ-py3.10/bin/activate
# wandb login <YOUR_WANDB_API_KEY>
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/custom_audioldm.yaml