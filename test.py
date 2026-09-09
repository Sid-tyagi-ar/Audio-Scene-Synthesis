from audioldm_train.utilities.model_util import instantiate_from_config
import torch
import yaml

config_yaml = yaml.load(open("audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_custom.yaml", "r"), Loader=yaml.FullLoader)

model = instantiate_from_config(config_yaml["model"])
ckpt = torch.load("log/latent_diffusion/2023_08_23_reproduce_audioldm/audioldm_custom/checkpoints/checkpoint-fad-133.00-global_step=499999.ckpt")["state_dict"]
model.load_state_dict(ckpt, strict=False)

model.get_log_dir = lambda :"."
output = model.generate_sample(["Children playing in a park"], name="tests")
print(output)