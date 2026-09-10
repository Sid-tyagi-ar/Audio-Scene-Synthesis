# import torch
# from torchvision import transforms
import pixelsnail
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

import os
import argparse
import pickle
import time
import json
import pprint
from functools import partial


import torch

# # Set max_split_size_mb (replace 256 with your desired value in MB)
# torch.cuda.set_device(0)  # Assuming you want to set it for device 0
torch.cuda.memory_summary(device='cuda', abbreviated=False)  # Optional: Check memory summary before setting
# # torch.cuda.set_memory_growth(True)  # Optional: Allow memory allocation to grow dynamically
# # torch.backends.cudnn.benchmark = False  # Optional: Disable cudnn heuristics for potentially better fragmentation

# max_split_size_mb = 256
# torch.cuda.set_max_split_size_mb(max_split_size_mb * 1024 * 1024)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA']='1'

from optim import Adam,RMSprop
# from tensorflow.keras.optimizers import Adam
# from torch.optim import Adam

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest='model', help='Select model architecture.', required=True)
# pixelcnn args
parser_a = subparsers.add_parser('pixelcnn')
parser_a.add_argument('--n_channels', default=128, type=int, help='Number of channels for gated residual convolutional layers.')
parser_a.add_argument('--n_out_conv_channels', default=1024, type=int, help='Number of channels for outer 1x1 convolutional layers.')
parser_a.add_argument('--n_res_layers', default=12, type=int, help='Number of Gated Residual Blocks.')
parser_a.add_argument('--kernel_size', default=5, type=int, help='Kernel size for the gated residual convolutional blocks.')
parser_a.add_argument('--norm_layer', default=True, type=eval, help='Add a normalization layer in every Gated Residual Blocks.')
# pixelcnn++ args
parser_b = subparsers.add_parser('pixelcnnpp')
parser_b.add_argument('--n_channels', default=128, type=int, help='Number of channels for residual blocks.')
parser_b.add_argument('--n_res_layers', default=5, type=int, help='Number of residual blocks at each stage.')
parser_b.add_argument('--n_logistic_mix', default=10, type=int, help='Number of of mixture components for logistics output.')
# pixelsnail args
parser_c = subparsers.add_parser('pixelsnail')
##  CHANGED THIS WITH 128 INSTEAD OF 256
parser_c.add_argument('--n_channels', default=256, type=int, help='Number of channels for residual blocks.')
##  CHANGED THIS WITH 4 INSTEAD OF 5 - Just like the paper
parser_c.add_argument('--n_res_layers', default=5, type=int, help='Number of residual blocks in each attention layer.')
##  CHANGED THIS WITH 10 INSTEAD OF 12
parser_c.add_argument('--attn_n_layers', default=12, type=int, help='Number of attention layers.')
parser_c.add_argument('--attn_nh', default=1, type=int, help='Number of attention heads.')
parser_c.add_argument('--attn_dq', default=16, type=int, help='Size of attention queries and keys.')
##  CHANGED THIS WITH 64 INSTEAD OF 128
parser_c.add_argument('--attn_dv', default=4, type=int, help='Size of attention values.')
parser_c.add_argument('--attn_drop_rate', default=0, type=float, help='Dropout rate on attention logits.')
parser_c.add_argument('--n_logistic_mix', default=10, type=int, help='Number of of mixture components for logistics output.')

# action
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, default=0, help='Which cuda device to use.')
parser.add_argument('--mini_data', action='store_true', help='Truncate dataset to mini_data number of examples.')
# data params
parser.add_argument('--dataset', choices='abstract')
parser.add_argument('--n_cond_classes', type=int, help='Number of classes for class conditional model.')
parser.add_argument('--n_bits', type=int, default=4, help='Number of bits of input data.')
parser.add_argument('--image_dims', type=int, nargs='+', default=(3,64,64), help='Dimensions of the input data.')
parser.add_argument('--data_path', default='/kaggle/input/abstract-37k/Abstract-jpg/', help='Location of datasets.')
# training param
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization.')
parser.add_argument('--polyak', type=float, default=0.9995, help='Polyak decay for parameter exponential moving average.')
parser.add_argument('--mix_precission', type=bool, default='False')
parser.add_argument('--batch_size', type=int, default=5, help='Training batch size.')
parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--step', type=int, default=20, help='Current step of training (number of minibatches processed).')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--log_interval', type=int, default=50, help='How often to show loss statistics and save samples.')
parser.add_argument('--eval_interval', type=int, default=1, help='How often to evaluate and save samples.')
# generation param
parser.add_argument('--n_samples', type=int, default=1, help='Number of samples to generate.')


# Model and data paths (modify these)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

# ... (import other necessary libraries)
model_path = r"C:\Users\Sukhvansh Jain\Desktop\IIT Study Material\Semester 4\CS-671-DL\PixelSNAIL code and model\results\pixelsnail\2024-03-17_12-28-52\checkpoint_35_model2.pt"

def generate(model, generate_fn, args, model_checkpoint_path):
  model.eval()
  
  # Load the model checkpoint

  
  if args.n_cond_classes:
    samples = []
    for h in range(args.n_cond_classes):
      h = torch.eye(args.n_cond_classes).cpu()[h, None]
      torch.cuda.empty_cache()
      samples += [generate_fn(model, args.n_samples, args.image_dims, args.device, h=h)]
    samples = torch.cat(samples)
  else:
    samples = generate_fn(model, args.n_samples, args.image_dims, args.device)
  return vutils.make_grid(samples.cpu(), normalize=True, scale_each=True, nrow=args.n_samples)


def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    args = parser.parse_args()
    args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else \
                        os.path.join('results', args.model, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    writer = SummaryWriter(log_dir = args.output_dir)

    # save config
    if not os.path.exists(os.path.join(args.output_dir, 'config.json')): save_json(args.__dict__, 'config', args)
    writer.add_text('config', str(args.__dict__))
    pprint.pprint(args.__dict__)

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')
    # args.device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    if args.model=='pixelsnail':
        import pixelsnail, pixelcnnpp
        model = pixelsnail.PixelSNAIL(args.image_dims, args.n_channels, args.n_res_layers, args.attn_n_layers, args.attn_nh, 
                args.attn_dq, args.attn_dv, args.attn_drop_rate, args.n_logistic_mix, args.n_cond_classes).to(args.device)
        loss_fn = pixelcnnpp.loss_fn
        generate_fn = pixelcnnpp.generate_fn
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995), polyak=args.polyak, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # if args.generate:
    # if args.step > 0: optimizer.swap_ema()
    samples = generate(model, generate_fn, args, model_path)
    writer.add_image('samples', samples, args.step)
    save_image(samples, os.path.join(args.output_dir, 'generation_sample_step_{}.png'.format(args.step)))
    # if args.step > 0: optimizer.swap_ema()

    