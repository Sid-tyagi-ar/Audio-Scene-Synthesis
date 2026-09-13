import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from .audio import get_audio_encoder
from transformers import AutoTokenizer

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name:str, d_in: int, d_out: int, sample_rate: int, window_size: int,
            hop_size: int, mel_bins: int, fmin: int, fmax: int, classes_num: int) -> None:
        super().__init__()

        audio_encoder = get_audio_encoder(audioenc_name)

        self.base = audio_encoder(
            sample_rate, window_size,
            hop_size, mel_bins, fmin, fmax,
            classes_num, d_in)

        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output

class TextEncoder(nn.Module):
    def __init__(self, d_out: int, text_model: str, transformer_embed_dim: int) -> None:
        super().__init__()
        self.text_model = text_model
        self.base = AutoModel.from_pretrained(text_model)

        if 'clip' in text_model:
            self.clip_text_projection = self.base.text_projection
            self.base = self.base.text_model
            if 'base' in text_model:
                transformer_embed_dim = 512
        
        self.projection = Projection(transformer_embed_dim, d_out)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set the pad_token to the eos_token

    def forward(self, texts):
        encoded_texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device = 'cuda')
        if 'clip' in self.text_model:
            pooled_output = self.base(**encoded_texts)[1] # get pooled output
            out = self.clip_text_projection(pooled_output)  # get CLS token output
        elif 'gpt' in self.text_model:
            batch_size = encoded_texts['input_ids'].shape[0]
            hidden_states = self.base(**encoded_texts)[0] # (batch_size=4, seq_len, 768)
            sequence_lengths = torch.ne(encoded_texts['input_ids'], 0).sum(-1) - 1 # tensor([13, 14, 18, 17])
            out = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths] # [batch_size, 768] = [4, 768]
        else:
            out = self.base(**encoded_texts)[0]
            out = out[:, 0, :]  # get CLS token output
        
        projected_vec = self.projection(out)
        return projected_vec

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Input shape: (1, 4, 1024), Output shape: (16, 4, 1024)
            nn.ReLU(),
            nn.LayerNorm([16, 4, 1024]),  # LayerNorm after first Conv2d layer
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Input shape: (16, 4, 1024), Output shape: (32, 4, 1024)
            nn.ReLU(),
            nn.LayerNorm([32, 4, 1024]),  # LayerNorm after second Conv2d layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Input shape: (32, 4, 1024), Output shape: (64, 4, 1024)
            nn.ReLU(),
            nn.LayerNorm([64, 4, 1024]),  # LayerNorm after third Conv2d layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Input shape: (64, 4, 1024), Output shape: (128, 4, 1024)
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output
            nn.Linear(128 * 4 * 1024, 512),

        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 128 * 4 * 8),  # Adjust size to match encoder output
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(128, 4, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(64 * 128, 1 * 4 * 1024),
            nn.Unflatten(dim=1, unflattened_size=(1, 4, 1024)),  # Adjust the linear layer to match the reshaped size
            nn.Sigmoid()  # Sigmoid activation for reconstruction
        )

class CLAP(nn.Module):
    def __init__(self,
                # audio
                audioenc_name: str,
                sample_rate: int, 
                window_size: int, 
                hop_size: int, 
                mel_bins: int, 
                fmin: int, 
                fmax: int, 
                classes_num: int, 
                out_emb: int,
                # text
                text_model: str,
                transformer_embed_dim: int,
                # common
                d_proj: int,
                device="cuda"
                ):
        super().__init__()

        
        self.audio_encoder = AudioEncoder(
            audioenc_name, out_emb, d_proj,
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)

        self.caption_encoder = TextEncoder(
            d_proj, text_model, transformer_embed_dim
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Input shape: (1, 4, 1024), Output shape: (16, 4, 1024)
            nn.ReLU(),
            nn.LayerNorm([16, 4, 1024]),  # LayerNorm after first Conv2d layer
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Input shape: (16, 4, 1024), Output shape: (32, 4, 1024)
            nn.ReLU(),
            nn.LayerNorm([32, 4, 1024]),  # LayerNorm after second Conv2d layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Input shape: (32, 4, 1024), Output shape: (64, 4, 1024)
            nn.ReLU(),
            nn.LayerNorm([64, 4, 1024]),  # LayerNorm after third Conv2d layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Input shape: (64, 4, 1024), Output shape: (128, 4, 1024)
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output
            nn.Linear(128 * 4 * 1024, 512),
            
        )
                
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.to(device=device)

    def forward_train(self, audio, text):
        audio_embed, _ = self.audio_encoder(audio)
        caption_embed = self.caption_encoder(text)
        return caption_embed, audio_embed, self.logit_scale.exp()
    
    def forward(self, text):
        caption_embed = self.caption_encoder(text).reshape(len(text)//4,1,4, 1024)
        print(caption_embed.size())
        caption_embed = self.encoder(caption_embed)
        print("DEBUG")
        return caption_embed
    