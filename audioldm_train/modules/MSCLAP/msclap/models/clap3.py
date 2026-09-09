import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from .audio import get_audio_encoder
from transformers import AutoTokenizer
from huggingface_hub.file_download import hf_hub_download

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

    def forward(self, x):
        if 'clip' in self.text_model:
            pooled_output = self.base(**x)[1] # get pooled output
            out = self.clip_text_projection(pooled_output)  # get CLS token output
        elif 'gpt' in self.text_model:
            batch_size = x['input_ids'].shape[0]
            hidden_states = self.base(**x)[0] # (batch_size=4, seq_len, 768)

            sequence_lengths = torch.ne(x['input_ids'], 0).sum(-1) - 1 # tensor([13, 14, 18, 17])
            out = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths] # [batch_size, 768] = [4, 768]
        else:
            out = self.base(**x)[0]
            out = out[:, 0, :]  # get CLS token output
        
        projected_vec = self.projection(out)

        return projected_vec


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
                embed_mode="text",
                max_random_mute_portion= 0.5,
                unconditional_prob=0.1,
                device="cuda",
                random_mute=False,
                version="CLAP_weights_2023.pth",
                model_repo = "microsoft/msclap"
                ):
        super().__init__()

        model_fp = hf_hub_download(model_repo, version)
        model_state_dict = torch.load(model_fp, map_location=torch.device(device))['model']
        self.unconditional_prob = unconditional_prob
        self.embed_mode = embed_mode
        
        self.audio_encoder = AudioEncoder(
            audioenc_name, out_emb, d_proj,
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
        
        self.caption_encoder = TextEncoder(
                d_proj, text_model, transformer_embed_dim
            )
        
        self.random_mute = random_mute
        self.tokenize = AutoTokenizer.from_pretrained(text_model)
        self.tokenize.pad_token = self.tokenize.eos_token  # Set the pad_token to the eos_token
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.unconditional_token = None
        self.eval()
        self.to(device = device)
        self.device = device
        self.max_random_mute_portion = max_random_mute_portion
        self.load_state_dict(model_state_dict, strict=False)
    
    def get_text_embedding(self, texts):
        embeds = self.caption_encoder(texts)
        return embeds.reshape(-1, 1024) 
    
    def get_audio_embedding(self, audio):
        embeds = self.audio_encoder(audio)
        return embeds.reshape(-1, 1024)         
        
    def tokenizer(self, texts):
        encoded_texts = self.tokenize(texts, padding=True, truncation=True, return_tensors="pt").to(device = self.device)
        return encoded_texts
    
    def forward(self, batch):
        if self.unconditional_token is None:
            self.build_unconditional_emb()

        # if(self.training_mode):
        #     assert self.model.training == True
        # else:
        #     assert self.model.training == False

        # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
        with torch.no_grad():
            if self.embed_mode == "audio":
                    embed, _ = self.get_audio_embedding(batch)
                    
            elif self.embed_mode == "text":
                    # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
                    text_data = self.tokenizer(batch)

                    if isinstance(batch, str) or (
                        isinstance(batch, list) and len(batch) == 1
                    ):
                        for key in text_data.keys():
                            text_data[key] = text_data[key].unsqueeze(0)
        
                    embed = self.get_text_embedding(text_data)
                    
            embed = embed.unsqueeze(1)
            for i in range(embed.size(0)):
                if self.make_decision(self.unconditional_prob):
                    embed[i] = self.unconditional_token
            # embed = torch.randn((batch.size(0), 1, 512)).type_as(batch)
            return embed.detach()
    

    def get_unconditional_condition(self, batchsize):
        self.unconditional_token = self.get_text_embedding(
            self.tokenizer([" ", " "])
        )[0:1]
        return torch.cat([self.unconditional_token.unsqueeze(0)] * batchsize, dim=0)

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def _random_mute(self, waveform):
        # waveform: [bs, t-steps]
        t_steps = waveform.size(-1)
        for i in range(waveform.size(0)):
            mute_size = int(
                self.random_uniform(0, end=int(t_steps * self.max_random_mute_portion))
            )
            mute_start = int(self.random_uniform(0, t_steps - mute_size))
            waveform[i, mute_start : mute_start + mute_size] = 0
        return waveform

    def cos_similarity(self, waveform, text):
        # waveform: [bs, t_steps]
        original_embed_mode = self.embed_mode
        with torch.no_grad():
            self.embed_mode = "audio"
            audio_emb = self(waveform.cuda())
            self.embed_mode = "text"
            text_emb = self(text)
            similarity = F.cosine_similarity(audio_emb, text_emb, dim=2)
        self.embed_mode = original_embed_mode
        return similarity.squeeze()

    def build_unconditional_emb(self):
        self.unconditional_token = self.get_text_embedding(
            self.tokenizer([" ", " "])
        )[0:1]