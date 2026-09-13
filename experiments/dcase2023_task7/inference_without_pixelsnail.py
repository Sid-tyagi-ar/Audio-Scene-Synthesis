from typing import List
from numpy import ndarray
from torch import Tensor
from abc import ABC, abstractmethod

import os
import argparse
import math
import time
import datetime
import torch
from tqdm import tqdm
import soundfile as sf

from vqvae import VQVAE
# from pixelsnail import PixelSNAIL
from HiFiGanWrapper import HiFiGanWrapper

import os
device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True} 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA']='1'
# hyper params

# from cVAE import CVAE
from cVAE_improved import CVAE

class SoundSynthesisModel(ABC):
    @abstractmethod
    def synthesize_sound(self, class_id: str, number_of_sounds: int) -> List[ndarray]:
        raise NotImplementedError


class DCASE2023FoleySoundSynthesis:
    def __init__(
        self, number_of_synthesized_sound_per_class: int = 100, batch_size: int = 16
    ) -> None:
        self.number_of_synthesized_sound_per_class: int = (
            number_of_synthesized_sound_per_class
        )
        self.batch_size: int = batch_size
        self.class_id_dict: dict = {
            0: 'dog_bark',
            1: 'footstep',
            2: 'gunshot',
            3: 'keyboard',
            4: 'moving_motor_vehicle',
            5: 'rain',
            6: 'sneeze_cough',
        }
        self.sr: int = 22050
        self.save_dir: str = "./synthesized"

    def synthesize(self, synthesis_model: SoundSynthesisModel) -> None:
        for sound_class_id in self.class_id_dict:
            sample_number: int = 1
            save_category_dir: str = (
                f'{self.save_dir}/{self.class_id_dict[sound_class_id]}'
            )
            os.makedirs(save_category_dir, exist_ok=True)
            for _ in tqdm(
                range(
                    math.ceil(
                        self.number_of_synthesized_sound_per_class / self.batch_size
                    )
                ),
                desc=f"Synthesizing {self.class_id_dict[sound_class_id]}",
            ):
                synthesized_sound_list: list = synthesis_model.synthesize_sound(
                    sound_class_id, self.batch_size
                )
                for synthesized_sound in synthesized_sound_list:
                    if sample_number <= self.number_of_synthesized_sound_per_class:
                        sf.write(
                            f"{save_category_dir}/{str(sample_number).zfill(4)}.wav",
                            synthesized_sound,
                            samplerate=self.sr,
                        )
                        sample_number += 1


# ================================================================================================================================================
class BaseLineModel(SoundSynthesisModel):
    def __init__(
        self, cvae_checkpoint: str, vqvae_snail_checkpoint: str
    ) -> None:
        super().__init__()

        device = torch.device("cuda")
        self.cvae = CVAE(1720, 200, 7).to(device)
        self.cvae.load_state_dict(torch.load(cvae_checkpoint, map_location='cpu'))  # Load cVAE
        self.cvae.cuda()
        self.cvae.eval()
        self.cvae.to(device)  # Ensure model is on the correct device

        self.vqvae = VQVAE()
        self.vqvae.load_state_dict(
            torch.load(vqvae_snail_checkpoint, map_location='cpu')
        )
        self.vqvae.cuda()
        self.vqvae.eval()

        self.hifi_gan = HiFiGanWrapper(
            './checkpoint/hifigan/g_00935000',
            './checkpoint/hifigan/hifigan_config.json',
        )

    # def one_hot(labels, class_size , tensor):
    #     targets = torch.zeros(labels.size(0), class_size)
    #     for i, label in enumerate(labels):
    #         targets[i, label] = torch.max(tensor)
    #     return targets.to(device)


    @torch.no_grad()
    def synthesize_sound(self, class_id: str, number_of_sounds: int) -> List[ndarray]:
        audio_list: List[ndarray] = list()

        feature_shape: list = [20, 86]
        vq_token: Tensor = torch.zeros(
            number_of_sounds, *feature_shape, dtype=torch.int64
        ).cuda()
        cache = dict()

        flattened_tensor = vq_token.view(1, -1).float()

        # print(flattened_tensor.shape)
        label = class_id
        
        targets = torch.zeros(7)
        # for i, label in enumerate(label):
        targets[label] = 10
        targets = targets.to(device)

        label = targets.unsqueeze(0)
        
        import csv

        labels_expanded = label.repeat(1, 245)
        with open('tensor_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels_expanded.view(-1).tolist())


        tensor2 = torch.tensor([[0,0,0,0,0]]).to(device)
        concatenated_tensor = torch.cat((labels_expanded, tensor2), dim=1)
        flattened_tensor = flattened_tensor + concatenated_tensor
        # print (label)
        # for i in tqdm(range(feature_shape[0]), desc="pixel_snail"):
        #     for j in range(feature_shape[1]):
        #         out, cache = self.pixel_snail(
        #             vq_token[:, : i + 1, :],
        #             label_condition=torch.full([number_of_sounds, 1], int(class_id))
        #             .long()
        #             .cuda(),
        #             cache=cache,
        #         )
        #         prob: Tensor = torch.softmax(out[:, :, i, j], 1)
        #         vq_token[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)

        recon_batch, mu, logvar = self.cvae(flattened_tensor, label)

        print (recon_batch)
        print("HELLLLLLLLLLLLLL")
        count = 0

        min_val = torch.min(recon_batch)
        # print (min_val)
        max_val = torch.max(recon_batch)
        # print (max_val)
        recon_batch = ((recon_batch - min_val) / (max_val - min_val))*511

        for i in tqdm(range(feature_shape[0]), desc="pixel_snail"):
            for j in range(feature_shape[1]):
                # # print("HEHE\n")
                # if(recon_batch[:, count] < 0):
                #   vq_token[:, i, j] = 0
                # elif(recon_batch[:, count] >= 511):
                #   vq_token[:, i, j] = 511
                # else:
                vq_token[:, i, j] = recon_batch[:, count]
                count+=1
        # split_data = recon_batch.split(86, dim=1)
        # print(type(split_data))

        # numpy_array_list = []
        # for tensor in split_data:
        #     cpu_tensor = tensor.cpu().numpy()
        #     numpy_array_list.append(cpu_tensor)
        # # pred_mel = self.vqvae.decode_code(vq_token).detach()
        # import numpy as np
        # from numpy import concatenate
        # reshaped_list = [np.expand_dims(arr, axis=0) for arr in numpy_array_list]  # Add a dimension

        # mel = np.concatenate(reshaped_list, axis=0)
        print(vq_token.shape)
        pred_mel = self.vqvae.decode_code(vq_token).detach()
        
        for j, mel in enumerate(pred_mel):
            audio_list.append(self.hifi_gan.generate_audio_by_hifi_gan(mel))
        return audio_list
        # print(type(mel))
        # audio_list.append(self.hifi_gan.generate_audio_by_hifi_gan(mel))
        # return audio_list


# ===============================================================================================================================================
if __name__ == '__main__':
    device = torch.device("cuda")

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cvae_checkpoint', type=str, default=r'checkpoint'
    )  # Add argument for cVAE checkpoint
    parser.add_argument(
        '--vqvae_checkpoint', type=str, default='./checkpoint/vqvae/vqvae_046.pt'
    )
    parser.add_argument(
        '--number_of_synthesized_sound_per_class', type=int, default=1
    )
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    dcase_2023_foley_sound_synthesis = DCASE2023FoleySoundSynthesis(
        args.number_of_synthesized_sound_per_class, args.batch_size
    )
    dcase_2023_foley_sound_synthesis.synthesize(
        synthesis_model=BaseLineModel(args.cvae_checkpoint, args.vqvae_checkpoint)
    )
    print(str(datetime.timedelta(seconds=time.time() - start)))