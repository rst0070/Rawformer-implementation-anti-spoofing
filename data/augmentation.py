import torch
import torch.nn as nn

from torch_audiomentations import Compose, AddColoredNoise, HighPassFilter, LowPassFilter, Gain

class WaveformAugmetation(nn.Module):
    def __init__(self, aug_list=['ACN', 'HPF', 'LPF', 'GAN'], 
                 params={
                     'sr': 16000,
                     'ACN':{
                        'min_snr_in_db': 10, 'max_snr_in_db': 40, 'min_f_decay': -2.0, 'max_f_decay': 2.0, 'p': 0.5
                     },
                     'HPF':{
                         'min_cutoff_freq': 20.0, 'max_cutoff_freq': 2400.0, 'p': 0.5
                     },
                     'LPF':{
                         'min_cutoff_freq': 150.0, 'max_cutoff_freq': 7500.0, 'p': 0.5
                     },
                     'GAN':{
                         'min_gain_in_db': -15.0, 'max_gain_in_db': 5.0, 'p': 0.5
                     }
                 }
        ):
        # RawBoost option = (min_snr_in_db=10, max_snr_in_db=40)
        # torch_audiomentations option = (min_snr_in_db=3, max_snr_in_db=30)
        # torch_audiomentations option = (min_gain_in_db = -18.0, max_gain_in_db = 6.0)
        
        super(WaveformAugmetation, self).__init__()
        self.sr = params['sr']
        transforms = []
        if 'ACN' in aug_list:
            transforms.append(
                AddColoredNoise(
                    min_snr_in_db = params['ACN']['min_snr_in_db'],
                    max_snr_in_db = params['ACN']['max_snr_in_db'],
                    min_f_decay = params['ACN']['min_f_decay'],
                    max_f_decay = params['ACN']['max_f_decay'],
                    p = params['ACN']['p'],
                )
            )
        if 'HPF' in aug_list:
            transforms.append(
                HighPassFilter(
                    min_cutoff_freq = params['HPF']['min_cutoff_freq'],
                    max_cutoff_freq = params['HPF']['max_cutoff_freq'],
                    p = params['HPF']['p'],
                )
            )
        if 'LPF' in aug_list:
            transforms.append(
                LowPassFilter(
                    min_cutoff_freq = params['LPF']['min_cutoff_freq'],
                    max_cutoff_freq = params['LPF']['max_cutoff_freq'],
                    p = params['LPF']['p'],
                )
            )
        if 'GAN' in aug_list:
            transforms.append(
                Gain(
                    min_gain_in_db = params['GAN']['min_gain_in_db'],
                    max_gain_in_db = params['GAN']['max_gain_in_db'],
                    p = params['GAN']['p'],
                )
            )
            
        self.apply_augmentation = Compose(transforms)
        self.device = 'cpu'
        
    def forward(self, wav):
        # device sync
        if wav.device != self.device:
            self.device = wav.device
            self.apply_augmentation.to(wav.device)
        # data.shape: 1 dimention (WaveSize) 
        return self.apply_augmentation(wav.unsqueeze(1), self.sr).squeeze(1)
    