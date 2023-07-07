from typing import Any
import torch
import torchaudio
import torch.utils.data as data
import os
import random
from data.augmentation import WaveformAugmetation
import config

class ASVspoof2019LA(data.Dataset):
    """_summary_

    
    """
    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
        super(ASVspoof2019LA, self).__init__()
        
        self.duration = exp_config.train_duration_sec * exp_config.sample_rate
        
        path_label_train = sys_config.path_label_asv_spoof_2019_la_train
        path_label_dev = sys_config.path_label_asv_spoof_2019_la_dev
        
        self.data_list = []
        """This contains tuples like (file_path:str, attack_type, is_real:int)"""
        
        # ------------------- save train list ------------------- #
        for line in open(path_label_train).readlines():
            
            line = line.replace('\n', '').split(' ')
            file, attack_type, label = line[1], line[3], 0 if line[4] == 'bonafide' else 1
            file = os.path.join(sys_config.path_asv_spoof_2019_la_train, f'{file}.flac')
            
            self.data_list.append((file, attack_type, label))
        
        # ------------------- save dev list ------------------- #
        for line in open(path_label_dev).readlines():
            
            line = line.replace('\n', '').split(' ')
            file, attack_type, label = line[1], line[3], 0 if line[4] == 'bonafide' else 1
            file = os.path.join(sys_config.path_asv_spoof_2019_la_dev, f'{file}.flac')
            
            self.data_list.append((file, attack_type, label))

        
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index: Any) -> Any:
        
        utter, _, label = self.data_list[index]
        utter, _ = torchaudio.load(utter)
        
        utter = self.adjustDuration(utter)
        
        return utter, label
    
    def adjustDuration(self, x):
        if len(x.shape) == 2:
            x = x.squeeze()
            
        x_len = len(x)
        if x_len < self.duration:
            tmp = [x for i in range(0, (self.duration // x_len))]
            
            residue = self.duration % x_len
            if residue > 0: 
                tmp.append(x[0:residue])
            
            x = torch.cat(tmp, dim=0)
            
        x_len = len(x)
        start_seg = random.randint(0, x_len - self.duration)
        
        return x[start_seg : start_seg + self.duration] 
        