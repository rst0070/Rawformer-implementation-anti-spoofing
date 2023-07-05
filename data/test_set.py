from typing import Any
import torch
import torchaudio
import torch.utils.data as data
import os
import random
import config

class ASVspoof2021LA_eval(data.Dataset):
    
    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
        
        self.duration = exp_config.test_duration_sec * exp_config.sample_rate
        
        path_label = sys_config.path_label_asv_spoof_2021_la_eval
        path_eval = sys_config.path_asv_spoof_2021_la_eval
        
        self.data_list = []
        """This contains tuples like (file_path:str, attack_type, is_real:int)"""
        
        for line in open(path_label).readlines():
            line = line.replace('\n', '').split(' ')
            if line[7] != 'eval':
                continue
            
            file, attack_type, label = line[1], line[4], 0 if line[4] == 'bonafide' else 1
            file = os.path.join(path_eval, f'{file}.flac')
            
            self.data_list.append((file, attack_type, label))
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index: Any) -> Any:
        
        utter, _, label = self.data_list[index]
        utter, _ = torchaudio.load(utter)
        utter = self.adjustDuration(utter)
        
        return utter, label
    
    def adjustDuration(self, x):
        """_summary_
        use test data with specific duration from start of the audio 
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(x.shape) == 2:
            x = x.squeeze()
            
        x_len = len(x)
        if x_len < self.duration:
            tmp = [x for i in range(0, (self.duration // x_len))]
            
            residue = self.duration % x_len
            if residue > 0: 
                tmp.append(x[0:residue])
            
            x = torch.cat(tmp, dim=0)
        
        return x[0 : self.duration] 