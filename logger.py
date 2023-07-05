import wandb
import os
from tqdm import tqdm
import config

class Logger:
    def __init__(self, device, sys_config=config.SysConfig()):
        self.device = device
        self.wandb_disabled = sys_config.wandb_disabled
        
        if device == 0 and not self.wandb_disabled:
            os.system(f"wandb login {sys_config.wandb_key}")
            wandb.init(
                project = sys_config.wandb_project,
                entity  = sys_config.wandb_entity,
                name    = sys_config.wandb_name,
                notes   = sys_config.wandb_notes
            )
        
    def wandbLog(self, contents:dict):
        if self.device != 0 or self.wandb_disabled:
            return
        
        wandb.log(contents)
    
    def print(self, *args):
        if self.device != 0:
            return
        
        print(*args)
    
    
            