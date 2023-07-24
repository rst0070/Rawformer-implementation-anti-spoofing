import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
import torch.nn as nn
from trainer import Trainer
from data.train_set import ASVspoof2019LA as TrainSet
from data.test_set import ASVspoof2021LA_eval as TestSet
from data.preprocess import PreEmphasis
from models.rawformer import Rawformer_L, Rawformer_S, Rawformer_SE
from logger import Logger
import wandb
import random
import torch.utils.data as data
import datetime
import numpy as np
import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def getDataLoader(dataset, batch_size, num_workers):
    return data.DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           pin_memory=True, shuffle=False,
                           sampler=DistributedSampler(dataset),
                           num_workers=num_workers
                           )

def run(rank, world_size, port):
    
    # ------------------------- DDP setup ------------------------- #
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = rank
    
    # ------------------------- DDP setup ------------------------- #
    logger = Logger(device=device)
    
    # ------------------------- import configs ------------------------- #
    sys_config, exp_config = config.SysConfig(), config.ExpConfig()
    set_seed(exp_config.random_seed)
        
    # ------------------------- Data sets ------------------------- #
    train_loader = getDataLoader(dataset=TrainSet(), batch_size=exp_config.batch_size_train, num_workers=sys_config.num_workers)
    test_loader = getDataLoader(dataset=TestSet(), batch_size=exp_config.batch_size_test, num_workers=sys_config.num_workers)
    
    # ------------------------- set model ------------------------- #
    preprocessor = PreEmphasis(device=device).to(device)
    #model = DDP( Rawformer_L(device=device, sample_rate=exp_config.sample_rate, transformer_hidden=exp_config.transformer_hidden).to(device) )
    model = DDP( Rawformer_SE(device=device, sample_rate=exp_config.sample_rate, transformer_hidden=exp_config.transformer_hidden).to(device) )
    loss_fn = nn.BCELoss().to(device) #DDP is not needed when a module doesn't have any parameter that requires a gradient.
    
    # ------------------------- optimizer ------------------------- #
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=exp_config.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=exp_config.max_epoch,
    #     T_mult=1,
    #     eta_min=exp_config.lr_min
    # )
    
    # ------------------------- trainer ------------------------- #
    trainer = Trainer(preprocessor=preprocessor, model=model, loss_fn=loss_fn, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader, logger=logger, device=device)
    
    best_eer = 100.
    # miss_count = 0
    for epoch in range(1, exp_config.max_epoch + 1):
        
        logger.print(f'epoch: {epoch}')
        trainer.train()
        #scheduler.step()
        
        # -------------------- evaluation ----------------------- #
        if epoch % 5 == 1 or epoch == exp_config.max_epoch:
            
            eer = trainer.test()
            logger.print(f'EER: {eer}')
            logger.wandbLog({'EER_LA' : eer, 'epoch' : epoch})
            
            if eer < best_eer:
                # miss_count = 0
                best_eer = eer
                logger.wandbLog({'BestEER_LA' : eer, 'epoch' : epoch})
            # else:
            #     miss_count += 1
            #     if miss_count > 1:
            #         break
    
    destroy_process_group()

if __name__ == "__main__":
    import sys
    
    set_seed(config.ExpConfig().random_seed)
    
    torch.cuda.empty_cache()
    
    port = f'10{datetime.datetime.now().microsecond % 100}'
    world_size = torch.cuda.device_count()
    mp.spawn(run, 
             args=(world_size, port),
             nprocs=world_size
             )