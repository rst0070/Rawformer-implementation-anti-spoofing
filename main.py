import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import torch


def run(rank: int, world_size):
    
    # ------------------------- DDP setup ------------------------- #
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12335'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    print("start!")
    

if __name__ == "__main__":
    import sys
    world_size = torch.cuda.device_cout()
    mp.spawn(run, args=(world_size), nprocs=world_size)