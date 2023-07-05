import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.distributed as dist
from ddp_util import all_gather
import logger

class Trainer:
    
    def __init__(self, model:nn.Module, loss_fn:nn.Module, optimizer, train_loader, test_loader, logger:logger.Logger, device):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.logger = logger
        
        self.device = device
        
    def train(self):
        
        self.model.train()
        self.loss_fn.train()
        
        iter_count = 0
        loss_sum = 0
        num_item_train = len(self.train_loader)
        pbar = tqdm(self.train_loader)
        for x, label in pbar:
            
            self.optimizer.zero_grad()
            
            x, label = x.to(self.device), label.float().to(self.device)
            x = self.model(x)
            loss = self.loss_fn(x, label)
            
            loss.backward()
            self.optimizer.step()
        
            loss = loss.detach()
            iter_count += 1
            loss_sum += loss
            
            pbar.set_description(f'loss: {loss}')
            
            if num_item_train * 0.02 <= iter_count:
                self.logger.wandbLog({'Loss' : loss_sum / float(iter_count)})
                loss_sum = 0
                iter_count = 0
                
            
            
        
    def test(self):
        
        self.model.eval()
        
        scores = []
        labels = []
        
        pbar = tqdm(self.test_loader, 'evaluation')
        with torch.no_grad():
            for x, label in pbar:
            
                x, label = x.to(self.device), label.float().to(self.device)
            
                score = self.model(x)

                scores.append(score.cpu())
                labels.append(label.cpu())
        
            scores = torch.cat(scores, dim=0)
            labels = torch.cat(labels, dim=0)
                    
        scores = all_gather(scores)
        labels = all_gather(labels)
        eer = self.calculate_EER(scores, labels)
        
        
        return eer
            
    def calculate_EER(self, scores, labels):
        fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer * 100
