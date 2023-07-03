import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import wandb

class Trainer:
    
    def __init__(self, model:nn.Module, loss_fn:nn.Module, optimizer, train_loader, test_loader, device):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.device = device
        
    def train(self):
        
        self.model.train()
        self.loss_fn.train()
        
        iter_count = 0
        loss_sum = 0
        pbar = tqdm(self.train_loader)
        for x, label in pbar:
            
            self.optimizer.zero_grad()
            
            x, label = x.to(self.device), label.to(self.device)
            loss = self.loss_fn(x, label)
            
            loss.backward()
            self.optimizer.step()
        
            loss = loss.detach()
            count += 1
            loss_sum += loss
            
            if count == 50:
                pbar.set_description(f'loss: {loss}')
                wandb.log({'loss' : loss})
        
    def test(self):
        
        self.model.eval()
        
        scores = []
        labels = []
        
        pbar = tqdm(self.test_loader, 'evaluation')
        for x, label in pbar:
            
            score = self.model(x)
            # score의 
            scores.append(score)
            labels.append(label)
        
        scores = torch.cat(scores, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        
        eer = self.calculate_EER(scores, labels)
        return eer
            
    def calculate_EER(self, scores, labels):
        fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer * 100