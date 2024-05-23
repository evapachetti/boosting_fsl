# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti (based on the code of Nikolas Adaloglou https://theaisummer.com/simclr/)
"""

## Pytorch lightning
import pytorch_lightning as pl

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Base feature extractor
from torchvision.models import vgg16, resnet18, resnet50, densenet121

model_mappings = {
        'VGG16': vgg16,
        'Resnet18': resnet18,
        'Resnet50': resnet50,
        'Densenet': densenet121
    }

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class SimCLR(pl.LightningModule):
    def __init__(self, pretrained_net, hidden_dim, learning_rate, temperature, weight_decay, max_epochs):
        super().__init__()
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        assert self.temperature > 0.0, "The temperature must be a positive float!"
        # Get model
        model = model_mappings.get(pretrained_net)(pretrained=True)
        if 'resnet' in pretrained_net.lower():
            self.num_ftrs = model.fc.in_features
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.num_ftrs, 4*hidden_dim, bias=False),
                            nn.BatchNorm1d(4*hidden_dim),
                            nn.ReLU(inplace=True), nn.Linear(4*hidden_dim, hidden_dim, bias=True))
            model.fc = Identity() # Get only feature extractor
            self.features = model
        elif 'vgg' in pretrained_net.lower():
            self.features = model.features
            self.num_ftrs = 512*4*4
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.num_ftrs, 4*hidden_dim, bias=False),
                            nn.BatchNorm1d(4*hidden_dim),
                            nn.ReLU(inplace=True), nn.Linear(4*hidden_dim, hidden_dim, bias=True))
        elif 'dense' in pretrained_net.lower():
            self.features = model.features
            self.num_ftrs = 1024*4*4
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.num_ftrs, 4*hidden_dim, bias=False),
                            nn.BatchNorm1d(4*hidden_dim),
                            nn.ReLU(inplace=True), nn.Linear(4*hidden_dim, hidden_dim, bias=True))
        

    def forward(self,x):
        x = self.features(x)
        out = self.classifier(x)
        return out

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.learning_rate / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        img1, img2 = batch
        imgs = (img1,img2)        
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.forward(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch):
        self.info_nce_loss(batch, mode="val")



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                