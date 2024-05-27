# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, densenet121
from torchvision.models.resnet import resnet18, resnet50

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
    

class IPIRM(nn.Module):
    def __init__(self, pretrained_net, feature_dim=128, channels=1):
        super(IPIRM, self).__init__()
        
        model = model_mappings.get(pretrained_net)(pretrained=True)
        if 'resnet' in pretrained_net.lower():
            self.num_ftrs = model.fc.in_features
            self.g = nn.Sequential(nn.Linear(self.num_ftrs, 4*feature_dim, bias=False), nn.BatchNorm1d(4*feature_dim),
                               nn.ReLU(inplace=True), nn.Linear(4*feature_dim, feature_dim, bias=True))
            model.fc = Identity() # Get only feature extractor
            self.f = model
        elif 'vgg' in pretrained_net.lower():
            self.f = model.features
            self.num_ftrs = 512*4*4
            self.g = nn.Sequential(nn.Linear(512*4*4, 4*feature_dim, bias=False), nn.BatchNorm1d(4*feature_dim),
                               nn.ReLU(inplace=True), nn.Linear(4*feature_dim, feature_dim, bias=True))
        elif 'dense' in pretrained_net.lower():
            self.f = model.features
            self.num_ftrs = 1024*4*4
            self.g = nn.Sequential(nn.Linear(1024*4*4, 4*feature_dim, bias=False), nn.BatchNorm1d(4*feature_dim),
                               nn.ReLU(inplace=True), nn.Linear(4*feature_dim, feature_dim, bias=True))
      
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

