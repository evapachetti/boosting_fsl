# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti (based on the code of Wang Tan https://github.com/Wangt-CN/IP-IRM)
"""

import torch.nn as nn
from torchvision.models import resnet18, resnet50, vgg16, densenet121

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

# Pre-trained feature extractor + new classifier
class FinetunedModel(nn.Module):
    def __init__(self,num_classes,pre_trained_model):
        super(FinetunedModel, self).__init__()

        self.features = pre_trained_model.f # Get pre-trained feature-extractor
        self.classifier=nn.Linear(pre_trained_model.num_ftrs, num_classes) # Re-define the last layer to classify

    def forward(self,x):
        x = self.features(x)
        out = self.classifier(x)
        return out  