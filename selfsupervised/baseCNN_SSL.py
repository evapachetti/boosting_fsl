# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:07:19 2023

@author: evapa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    def __init__(self,hidden_dim):
        super(BaseCNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
         # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding='same')
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding='same')
        self.conv3 = nn.Conv2d(16,32, kernel_size=(3, 3), padding='same')

        # Max pooling layers
        self.max_pool = nn.MaxPool2d((2, 2))
        
        self.fc = nn.Linear(32*48*48, 4*hidden_dim)
    
        #Others 
        self.relu = nn.ReLU()
        self.batch1=nn.BatchNorm2d(8)
        self.batch2=nn.BatchNorm2d(16)
        self.batch3=nn.BatchNorm2d(32)

        self.drop=nn.Dropout(p=0.2)        
    

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.batch1(out)
        out = self.max_pool(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.batch2(out)
        out = self.max_pool(out)
                
        out = self.conv3(out)
        out = self.relu(out)
        out = self.batch3(out)
        out = self.max_pool(out)

        out = out.view(out.size(0), -1) # FLATTEN
        
        out = self.fc(out)
        
        return out


