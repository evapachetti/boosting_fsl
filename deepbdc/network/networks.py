# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti (based on the code of Fei Long https://github.com/Fei-Long121/DeepBDC)
"""

from torchvision import models
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class Resnet18(nn.Module):
    """
    ResNet18 model customized for feature extraction in few-shot learning scenarios.

    Args:
        params (object): Hyperparameters and configuration settings.
        pre_trained_model (nn.Module, optional): A pre-trained model to use for feature extraction. Default is None.
        dataset (str, optional): The dataset type ('picai' by default).

    Attributes:
        n_shot (int): Number of support examples per class.
        n_query (int): Number of query examples per class.
        n_way (int): Number of classes per episode (way).
        image_size (int): Size of the input images.
        dataset (str): The dataset type.
        features (nn.Module): The feature extractor model.
    """

    def __init__(self, params, pre_trained_model=None, dataset='picai'):
        super(Resnet18, self).__init__()

        self.n_shot = params.n_shot
        self.n_query = params.n_query
        self.n_way = getattr(params, 'train_n_way', getattr(params, 'test_n_way'))
        self.image_size = params.image_size
        self.dataset = dataset

        if pre_trained_model is not None:
            if params.pretrain_method == 'SimCLR':
                self.features = pre_trained_model.features  
            elif params.pretrain_method == 'IPIRM':
                self.features = pre_trained_model.f 
            self.features.avgpool = Identity()
        else:
            model = models.resnet18(pretrained=True)
            model.avgpool = Identity()
            model.fc = Identity()  # Remove classification layer to get only feature extractor
            self.features = model

    def forward(self, x):
        x = self.features(x)
        if self.image_size == 128:
            out = torch.reshape(x, ((self.n_shot + self.n_query) * self.n_way, 512, 4, 4))
        elif self.image_size == 224:
            out = torch.reshape(x, ((self.n_shot + self.n_query) * self.n_way, 512, 7, 7))
        return out

    

class Resnet50(nn.Module):
    """
    ResNet50 model customized for feature extraction in few-shot learning scenarios.

    Args:
        params (object): Hyperparameters and configuration settings.
        pre_trained_model (nn.Module, optional): A pre-trained model to use for feature extraction. Default is None.
        dataset (str, optional): The dataset type ('picai' by default).

    Attributes:
        n_shot (int): Number of support examples per class.
        n_query (int): Number of query examples per class.
        n_way (int): Number of classes per episode (way).
        image_size (int): Size of the input images.
        dataset (str): The dataset type.
        features (nn.Module): The feature extractor model.
    """

    def __init__(self, params, pre_trained_model=None, dataset='picai'):
        super(Resnet18, self).__init__()

        self.n_shot = params.n_shot
        self.n_query = params.n_query
        self.n_way = getattr(params, 'train_n_way', getattr(params, 'test_n_way'))
        self.image_size = params.image_size
        self.dataset = dataset

        if pre_trained_model is not None:
            if params.pretrain_method == 'SimCLR':
                self.features = pre_trained_model.features  
            elif params.pretrain_method == 'IPIRM':
                self.features = pre_trained_model.f 
            self.features.avgpool = Identity()
        else:
            model = models.resnet50(pretrained=True)
            model.avgpool = Identity()
            model.fc = Identity()  # Remove classification layer to get only feature extractor
            self.features = model

    def forward(self, x):
        x = self.features(x)
        if self.image_size == 128:
            out = torch.reshape(x, ((self.n_shot + self.n_query) * self.n_way, 2048, 4, 4))
        elif self.image_size == 224:
            out = torch.reshape(x, ((self.n_shot + self.n_query) * self.n_way, 2048, 7, 7))
        return out



class VGG16(nn.Module):
    """
    VGG16 model customized for feature extraction in few-shot learning scenarios.

    Args:
        params (object): Hyperparameters and configuration settings.
        pre_trained_model (nn.Module, optional): A pre-trained model to use for feature extraction. Default is None.
        dataset (str, optional): The dataset type ('picai' by default).

    Attributes:
        n_shot (int): Number of support examples per class.
        n_query (int): Number of query examples per class.
        n_way (int): Number of classes per episode (way).
        image_size (int): Size of the input images.
        dataset (str): The dataset type.
        features (nn.Module): The feature extractor model.
    """

    def __init__(self, params,pre_trained_model=None):
        super(VGG16, self).__init__()

        self.n_shot = params.n_shot
        self.n_query = params.n_query
        self.n_way = getattr(params, 'train_n_way', getattr(params, 'test_n_way'))
        self.image_size = params.image_size 

        if pre_trained_model is not None:
            if params.pretrain_method == 'SimCLR':
                self.features = pre_trained_model.features 
            elif params.pretrain_method == 'IPIRM':
                self.features = pre_trained_model.f 
            self.features.avgpool = Identity()
        else:
            model = models.vgg16(pretrained=True)
            model.avgpool = Identity()
            model.classifier = Identity() # Cancel classification layer to get only feature extractor
            self.features = model

    def forward(self, x):
        x = self.features(x)
        if self.image_size == 128:
            out = torch.reshape(x,((self.n_shot+self.n_query)*self.n_way,512,4,4)) 
        elif self.image_size == 224:
            out = torch.reshape(x,((self.n_shot+self.n_query)*self.n_way,512,7,7))         
        return out
    

class Densenet(nn.Module):
    """
    Densenet model customized for feature extraction in few-shot learning scenarios.

    Args:
        params (object): Hyperparameters and configuration settings.
        pre_trained_model (nn.Module, optional): A pre-trained model to use for feature extraction. Default is None.
        dataset (str, optional): The dataset type ('picai' by default).

    Attributes:
        n_shot (int): Number of support examples per class.
        n_query (int): Number of query examples per class.
        n_way (int): Number of classes per episode (way).
        image_size (int): Size of the input images.
        dataset (str): The dataset type.
        features (nn.Module): The feature extractor model.
    """

    def __init__(self, params,pre_trained_model=None):
        super(Densenet, self).__init__()

        self.n_shot = params.n_shot
        self.n_query = params.n_query
        self.n_way = getattr(params, 'train_n_way', getattr(params, 'test_n_way'))
        self.image_size = params.image_size        
        
        if pre_trained_model is not None:
            if params.pretrain_method == 'SimCLR':
                self.features = pre_trained_model.features #SIMCLR
            elif params.pretrain_method == 'IPIRM':
                self.features = pre_trained_model.f #IPIRM
            self.features.avgpool = Identity()
        else:
            model = models.densenet121(pretrained=True)
            model.avgpool = Identity()
            model.classifier = Identity() # Cancel classification layer to get only feature extractor
            self.features = model

    def forward(self, x):
        x = self.features(x)
        if self.image_size == 128:
            out = torch.reshape(x,((self.n_shot+self.n_query)*self.n_way,1024,4,4)) 
        elif self.image_size == 224:
            out = torch.reshape(x,((self.n_shot+self.n_query)*self.n_way,1024,7,7))         
        return out