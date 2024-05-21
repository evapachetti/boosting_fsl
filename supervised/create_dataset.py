# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

import torch
import pandas as pd
import numpy as np
from torch.utils import data
from PIL import Image
import os

#%% BreakHis dataset
class BREAKHISDataset2D(data.Dataset):
    """
        Custom Dataset for loading and preprocessing the BREAKHIS dataset for 2D image classification.

        Attributes:
        ----------
        cls_type : str
            The type of classification task. Should be either 'binary' or 'multiclass'.
        info : pandas.DataFrame
            DataFrame containing dataset information loaded from a CSV file.
        parent_path : str
            The parent directory path of the current working directory.
        dir_path : str
            The directory path where the image files are stored.

        Methods:
        -------
        __len__():
            Returns the total number of samples in the dataset.
        
        __getitem__(idx):
            Retrieves the sample (image and label) at the specified index.
        
        Parameters:
        ----------
        csv_path : str
            Path to the CSV file containing dataset information.
        cls_type : str, optional
            The type of classification task. Default is "binary". Must be either "binary" or "multiclass".
    """
    
    def __init__(self, csv_path, cls_type = "binary"):
        
        self.cls_type = cls_type
        assert self.cls_type in ["binary","multiclass"]
        self.info = pd.read_csv(csv_path)
        self.parent_path = os.path.dirname(os.getcwd())
        self.dir_path = os.path.join(self.parent_path,"BreakHis_dataset","dataset")
       
    def __len__ (self):
            return len(self.info)
        
    def __getitem__(self, idx):
            if torch.is_tensor(idx): # se idx è un tensore
                idx = idx.tolist() # lo converto in una lista
            image = str(self.info.iloc[idx]['image'])
            image_path = os.path.join(self.dir_path, image+".png")
        
            image = Image.open(image_path)
            
            if self.cls_type == "binary":
                label = int(self.info.iloc[idx]['binary_target'])
            else:
                label = int(self.info.iloc[idx]['multi_target'])

            return image, label
        

class ToTensorBREAKHISDataset2D(torch.utils.data.Subset):

    """
    Dataset class to convert BREAKHIS dataset samples to PyTorch tensors.

    This class inherits from torch.utils.data.Subset.

    Attributes:
    ----------
    dataset : torch.utils.data.Dataset
        The original dataset containing image samples and their labels.
    transform : callable, optional
        A function/transform to be applied to the samples. Default is None.

    Methods:
    -------
    __getitem__(idx):
        Retrieve the sample (image and label) at the specified index and apply the transformation.
    
    __len__():
        Return the total number of samples in the dataset after transformation.

    Parameters:
    ----------
    dataset : torch.utils.data.Dataset
        The original dataset containing image samples and their labels.
    transform : callable, optional
        A function/transform to be applied to the samples. Default is None.
    """
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
    
        image = torch.from_numpy(np.array(image)).float()
        image = image.permute(2,0,1)
        label = torch.tensor(label)
        return image,label

    def __len__(self):
        return len(self.dataset)

#%% PICAI Dataset

class PICAIDataset2D(data.Dataset):

    def __init__(self, csv_path, folder_name = "supervised"):
        self.info = pd.read_csv(csv_path)
        self.parent_path = os.path.dirname(os.getcwd())
        self.dir_path = os.path.join(self.parent_path,"picai",folder_name)
      
    def __len__ (self):
            return len(self.info)
        
    def __getitem__(self, idx):
            if torch.is_tensor(idx): 
                idx = idx.tolist()
            patient = str(self.info.iloc[idx]['patient_id'])
            study = str(self.info.iloc[idx]['study_id'])
            s = str(self.info.iloc[idx]['slice'])
            image_path = os.path.join(self.dir_path, patient+"_"+study+"_"+s+".png")
            image = Image.open(image_path)
            label = int(self.info.iloc[idx]['label'])
    
            return image, label
        

class ToTensorPICAIDataset2D(torch.utils.data.Subset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
    
        image = torch.from_numpy(np.array(image)).float().unsqueeze(dim=0)
        label = torch.tensor(label)

        return image,label

    def __len__(self):
        return len(self.dataset)
    
    
    
