
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:08:40 2021

@author: Germanese
"""
    

import torch
import pandas as pd
import numpy as np
from torch.utils import data
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

#%% PICAI

class PICAIDataset2DSSL(data.Dataset):
    
    
    def __init__(self, csv_path, dataset_name = "UNSUPERVISED", aug_folder='Originale',im_mod = 't2w', size_xy = 384, size_z = 12):
        
        """
        Parameters:
            
            - csv_file (string): percorso al file csv con le annotazioni
            - path_name (string): nome del dataset con cui allenare e anche folder da cui prendere i dati
            - aug_folder (string): folder contenente le immagini "aumentate" da aggiungere al dataset
            - size_xy (int): risoluzione xy delle immagini
            - size_z (int): numero di slice nel volume
        """
        
        self.info = pd.read_csv(csv_path)
        self.aug_folder = aug_folder
        self.parent_path = os.path.dirname(os.getcwd())
        self.dir_path = os.path.join(self.parent_path,"PI-CAI_dataset","cropped_images",dataset_name)
        self.im_mod = im_mod
        self.size_xy = size_xy
        self.size_z = size_z
       
    def __len__ (self):
            return len(self.info)
        
    def __getitem__(self, idx):
            if torch.is_tensor(idx): # se idx è un tensore
                idx = idx.tolist() # lo converto in una lista
            patient = str(self.info.iloc[idx]['patient_id'])
            study = str(self.info.iloc[idx]['study_id'])
            s = str(self.info.iloc[idx]['slice'])
            image_path = os.path.join(self.dir_path, patient+"_"+study+"_"+s+".png")
        
            image = Image.open(image_path)
            return image, patient
        

class PICAIToTensorDataset2DSSL(torch.utils.data.Subset):
    """
    Dato un dataset crea un altro dataset applicando una funzione data
    ad ognuno dei suoi valori. In questo caso converte i volumi e le label
    in tensori da fornire in ingresso ad una rete neurale.
   
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        
        #image = image.permute(3,2,0,1) #switch channels in the correct order
        
        if self.transform:
            aug_images = self.transform(image)
            #aug_images = [im.squeeze() for im in aug_images]
            #image = torch.from_numpy(np.array(image)).float()
            return aug_images
        
        image = torch.from_numpy(np.array(image)).float()

        return image

    def __len__(self):
        return len(self.dataset)


#%% BREAKHIS

class BREAKHISDataset2DSSL(data.Dataset):
    
    
    def __init__(self, csv_path):
        
        """
        Parameters:
            - magniude: Microscopic images magnitude level
            - cls_type: Whether classification refers to binary (benign vs. malignant) or multiclass
        """
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
              
            return image
        
        

class BREAKHISToTensorDataset2DSSL(torch.utils.data.Subset):
    """
    Dato un dataset crea un altro dataset applicando una funzione data
    ad ognuno dei suoi valori. In questo caso converte i volumi e le label
    in tensori da fornire in ingresso ad una rete neurale.
   
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[idx]#[0]
        
        #image = image.permute(3,2,0,1) #switch channels in the correct order
        
        if self.transform:
            aug_images = self.transform(image)
            #aug_images = [im.squeeze() for im in aug_images]
            #image = torch.from_numpy(np.array(image)).float()
            return aug_images
        
        image = torch.from_numpy(np.array(image)).float()

        return image

    def __len__(self):
        return len(self.dataset)