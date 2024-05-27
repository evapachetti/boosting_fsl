# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti (based on the code of Fei Long https://github.com/Fei-Long121/DeepBDC)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms


identity = lambda x: x

class SimplePICAIDataset(data.Dataset):

    """
    Custom Dataset for loading and preprocessing the PI-CAI dataset for 2D image classification.

    Attributes:
    -----------
    info : pandas.DataFrame
        DataFrame containing metadata for the dataset, loaded from the provided CSV file.
    parent_path : str
        The parent directory path of the current working directory.
    dir_path : str
        The directory path where the images are stored, constructed from the parent path 
        and a folder name.

    Methods:
    --------
    __init__(csv_path, folder_name="supervised"):
        Initializes the dataset by loading the metadata from the CSV file and setting 
        up the directory path for images.
    __len__():
        Returns the number of samples in the dataset.
    __getitem__(idx):
        Retrieves the image and label corresponding to the given index.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the dataset metadata.
    folder_name : str, optional
        Name of the folder containing the images (default is "supervised").
    """

    def __init__(self, csv_path, folder_name = "supervised"):
        self.info = pd.read_csv(csv_path)
        self.parent_path = os.path.dirname(os.getcwd())
        self.dir_path = os.path.join(self.parent_path,"dataset","picai",folder_name)
      
    def __len__ (self):
            return len(self.info)
        
    def __getitem__(self, idx):
            if torch.is_tensor(idx): 
                idx = idx.tolist()
            patient = str(self.info.iloc[idx]['patient_id'])
            study = str(self.info.iloc[idx]['study_id'])
            s = str(self.info.iloc[idx]['slice'])
            image_path = os.path.join(self.dir_path, f"{patient}_{study}_{s}.png")
            image = Image.open(image_path)
            label = int(self.info.iloc[idx]['label'])
    
            return image, label
        
    
class SetPICAIDataset:
    """
    Dataset class for PICAIDataset.

    Args:
        csv_path (str): Path to the CSV file containing dataset information.
        batch_size (int): Batch size for data loading.
        transform (callable): A function/transform to apply to the data.
        folder_name (str, optional): Name of the folder containing dataset images. Defaults to "supervised".

    Attributes:
        info (pd.DataFrame): DataFrame containing dataset information.
        parent_path (str): Parent directory path.
        dir_path (str): Directory path containing dataset images.
        data (list): List of image paths.
        label (list): List of image labels.
        transform (callable): A function/transform to apply to the data.
        cl_list (list): List of unique class labels.
        sub_meta (dict): Dictionary containing image paths for each class label.
        sub_dataloader (list): List of DataLoader objects for each class label.
    """

    def __init__(self, csv_path, batch_size, folder_name="supervised"):
        
        self.info = pd.read_csv(csv_path)
        self.parent_path = os.path.dirname(os.getcwd())
        self.dir_path = os.path.join(self.parent_path,"dataset","picai",folder_name)
        data = []
        label = []

        # Get all data and labels at first
        for _,row in self.info.iterrows():
            patient = str(row['patient_id'])
            study = str(row['study_id'])
            s = str(row['slice'])
            target = int(row['label'])
            img_path = os.path.join(self.dir_path, f"{patient}_{study}_{s}.png")
            data.append(img_path)
            label.append(target)

        self.data = data
        self.label = label
        self.cl_list = np.unique(self.label).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.data, self.label):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubPICAIDataset(self.sub_meta[cl], cl)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)
    

class SubPICAIDataset:
    """
    Dataset class for a subset of the PICAIDataset.

    Args:
        sub_meta (list): List of image paths for the subset.
        cl (int): Class label for the subset.
        transform (callable, optional): A function/transform to apply to the image (default: transforms.ToTensor()).
        target_transform (callable, optional): A function/transform to apply to the target (default: identity).

    Attributes:
        sub_meta (list): List of image paths for the subset.
        cl (int): Class label for the subset.
        transform (callable): A function/transform to apply to the image.
        target_transform (callable): A function/transform to apply to the target.
    """

    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        img_path = os.path.join(self.sub_meta[i])
        img = Image.open(img_path)
        img = self.transform(img)
        target = self.target_transform(self.cl)       
        return img, target

    def __len__(self):
        return len(self.sub_meta)
    


class SimpleBREAKHISDataset(data.Dataset):
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
        __init__(csv_path, cls_type = "binary", folder_name="supervised"):
            Initializes the dataset by loading the metadata from the CSV file and setting 
            up the directory path for images.
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
        folder_name : str, optional
            Name of the folder containing the images (default is "supervised").
    """
    
    def __init__(self, csv_path, cls_type = "binary", folder_name = "supervised"):
        
        self.cls_type = cls_type
        assert self.cls_type in ["binary","multiclass"]
        self.info = pd.read_csv(csv_path)
        self.parent_path = os.path.dirname(os.getcwd())
        self.dir_path = os.path.join(self.parent_path,"dataset","breakhis",folder_name)
       
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
        
    

class SetBREAKHISDataset:
    """
    Dataset class for the BREAKHIS dataset.

    Args:
        csv_path (str): Path to the CSV file containing dataset information.
        batch_size (int): Batch size for data loading.
        transform (callable): A function/transform to apply to the data.
        folder_name (str, optional): Name of the folder containing dataset images. Defaults to "supervised".

    Attributes:
        info (pd.DataFrame): DataFrame containing dataset information.
        parent_path (str): Parent directory path.
        dir_path (str): Directory path containing dataset images.
        data (list): List of image paths.
        label (list): List of image labels.
        transform (callable): A function/transform to apply to the data.
        cl_list (list): List of unique class labels.
        sub_meta (dict): Dictionary containing image paths for each class label.
        sub_dataloader (list): List of DataLoader objects for each class label.
    """

    def __init__(self, csv_path, batch_size, folder_name = "supervised"):
        
        self.info = pd.read_csv(csv_path)
        self.parent_path = os.path.dirname(os.getcwd())
        self.dir_path = os.path.join(self.parent_path,"dataset","breakhis",folder_name)

        data = []
        label = []

        # Get all data and labels at first
        for _,row in self.info.iterrows():
            image = str(row['image'])
            target = int(row['label'])
            img_path = os.path.join(self.dir_path, image+".png")
            data.append(img_path)
            label.append(target)

        self.data = data
        self.label = label
        self.cl_list = np.unique(self.label).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.data, self.label):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubBREAKHISDataset(self.sub_meta[cl], cl)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)
    

class SubBREAKHISDataset:
    """
    Dataset class for a subset of the BREAKHIS dataset.

    Args:
        sub_meta (list): List of image paths for the subset.
        cl (int): Class label for the subset.
        transform (callable, optional): A function/transform to apply to the image (default: transforms.ToTensor()).
        target_transform (callable, optional): A function/transform to apply to the target (default: identity).

    Attributes:
        sub_meta (list): List of image paths for the subset.
        cl (int): Class label for the subset.
        transform (callable): A function/transform to apply to the image.
        target_transform (callable): A function/transform to apply to the target.
    """

    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        img_path = os.path.join(self.sub_meta[i])
        img = Image.open(img_path)
        img = self.transform(img)
        target = self.target_transform(self.cl)       
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    """
    Sampler for generating episodic batches in few-shot learning.

    Args:
        n_classes (int): Total number of classes in the dataset.
        n_way (int): Number of classes per episode (way).
        n_episodes (int): Number of episodes to sample.
    """
     
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


