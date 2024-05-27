# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti (based on the code of Fei Long https://github.com/Fei-Long121/DeepBDC)
"""

import torch
import torchvision.transforms as transforms
from data.dataset import SimplePICAIDataset, SetPICAIDataset, SimpleBREAKHISDataset, SetBREAKHISDataset, EpisodicBatchSampler
from abc import abstractmethod


class TransformLoader:
    def __init__(self, image_size):        
        self.image_size = image_size

    def get_composed_transform(self):
            transform = transforms.Compose([transforms.ToTensor()])
            return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file):
        pass


class SimpleDataManager(DataManager):
    """
        Get a DataLoader for the specified dataset.

        Args:
            csv_file (str): Path to the CSV file containing dataset information.
            data (str): Name of the dataset ('picai' or 'breakhis').

        Returns:
            torch.utils.data.DataLoader: DataLoader object for the specified dataset.
    """

    def __init__(self, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size

    def get_data_loader(self, csv_file, data):  
        transform = self.trans_loader.get_composed_transform()
        if data=='picai':
            dataset = SimplePICAIDataset(csv_file, transform)
        elif data == 'breakhis':
            dataset = SimpleBREAKHISDataset(csv_file, transform)
        else:
            raise Exception("data must be picai or breakhis")
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    """
    DataManager class for generating episodic data loaders.

    Args:
        csv_path (str): Path to the CSV file containing dataset information.
        n_way (int): Number of classes per episode (way).
        n_support (int): Number of support examples per class.
        n_query (int): Number of query examples per class.
        n_episode (int): Number of episodes to sample.

    Attributes:
        n_way (int): Number of classes per episode (way).
        batch_size (int): Total batch size for data loading.
        n_episode (int): Number of episodes to sample.
        csv_path (str): Path to the CSV file containing dataset information.
    """
    
    def __init__(self, csv_path, n_way, n_support, n_query, n_episode):
        super(SetDataManager, self).__init__()
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.csv_path = csv_path

    def get_data_loader(self, data): 
        if data == 'picai':
            dataset = SetPICAIDataset(self.csv_path, self.batch_size)
        elif data == 'breakhis':
            dataset = SetBREAKHISDataset(self.csv_path, self.batch_size)
        else:
            raise Exception("Dataset not implemented")

        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=0, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader



