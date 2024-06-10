# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti (based on the code of Fei Long https://github.com/Fei-Long121/DeepBDC)
"""

import os
from datetime import datetime

import torch
import torch.optim
import matplotlib.pyplot as plt

from ..selfsupervised.simclr import SimCLR
from ..ipirm.model import IPIRM

from data.datamgr import SetDataManager
from methods.meta_deepbdc import MetaDeepBDC
from utils import *
from libauc.optimizers import PESG
from libauc.losses import AUCMLoss
import argparse

def train(params, base_loader, train_loader, val_loader, model, stop_epoch, decay_epochs, optimizer):

    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_auroc'] = []
    trlog['val_auroc'] = []
    trlog['max_auroc'] = 0.0
    trlog['max_auroc_epoch'] = 0
    trlog['val_ranks'] = []
    trlog['test_ranks'] = []

    for epoch in range(0, stop_epoch):
        print("Epoch: ",epoch)
        if epoch in decay_epochs:
            optimizer.update_regularizer(decay_factor=10) # decrease learning rate by 10x & update regularizer

        model.train()
        trainObj, auroc_train = model.train_loop(epoch, base_loader, optimizer)

        model.eval()
        valObj, auroc_val = model.test_loop(val_loader) # Evaluation on validation set
        _,auroc_train = model.test_loop(train_loader) # Evaluation on training set

        trlog['val_ranks'].append((epoch,auroc_val))

        if auroc_val > trlog['max_auroc']:
            print("best model! save...")
            trlog['max_auroc'] = auroc_val
            trlog['max_auroc_epoch'] = epoch
            outfile = os.path.join(params.output_path, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        trlog['train_loss'].append(trainObj)
        trlog['train_auroc'].append(auroc_train)
        trlog['val_loss'].append(valObj)
        trlog['val_auroc'].append(auroc_val)
        
        dateTimeObj = datetime.now() # current date and time
        date_time = dateTimeObj.strftime("%Y%m%d%H%M%S")  
        
        with open(os.path.join(params.output_path,"logger.txt"),'a') as fout:
            fout.write(f"[{date_time}] Epoch: {epoch}, Loss/train: {trainObj}, Loss/valid: {valObj}, AUROC/train: {auroc_train}, AUROC/val: {auroc_val}\n")

        if epoch - trlog['max_auroc_epoch'] > 25: # If the model isn't saved for 25 epochs, stop training
            print("Early stop at epoch: ",epoch)
            return model, trlog

    return model, trlog


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', default=128, type=int, choices=[128,224], help='input image size, 128 for picai')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate of the backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay of the backbone')
    parser.add_argument("--margin", default=1.0, type=float, help="Margin")
    parser.add_argument("--epoch_decay", default=0.003, type=float, help="Epoch decay (gamma)")
    parser.add_argument('--epoch', default=100, type=int, help='Stopping epoch')
    parser.add_argument("--feature_dim", default=128, type=int, help="Embedding size")

    parser.add_argument('--metatrain_dataset', default='picai', choices=['picai','breakhis'])
    parser.add_argument('--metatest_dataset', default='picai', choices=['picai','breakhis'])

    parser.add_argument('--csv_path_train', default='', type=str, help='trainset path')
    parser.add_argument('--csv_path_val', default='', type=str, help='valset path')

    parser.add_argument('--model', default='Resnet50', type=str, choices=['Resnet18','Resnet50','VGG16','Densenet'])
    parser.add_argument('--method', default='meta_deepbdc', choices='meta_deepbdc')

    parser.add_argument('--train_n_episode', default=600, type=int, help='number of episodes in meta train')
    parser.add_argument('--val_n_episode', default=300, type=int, help='number of episodes in meta val')
    parser.add_argument('--train_n_way', default=2, type=int, help='number of classes used for meta train')
    parser.add_argument('--val_n_way', default=2, type=int, help='number of classes used for meta val')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=16, type=int, help='number of unlabeled data in each class')

    parser.add_argument('--num_classes', default=4, type=int, help='total number of classes in pretrain')
    parser.add_argument('--pretrain_method', default=None, choices=['SimCLR','IPIRM'], help='pre-trained model .tar file path')
    parser.add_argument('--pretrain_path', default=None, help='pre-trained model .tar file path')
    parser.add_argument('--output_path', default=None, help='output finetuned model .tar file path')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimension of BDC dimensionality reduction layer')

    params = parser.parse_args()

    set_seed(params.seed)

    # TRAINING
    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(params.csv_path_train, params.image_size, n_query=params.n_query, n_episode=params.train_n_episode, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(data=params.metatrain_dataset)

    # INFERENCE
    test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(params.csv_path_val, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(data=params.metatest_dataset)

    train_datamgr = SetDataManager(params.csv_path_train, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode,**test_few_shot_params)
    train_loader = train_datamgr.get_data_loader(data=params.metatrain_dataset)

    if params.epoch == 100:
        decay_epochs = [20,50]
    elif params.epoch == 400:
        decay_epochs = [80,200]
    else:
        decay_epochs = [2000]


    
    #Define pre-trained model
    if params.pretrain_method == 'SimCLR':
        pretrained_model = SimCLR(
        pretrained_net = params.pretrained_net,
        hidden_dim=params.feature_dim)
        pretrained_model.load_state_dict(torch.load(os.path.join(params.pretrain_path ,"best_finetuned_simclr_model.pth")))        
   
    elif params.pretrain_method == 'IPIRM':
        pretrained_model = IPIRM(pretrained_net=params.pretrained_net, feature_dim=params.feature_dim)
        pretrained_model.load_state_dict(torch.load(os.path.join(params.pretrain_path, "best_finetuned_ipirm_model.pth")))
   
    else:
        pretrained_model = None

    #Define finetuned model
    model = MetaDeepBDC(params, model_dict[params.model](params,pre_trained_model=pretrained_model), **train_few_shot_params)

    model = model.cuda()
        
    loss_fn = AUCMLoss()
    optimizer = PESG(model.parameters(), 
            loss_fn=loss_fn,
            lr=params.learning_rate, 
            momentum=0.9,
            margin=params.margin, 
            epoch_decay=params.epoch_decay, 
            weight_decay=params.weight_decay)
    

    with open(os.path.join(params.output_path,"logger.txt"),'w') as fout:
        fout.write("results\n")
    model, trlog = train(params, base_loader, train_loader, val_loader, model, params.epoch, decay_epochs, optimizer)

    # Plot 
    plt.figure()
    plt.plot(range(len(trlog['train_auroc'])),trlog['train_auroc'],label="Training",color='navy')
    plt.plot(range(len(trlog['val_auroc'])),trlog['val_auroc'],label="Validation",color='magenta')
    plt.legend(loc="upper left")
    plt.title('AUROC')
    plt.show()
    plt.savefig(os.path.join(params.output_path,"AUROC.pdf"))

    plt.figure()
    plt.plot(range(len(trlog['train_loss'])),trlog['train_loss'],label="Training",color='green')
    plt.plot(range(len(trlog['val_loss'])),trlog['val_loss'],label="Validation",color='red')
    plt.legend(loc="upper left")
    plt.title('Loss')
    plt.show()
    plt.savefig(os.path.join(params.output_path,"LOSS.pdf"))

if __name__ == '__main__':
    main()    