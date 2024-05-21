# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

#%% Import libraries
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import numpy as np
import copy
import argparse
import logging
logging.basicConfig(level=logging.INFO)

## Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

## Torchvision
from torchvision import models

#Loss
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

#Dataset
from create_dataset import PICAIDataset2D, ToTensorPICAIDataset2D, BREAKHISDataset2D, ToTensorBREAKHISDataset2D

#Metrics
from sklearn.metrics import roc_auc_score


# Setting the seed
# Ensure that all operations are deterministic on GPU (if used) for reproducibility


def set_seed(seed):
    """
    Set the random seed for reproducibility across various libraries and ensure
    deterministic behavior in PyTorch operations.
    
    Parameters:
    seed (int): The seed value to set.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for numpy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # If CUDA is available, set the seed for CUDA as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure reproducibility in PyTorch by disabling benchmark and enabling deterministic mode
    torch.backends.cudnn.benchmark = False  # This might affect performance
    torch.backends.cudnn.deterministic = True  # This might affect performance



#%% Functions
def evaluate_model(model, loader, device, loss_fn, binary):
    model.eval()
    losses, true_labels, class_probabilities = [],[],[]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)

            loss = loss_fn(outputs, labels.float())
            losses.append(loss.item())

            true_labels.extend(labels.cpu().numpy())
            class_probabilities.extend(probabilities.cpu().numpy())

    true_labels = np.array(true_labels)
    class_probabilities = np.array(class_probabilities).squeeze()

    if binary:
        roc_auc = roc_auc_score(true_labels, class_probabilities[:, 1])
    else:
        roc_auc = roc_auc_score(true_labels, class_probabilities, average='macro', multi_class='ovr')

    return np.mean(losses), roc_auc

def train(params, model, train_loader, train_for_eval_loader, val_loader, loss_fn, optimizer, device, num_epochs, decay_epochs):
    aurocs_train, aurocs_val, losses_train, losses_val = [],[],[],[]

    best_AUROC = 0.0
    best_epoch = 0
    model = model.to(device)

    for epoch in tqdm(range(num_epochs)):
        if epoch in decay_epochs:
            optimizer.update_regularizer(decay_factor=10)  # decrease learning rate by 10x & update regularizer

        model.train()
        losses_epoch_train = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            train_loss = loss_fn(outputs, labels.float())
            losses_epoch_train.append(train_loss.item())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        losses_train.append(np.mean(losses_epoch_train))

        # Evaluate on validation set
        val_loss, roc_auc_val = evaluate_model(model, val_loader, device, loss_fn, params.binary)
        losses_val.append(val_loss)
        aurocs_val.append(roc_auc_val)

        # Save model if validation AUROC improves
        if roc_auc_val > best_AUROC:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_AUROC = roc_auc_val

        if epoch - best_epoch > 25:
            print("Early stop at epoch:", epoch)
            return best_model, best_epoch, aurocs_train, aurocs_val, losses_train, losses_val

        # Evaluate on training set for monitoring
        train_loss, roc_auc_train = evaluate_model(model, train_for_eval_loader, device, loss_fn, params.binary)
        aurocs_train.append(roc_auc_train)

    logging.info('--- Finished training ---')

    return best_model, best_epoch, aurocs_train, aurocs_val, losses_train, losses_val


def main():
    parser = argparse.ArgumentParser(
        prog='Supervised training',
        description='Train a neural network to be used as backbone for further experiments'
    )
    parser.add_argument("--seed", default=42, type=int, help="Reproducibility seed")
    parser.add_argument("--dataset", default="picai", choices=["picai", "breakhis"], type=str, help="Dataset to train on")
    parser.add_argument("--num_epochs", default=100, choices=[100, 400], type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", default=30, type=int, help="Batch size for training")
    parser.add_argument("--num_classes", default=1, type=int, help="Number of classes")
    parser.add_argument("--learning_rate", default=1e-1, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay")
    parser.add_argument("--margin", default=1.0, type=float, help="Margin")
    parser.add_argument("--epoch_decay", default=0.003, type=float, help="Epoch decay (gamma)")
    parser.add_argument("--evaluate", default=False, action='store_true', help="Evaluate the model")
    parser.add_argument("--train", default=False, action='store_true', help="Train the model")
    parser.add_argument("--output_path", default='', type=str, help="Path to save model")
    parser.add_argument("--pretrained_net", choices=['Resnet18', 'Resnet50', 'VGG16', 'Densenet121'], default='Resnet18', type=str, help="Pretrained network to fine-tune")
    parser.add_argument("--csv_path_train", default='', type=str, help='Path to train set CSV')
    parser.add_argument("--csv_path_val", default='', type=str, help='Path to validation set CSV')
    parser.add_argument("--csv_path_test", default='', type=str, help='Path to test set CSV')
    parser.add_argument("--binary", default=False, action="store_true", help='Binary classification flag')

    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    
    set_seed(args.seed)
    
    #%% Import dataset

    if args.dataset=="picai":
         Dataset = PICAIDataset2D
         ToTensorDataset = ToTensorPICAIDataset2D
    elif args.dataset=='breakhis':
         Dataset = BREAKHISDataset2D
         ToTensorDataset = ToTensorBREAKHISDataset2D
    else: raise Exception("Dataset not supported")

    trainset = Dataset(args.csv_path_train)
    valset = Dataset(args.csv_path_val)
    testset = Dataset(args.csv_path_test)

    tensor_trainset = ToTensorDataset(trainset)
    tensor_valset = ToTensorDataset(valset)
    tensor_testset = ToTensorDataset(testset)

    device = torch.device("cuda")

    #Model
    if args.pretrained_net == 'Resnet18':
        model = models.resnet18(pretrained=True)
        if args.dataset == 'picai':
            model.conv1 = nn.Conv2d(1,64,kernel_size=7, stride=2, padding=3, bias=False) 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes) 
    elif args.pretrained_net == 'Resnet50':
        model = models.resnet50(pretrained=True)
        if args.dataset == 'picai':
            model.conv1 = nn.Conv2d(1,64,kernel_size=7, stride=2, padding=3, bias=False) 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
    elif args.pretrained_net == 'VGG16':
        model = models.vgg16(pretrained=True)
        if args.dataset == 'picai':
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        model.classifier[6] = nn.Linear(4096, args.num_classes)
    elif args.pretrained_net == 'Densenet121':
        model = models.densenet121(pretrained=True)
        if args.dataset == 'picai':
            model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        model.classifier = nn.Linear(1024, args.num_classes)
        

    #Loss function and optimizer
    loss_fn = AUCMLoss()
    optimizer = PESG(model, 
                 loss_fn=loss_fn,
                 lr=args.learning_rate, 
                 momentum=0.9,
                 margin=args.margin, 
                 epoch_decay=args.epoch_decay, 
                 weight_decay=args.weight_decay)
    
  
    #Data loaders
    train_loader = DataLoader(tensor_trainset, batch_size=args.batch_size, shuffle = True, num_workers=0, pin_memory=True,drop_last=False)
    train_for_eval_loader = DataLoader(tensor_trainset, batch_size=1, shuffle = False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(tensor_valset, batch_size=1, shuffle = False, num_workers=0, pin_memory=True) 
    test_loader = DataLoader(tensor_testset, batch_size=1, shuffle = False, num_workers=0, pin_memory=True) 
    
    # Where to save model and results
    save_path = os.path.join(args.dataset,args.output_path)

    if args.train:
        if args.num_epochs == 100:
            decay_epochs = [20,50]
        elif args.num_epochs == 400:
            decay_epochs = [80,200]
    
        best_model,_,aurocs_train, aurocs_val, losses_train, losses_val = train(args,model,train_loader, train_for_eval_loader, val_loader, loss_fn=loss_fn, optimizer=optimizer, device=device, num_epochs=args.num_epochs, decay_epochs=decay_epochs)
        torch.save(best_model.state_dict(), os.path.join(save_path,"best_model.pth"))
        
        # Plot 
        plt.figure()
        plt.plot(range(len(aurocs_train)),aurocs_train,label="Training",color='navy')
        plt.plot(range(len(aurocs_val)),aurocs_val,label="Validation",color='magenta')
        plt.legend(loc="upper left")
        plt.title('AUROC')
        plt.show()
        plt.savefig(os.path.join(save_path,"AUROC.pdf"))

        plt.figure()
        plt.plot(range(len(losses_train)),losses_train,label="Training",color='green')
        plt.plot(range(len(losses_val)),losses_val,label="Validation",color='red')
        plt.legend(loc="upper left")
        plt.title('Loss')
        plt.show()
        plt.savefig(os.path.join(save_path,"LOSS.pdf"))

        np.save(os.path.join(args.output_path,"auroc_val.npy"),np.array(aurocs_val))
        np.save(os.path.join(args.output_path,"auroc_train.npy"),np.array(aurocs_train))


    if args.evaluate:

            model.load_state_dict(torch.load(os.path.join(save_path,"best_model.pth")))
            
            _, auroc_tr = evaluate_model(args,model,train_for_eval_loader,device,dataset='train')
            _, auroc_v = evaluate_model(args,model,val_loader,device,dataset='val')   
            _, auroc_te = evaluate_model(args,model,test_loader,device,dataset='test')

            with open(os.path.join(save_path,'results.txt'), 'w') as f:
                f.write("*Training*\n")
                f.write("AUROC: "+str(auroc_tr)+"\n")
                f.write("\n")
                f.write("*Validation*\n")
                f.write("AUROC: "+str(auroc_v)+"\n")
                f.write("\n")
                f.write("*Test*\n")
                f.write("AUROC: "+str(auroc_te)+"\n")
                f.write("\n")


if __name__ == "__main__":
    main()