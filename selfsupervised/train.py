# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti
"""

#%% Import libraries

import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import numpy as np

## Pytorch lightning
import pytorch_lightning as pl

## PyTorch
import torch
import torch.utils.data as data
import torch.nn.functional as F

## Torchvision
from torchvision import transforms

#PI-CAI dataset
from create_dataset_ssl import PICAIDatasetSSL, ToTensorPICAIDatasetSSL, BREAKHISDatasetSSL, ToTensorBREAKHISDatasetSSL
from create_dataset import PICAIDataset, ToTensorPICAIDataset, BREAKHISDataset, ToTensorBREAKHISDataset

## SimCLR class
from simclr import SimCLR

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

import warnings

import argparse

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, auc, roc_curve

import copy
from finetuned_model import FinetunedModel
import logging


warnings.simplefilter("ignore")


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


class ContrastiveTransformations(object):
    """
    Apply a set of base transformations to an input image multiple times, producing several augmented views of the image.

    This class is designed to facilitate contrastive learning, where multiple views of the same image are used to learn
    representations that are invariant to certain transformations.

    Attributes:
    -----------
    base_transforms : callable
        A function or a composition of functions that defines the transformations to be applied to the input image.
    n_views : int, optional
        The number of augmented views to be generated from the input image (default is 2).

    Methods:
    --------
    __call__(x):
        Apply the base transformations to the input image `x` `n_views` times and return the list of augmented images.

    Parameters:
    -----------
    base_transforms : callable
        The transformations to apply to the input image.
    n_views : int, optional
        The number of different views to generate from each input image (default is 2).
    """

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


def pretrain(args, trainset, valset, model, device):
    """
    Pretrain a model using PyTorch Lightning.

    This function trains the specified model using the given training dataset and evaluates it on the validation dataset.
    If a pretrained model checkpoint exists, it loads the checkpoint and skips training. Otherwise, it trains the model
    for the specified number of epochs.

    Parameters:
    -----------
    args : argparse.Namespace
        Arguments containing various settings for the pretraining process.
    trainset : torch.utils.data.Dataset
        Training dataset.
    valset : torch.utils.data.Dataset
        Validation dataset.
    model : torch.nn.Module
        The model to be pretrained.
    device : torch.device
        The device on which to perform the training (e.g., 'cuda' or 'cpu').

    Returns:
    --------
    torch.nn.Module
        The pretrained model.

    """
    trainer = pl.Trainer(default_root_dir=args.pre_trained_model_path,
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=args.num_epochs,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top1'),
                                    pl.callbacks.LearningRateMonitor('epoch')])
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(args.pre_trained_model_path,'best_pre_trained_SSL_model.ckpt')    
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = model.load_from_checkpoint(pretrained_filename)
    else:
        print(f'No pretrained model found, training...')
        train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                       drop_last=True, pin_memory=True, num_workers=args.num_workers)
        val_loader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                     drop_last=False, pin_memory=True, num_workers=args.num_workers)
        pl.seed_everything(42) # To be reproducible
        trainer.fit(model, train_loader, val_loader)
    return model


def evaluate_model(model, loader, device, loss_fn, binary):
    """
    Evaluates a given model on a provided data loader using a specified loss function.

    Parameters:
    model (torch.nn.Module): The model to evaluate.
    loader (torch.utils.data.DataLoader): The data loader providing the evaluation dataset.
    device (torch.device): The device on which to perform the evaluation (e.g., 'cpu' or 'cuda').
    loss_fn (torch.nn.Module): The loss function to use for calculating losses.
    binary (bool): Indicates whether the task is binary classification (True) or multi-class classification (False).

    Returns:
    tuple: A tuple containing:
        - float: The average loss over the evaluation dataset.
        - float: The ROC AUC score for the evaluation dataset. For binary classification, this is computed for the positive class. For multi-class classification, the macro-average ROC AUC score is computed using a one-vs-rest approach.
    """
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
    """
    Trains a model and evaluates it periodically on a validation set, with early stopping based on validation performance.

    Parameters:
    params (argparse.Namespace): Parameters containing the a series of attributes.
    model (torch.nn.Module): The model to train.
    train_loader (torch.utils.data.DataLoader): Data loader for the training dataset.
    train_for_eval_loader (torch.utils.data.DataLoader): Data loader for the training dataset used for evaluation.
    val_loader (torch.utils.data.DataLoader): Data loader for the validation dataset.
    loss_fn (torch.nn.Module): The loss function to use for training.
    optimizer (torch.optim.Optimizer): The optimizer to use for training.
    device (torch.device): The device on which to perform training (e.g., 'cpu' or 'cuda').
    num_epochs (int): The number of epochs to train the model.
    decay_epochs (list of int): Epochs at which to decay the learning rate by a factor of 10.

    Returns:
    tuple: A tuple containing:
        - torch.nn.Module: The best model based on validation AUROC.
        - int: The epoch at which the best model was obtained.
        - list of float: The AUROC scores on the training dataset over the epochs.
        - list of float: The AUROC scores on the validation dataset over the epochs.
        - list of float: The average training losses over the epochs.
        - list of float: The average validation losses over the epochs.
    """

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
    parser=argparse.ArgumentParser()
    parser.add_argument("--pretrain_dataset", default="picai", choices=["picai","breakhis"], type=str, help="Dataset where to perform pre-training step")
    parser.add_argument("--finetune_dataset", default="breakhis", choices=["picai","breakhis"], type=str, help="Dataset where to perform fully supervised fine-tuning step")
    parser.add_argument("--image_dim", default=128, choices=[128,224],type=int, help="Image dimensions")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", default=30, type=int, help="Batch size for training")
    parser.add_argument("--feature_dim", default=128, type=int, help="Embedding size")
    parser.add_argument("--temperature", default=0.07, type=float, help="Temperature")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of workers")
    parser.add_argument("--margin", default=1.0, type=float, help="Margin")
    parser.add_argument("--epoch_decay", default=0.003, type=float, help="Epoch decay (gamma)")
    parser.add_argument("--num_classes", default=1, type=int, help="Number of output classes")
    parser.add_argument("--pretrained_net", choices = ['Resnet18','Resnet50','VGG16','Densenet'],default=None, type=str, help="Pretrained network to fine-tune")
    parser.add_argument("--ssl_pretrain", default=False, action='store_true', help='Whether to perform pre-training in a self-supervised way')
    parser.add_argument("--retrain", default=False, action='store_true', help='Whether to fine-tune a pre-trained model')
    parser.add_argument("--evaluate", default=False, action='store_true', help='Whether to evaluate a fine-tuned model')
    parser.add_argument("--get_features", default=False, action='store_true', help='Get features from pre-trained model')
    parser.add_argument('--output_path', type=str, default=os.path.join(os.getcwd(),"output"),
                        help='Folder where to save models')
    parser.add_argument('--pretrained_model_path', type=str, default=os.path.join(os.getcwd(),"output","pretrained_models"),
                        help='Folder where to get pretrained models')
    parser.add_argument('--csv_path_train', default=os.path.join(os.path.dirname(os.getcwd())), type=str, help='Path of csv file for training set')
    parser.add_argument('--csv_path_val', default=os.path.join(os.path.dirname(os.getcwd())), type=str, help='Path of csv file for validation set')
    parser.add_argument('--csv_path_test', default=os.path.join(os.path.dirname(os.getcwd())), type=str, help='Path of csv file for test set')

    args = parser.parse_args()

    for arg in vars(args):
         print (arg, getattr(args, arg))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")

    
    if args.ssl_pretrain:

        # Select the dataset and transformation based on the pretrain_dataset argument
        dataset_mapping = {
            "picai": (PICAIDatasetSSL, ToTensorPICAIDatasetSSL),
            "breakhis": (BREAKHISDatasetSSL, ToTensorBREAKHISDatasetSSL)
        }

        model = SimCLR(
        pretrained_net = args.pretrained_net,
        max_epochs=args.num_epochs,
        hidden_dim=args.feature_dim,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        weight_decay=args.weight_decay
    )
        #%% Define augmentations
        contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=128),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor()
                                         ])
        
        # Calculate learning rate from the original paper
        learning_rate = 0.3 * args.batch_size / 256

        # Construct the root path
        root_path = os.path.join(
            args.output_path,
            args.pretrain_dataset,
            f"{args.pretrained_net}_{args.num_epochs}_{args.batch_size}_{learning_rate}_{args.weight_decay}"
        )

        # Create the directory if it doesn't exist
        os.makedirs(root_path, exist_ok=True)

        Dataset, ToTensorDataset = dataset_mapping.get(args.pretrain_dataset, (PICAIDatasetSSL, ToTensorPICAIDatasetSSL))

        # Create datasets
        trainset = Dataset(args.csv_path_train)
        valset = Dataset(args.csv_path_val)

        # Apply transformations
        unlabeled_train = ToTensorDataset(trainset, transform=ContrastiveTransformations(contrast_transforms, n_views=2))
        unlabeled_val = ToTensorDataset(valset, transform=ContrastiveTransformations(contrast_transforms, n_views=2))

        # Pretrain the model
        simclr_model = pretrain(
            trainset=unlabeled_train,
            valset=unlabeled_val,
            model=model,
            device=device,
        )

        # Save the pretrained model
        torch.save(simclr_model.state_dict(), os.path.join(root_path, "best_pretrained_model.pth"))
        
    if args.retrain:

        # Select the dataset and transformation based on the pretrain_dataset argument
        dataset_mapping = {
            "picai": (PICAIDataset, ToTensorPICAIDataset),
            "breakhis": (BREAKHISDataset, ToTensorBREAKHISDataset)
        }
      
        csv_path_train = args.csv_path_train
        csv_path_val = args.csv_path_val
        csv_path_test = args.csv_path_test

        Dataset, ToTensorDataset = dataset_mapping.get(args.finetune_dataset, (PICAIDatasetSSL, ToTensorPICAIDatasetSSL))


        trainset = Dataset(csv_path_train)
        valset = Dataset(csv_path_val)
        testset = Dataset(csv_path_test)

        tensor_trainset = ToTensorDataset(trainset)
        tensor_valset = ToTensorDataset(valset)
        tensor_testset = ToTensorDataset(testset)

        train_loader = DataLoader(tensor_trainset, batch_size=args.batch_size, shuffle = True, num_workers=0, pin_memory=True,drop_last=False)
        train_loader_eval = DataLoader(tensor_trainset, batch_size=1, shuffle = True, num_workers=0, pin_memory=True,drop_last=False)
        val_loader = DataLoader(tensor_valset, batch_size=1, shuffle = False, num_workers=0, pin_memory=True) 
        test_loader = DataLoader(tensor_testset, batch_size=1, shuffle = False, num_workers=0, pin_memory=True) 

       
        pre_trained_model = SimCLR(
            pretrained_net = args.pretrained_net,
            max_epochs=args.num_epochs,
            hidden_dim=args.feature_dim,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            weight_decay=args.weight_decay
        )

        if args.num_epochs == 100:
            decay_epochs = [20,50]
        elif args.num_epochs == 400:
            decay_epochs = [80,200]

        pre_trained_model.load_state_dict(torch.load(os.path.join(args.pretrained_model_path ,"best_simclr_model.pth")))
        finetuned_model = FinetunedModel(num_classes=args.num_classes,pre_trained_model = pre_trained_model)
        
        loss_fn = AUCMLoss()
        optimizer = PESG(finetuned_model.parameters(), 
                loss_fn=loss_fn,
                lr=args.learning_rate, 
                momentum=0.9,
                margin=args.margin, 
                epoch_decay=args.epoch_decay, 
                weight_decay=args.weight_decay)

        logging.info("\n\n--- Start re-training of pre-trained CNN---\n\n")

        final_model, _, aurocs_train,aurocs_val, losses_train, losses_val = train(finetuned_model, train_loader, train_loader_eval, val_loader, loss_fn, optimizer,device,args.num_epochs,decay_epochs,args.finetune_dataset)
        final_model_path = os.path.join(args.output_path,"best_finetuned_simclr_model.pth")
        
        torch.save(final_model.state_dict(),final_model_path)
        
        # Plot 
        plt.figure()
        plt.plot(range(len(aurocs_train)),aurocs_train,label="Training",color='navy')
        plt.plot(range(len(aurocs_val)),aurocs_val,label="Validation",color='magenta')
        plt.legend(loc="upper left")
        plt.title('AUROC')
        plt.show()
        plt.savefig(os.path.join(args.output_path,"AUROC.pdf"))

        plt.figure()
        plt.plot(range(len(losses_train)),losses_train,label="Training",color='green')
        plt.plot(range(len(losses_val)),losses_val,label="Validation",color='red')
        plt.legend(loc="upper left")
        plt.title('Loss')
        plt.show()
        plt.savefig(os.path.join(args.output_path,"LOSS.pdf"))

        if args.evaluate:

            logging.info("\n--- Evaluating final model ---\n")

            final_model_path = os.path.join(args.output_path,"best_finetuned_simclr_model.pth")
            final_model = FinetunedModel(num_classes=args.num_classes,pre_trained_model = pre_trained_model)
            final_model.load_state_dict(torch.load(final_model_path))
            
            results_tr,_,_ = evaluate_model(final_model,train_loader_eval,device,args.finetune_dataset)
            results_v,_,_ = evaluate_model(final_model,val_loader,device,args.finetune_dataset)
            results_te,true_labels,class_probabilities = evaluate_model(final_model,test_loader,device,args.finetune_dataset)

            metrics = ["AUROC"]

            with open(os.path.join(args.output_path,'results.txt'), 'w') as f:
                f.write("*Training*\n")
                for i,metric in enumerate(metrics):
                    f.write(metric+": "+str(results_tr[i])+"\n")
                    f.write("\n")
                f.write("*Validation*\n")
                for i,metric in enumerate(metrics):
                    f.write(metric+": "+str(results_v[i])+"\n")
                    f.write("\n")
                f.write("*Test*\n")
                for i,metric in enumerate(metrics):
                    f.write(metric+": "+str(results_te[i])+"\n")
                    f.write("\n")
            
            # Compute ROC curve and ROC area for each class for test set
            n_classes = args.num_classes
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            auroc_scores = class_probabilities

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(true_labels, auroc_scores[:,i],pos_label=i)
                roc_auc[i] = auc(fpr[i], tpr[i])


            # Plot of a ROC curve for a specific class
            plt.figure()
            for i in range(n_classes): 
                plt.plot(fpr[i], tpr[i], label='ISUP '+str(i+2)+' vs. rest (area = %0.2f)' % roc_auc[i])
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(args.output_path,"TEST_ROC_CURVE.pdf"))
                plt.show()


if __name__ == '__main__':
    main()

