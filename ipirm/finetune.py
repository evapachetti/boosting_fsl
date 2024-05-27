# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti (based on the code of Wang Tan https://github.com/Wangt-CN/IP-IRM)
"""

import os
import copy
import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve, auc

from tqdm.auto import tqdm

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

import utils
from model import IPIRM
from finetuned_model import FinetunedModel


# train or test for one epoch
def train_val(model, data_loader, loss_fn, optimizer):

    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_num = 0.0, 0
    true_labels, predicted_labels, class_probabilities = [],[],[]

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_loader:
            data, target = data.cuda(non_blocking=True).float(), target.cuda(non_blocking=True)
            out = model(data)
            probabilities = torch.nn.Softmax(dim=1)(out)
            loss = loss_fn(probabilities,target)
           
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)

            if not is_train:
                probabilities = probabilities.detach().cpu()
                target = int(target.detach().cpu().item())

                true_labels.append(target)
                class_probabilities.append(np.array(probabilities))

        if not is_train:
            predicted_labels = [np.argmax(i) for i in class_probabilities]

        true_labels = np.array(true_labels)
        class_probabilities = np.array(class_probabilities).squeeze()
        predicted_labels = np.array(predicted_labels)
        
        return true_labels, predicted_labels, class_probabilities, total_loss / total_num, model
    

def evaluate(net, data_loader, loss_fn,dataset):
    net.eval()
    predicted_labels = []
    true_labels = []
    class_probabilities = []
    total_loss, total_num, data_bar = 0.0, 0, tqdm(data_loader)

    with (torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True).float(), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_fn(out, target)

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            
            probabilities = torch.nn.Softmax(dim=1)(out)

            # Collect labels and predictions
            true_labels.append(target.item())
            class_probabilities.append(probabilities.detach().cpu().numpy())

    predicted_labels = [np.argmax(i) for i in class_probabilities]

    true_labels = np.array(true_labels)
    class_probabilities = np.array(class_probabilities).squeeze()
    predicted_labels = np.array(predicted_labels)
    
    if dataset == 'picai':
        auroc = roc_auc_score(true_labels,class_probabilities,average='macro',multi_class='ovr')
    elif dataset == 'breakhis':
        auroc = roc_auc_score(true_labels,class_probabilities[:,1])
    
    return [auroc], true_labels, class_probabilities


        
def main():
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--pretrained_model_path', type=str, default='results/model_400.pth',
                        help='The pretrained model path')
    parser.add_argument('--output_path', type=str, default='',
                        help='Save fine-tuned model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay")
    parser.add_argument("--margin", default=1.0, type=float, help="Margin")
    parser.add_argument("--epoch_decay", default=0.003, type=float, help="Epoch decay (gamma)")
    parser.add_argument("--num_classes", default=1, type=int, help="Number of classes")
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument("--pretrained_net", choices = ['Resnet18','Resnet34','Resnet50','Resnet101','VGG11','VGG13','VGG16','VGG19','Densenet'],default=None, type=str, help="Pretrained network to fine-tune")
    parser.add_argument('--dataset', type=str, default='picai', choices=['picai','breakhis'], help='experiment dataset')
    parser.add_argument('--txt', action="store_true", default=False, help='save txt?')
    parser.add_argument('--name', type=str, default='None', help='exp name?')
    parser.add_argument("--parallel", default=False, action='store_true', help="If parallelize process")
    parser.add_argument("--train", default=False, action='store_true', help='Whether retrain a pretrained model')
    parser.add_argument("--evaluate", default=False, action='store_true', help='Whether evaluate a the retrained model')
    parser.add_argument('--csv_path_train', default='', type=str, help='trainset path')
    parser.add_argument('--csv_path_val', default='', type=str, help='valset path')
    parser.add_argument('--csv_path_test', default='', type=str, help='testset path')

    args = parser.parse_args()

    for arg in vars(args):
         print (arg, getattr(args, arg))

    if not os.path.exists('downstream/{}/{}'.format(args.dataset, args.name)):
        os.makedirs('downstream/{}/{}'.format(args.dataset, args.name))

    # seed
    utils.set_seed(42)

    batch_size, epochs = args.batch_size, args.num_epochs


    if args.dataset == 'picai':
        train_data = utils.PICAI(csv_path=args.csv_path_train, transform=utils.test_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                  drop_last=True)
        train_for_eval_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

        val_data = utils.PICAI(csv_path = args.csv_path_val, transform=utils.test_transform)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        test_data = utils.PICAI(csv_path = args.csv_path_test, transform=utils.test_transform)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    elif args.dataset == 'breakhis':
        train_data = utils.BREAKHIS(csv_path=args.csv_path_train, transform=utils.test_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                                  drop_last=True)
        train_for_eval_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

        val_data = utils.BREAKHIS(csv_path = args.csv_path_val, transform=utils.test_transform)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        test_data = utils.BREAKHIS(csv_path = args.csv_path_test, transform=utils.test_transform)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    else:
        raise Exception("Dataset not implemented")
   

    pretrained_model = IPIRM(pretrained_net=args.pretrained_net, feature_dim=args.feature_dim)
    pretrained_model.load_state_dict(torch.load(args.pretrained_model_path))
    model = FinetunedModel(num_classes=args.num_classes,pre_trained_model=model).cuda()

    loss_fn = AUCMLoss()
    optimizer = PESG(model.parameters(), 
            loss_fn=loss_fn,
            lr=args.learning_rate, 
            momentum=0.9,
            margin=args.margin, 
            epoch_decay=args.epoch_decay, 
            weight_decay=args.weight_decay)
        
    if args.train:

        losses_train = []
        losses_val = []
        roc_aucs_train = []
        roc_aucs_val = []

        if args.num_epochs == 100:
                decay_epochs = [20,50]
        elif args.num_epochs == 400:
                decay_epochs = [80,200]
        else:
            decay_epochs = [1e10]

        epochs_bar = tqdm(range(1, epochs + 1))
        best_AUROC = 0.0

        for epoch in epochs_bar:
            if epoch in decay_epochs:
                optimizer.update_regularizer(decay_factor=10) # decrease learning rate by 10x & update regularizer
            #Training
            _, _, _, train_loss, _ = train_val(model, train_loader, loss_fn, optimizer)

            #Evaluation on validation set
            true_labels_val, _, class_probabilities_val, val_loss, model = train_val(model, val_loader, loss_fn, None)
            roc_auc_val = roc_auc_score(
            true_labels_val,
            class_probabilities_val if args.dataset == 'picai' else class_probabilities_val[:, 1],
            average='macro' if args.dataset == 'picai' else None,
            multi_class='ovr' if args.dataset == 'picai' else None
            )

            roc_aucs_val.append(roc_auc_val)

            if roc_auc_val > best_AUROC:
                best_model = copy.deepcopy(model)
                best_AUROC = roc_auc_val

            #Evaluation on training set
            true_labels_train, _, class_probabilities_train, _,_ = train_val(model, train_for_eval_loader, loss_fn, None)
            roc_auc_train = roc_auc_score(
            true_labels_train,
            class_probabilities_train if args.dataset == 'picai' else class_probabilities_train[:, 1],
            average='macro' if args.dataset == 'picai' else None,
            multi_class='ovr' if args.dataset == 'picai' else None
            )

            roc_aucs_train.append(roc_auc_train)

            # Save losses
            losses_train.append(train_loss)
            losses_val.append(val_loss)


        torch.save(best_model.state_dict(),os.path.join(args.output_path,"best_finetuned_ipirm_model.pth"))
        print('Saved final model')

        plt.figure()
        plt.plot(range(len(losses_train)),losses_train,label="Training",color='green')
        plt.plot(range(len(losses_val)),losses_val,label="Validation",color='red')
        plt.legend(loc="upper left")
        plt.title('LOSS')
        plt.show()
        plt.savefig(os.path.join(args.output_path,"LOSS.pdf"))

        plt.figure()
        plt.plot(range(len(roc_aucs_train)),roc_aucs_train,label="Training",color='navy')
        plt.plot(range(len(roc_aucs_val)),roc_aucs_val,label="Validation",color='magenta')
        plt.legend(loc="upper left")
        plt.title('AUROCS')
        plt.show()
        plt.savefig(os.path.join(args.output_path,"AUROC.pdf"))

    if args.evaluate:

        model.load_state_dict(torch.load(os.path.join(args.output_path,"best_finetuned_ipirm_model.pth")))

        results_tr,_,_ = evaluate(model,train_for_eval_loader,loss_fn,args.dataset)
        results_v,_,_ = evaluate(model,val_loader,loss_fn,args.dataset)
        results_te,true_labels,class_probabilities = evaluate(model,test_loader,loss_fn,args.dataset)
        
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
        
        if args.dataset == 'picai':

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(true_labels, auroc_scores[:,i],pos_label=i)
                roc_auc[i] = auc(fpr[i], tpr[i])


            # Plot of a ROC curve for a specific ISUP class
            plt.figure()
            for i in range(n_classes): 
                plt.plot(fpr[i], tpr[i], label='ISUP '+str(i+2)+' vs. rest (area = %0.2f)' % roc_auc[i])
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(args.output_path,"TEST_ROC_CURVE.pdf"))
                plt.show()
        
        elif args.dataset == 'breakhis':
            
            fpr, tpr, _ = roc_curve(true_labels, auroc_scores[:,1])
            roc_auc = auc(fpr, tpr)

            # Plot of a ROC curve for a specific class
            plt.figure()
            plt.plot(fpr, tpr, label='AUROC = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(args.output_path,"TEST_ROC_CURVE.pdf"))
            plt.show()
        
if __name__ == '__main__':
    main()          