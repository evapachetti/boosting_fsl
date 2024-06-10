# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti
"""

import os
import argparse

import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from matplotlib import pyplot as plt

from data.datamgr import SetDataManager
from methods.protonet import ProtoNet
from methods.meta_deepbdc import MetaDeepBDC
from utils import *
from torch.optim import *
from ..selfsupervised.simclr import SimCLR
from ..ipirm.model import IPIRM


def evaluate(data_loader, model, params):
    auroc_all_task = []
    results = []

    for _ in range(params.test_task_nums):
        auroc_all = []
        with torch.no_grad():
            for x, _ in tqdm.tqdm(data_loader):
                model.n_query = params.n_query
                scores = model.set_forward(x, False)

                y = np.repeat(range(params.test_n_way), params.n_query)
                softmax_scores = torch.nn.Softmax(dim=1)(scores)
                auroc_softmax_scores = softmax_scores.detach().cpu().numpy()

                auroc = roc_auc_score(
                    y,
                    auroc_softmax_scores if params.metatestdataset == 'picai' else auroc_softmax_scores[:, 1],
                    average='macro' if params.dataset == 'picai' else None,
                    multi_class='ovr' if params.dataset == 'picai' else None
                )

                auroc_all.append(auroc)

        auroc_all_task.append(auroc_all)

    results.append((np.mean(auroc_all_task), np.std(auroc_all_task)))

    return results, y, auroc_softmax_scores

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', default=128, type=int, choices=[128,224], help='input image size, 128 for picai')

    parser.add_argument('--metatrain_dataset', default='picai', choices=['picai','breakhis'])
    parser.add_argument('--metatest_dataset', default='picai', choices=['picai','breakhis'])

    parser.add_argument('--model', default='Resnet18', choices=['Resnet18', 'Resnet50','Resnet18CA', 'Resnet50CA','Resnet18Ablation','VGG16','Densenet'])
    parser.add_argument('--method', default='stl_deepbdc', choices=['meta_deepbdc', 'stl_deepbdc', 'protonet', 'good_embed'])
    parser.add_argument("--feature_dim", default=128, type=int, help="Embedding size")

    parser.add_argument('--csv_path_test', default='', type=str, help='testset path')

    parser.add_argument('--test_n_way', default=5, type=int, help='number of classes used for testing (validation)')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

    parser.add_argument('--test_n_episode', default=2000, type=int, help='number of episodes in test')
    parser.add_argument('--pretrain_method', default='Imagenet', choices=['Imagenet','SimCLR','IPIRM'], help='pre-trained model .tar file path')
    parser.add_argument('--pretrain_path', default='', help='pre-trained model .tar file path')
    parser.add_argument('--model_path', default='', help='meta-trained or pre-trained model .tar file path')
    parser.add_argument('--output_path', default='', help='results output path')
    parser.add_argument('--test_task_nums', default=5, type=int, help='test numbers')

    parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimensions of BDC dimensionality reduction layer')

    params = parser.parse_args()

    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    test_datamgr = SetDataManager(params.csv_path_test, params.image_size, n_query=params.n_query, n_episode=params.test_n_episode, **test_few_shot_params)
    test_loader = test_datamgr.get_data_loader(data=params.metatest_dataset)

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
    model = MetaDeepBDC(params, model_dict[params.model](params,pre_trained_model=pretrained_model), **test_few_shot_params)
        
    # model save path
    model = model.cuda()
    model.eval()

    model_file = os.path.join(params.model_path)
    model = load_model(model, model_file)

    metrics = ["AUROC"]

    results_test,y,auroc_softmax_scores = evaluate(test_loader,model,params)

    with open(os.path.join(params.output_path,'results.txt'), 'w') as f:
        for i,metric in enumerate(metrics):
            f.write(metric+": "+str(results_test[i][0])+" ("+str(results_test[i][1])+")")
            f.write("\n")


    if params.metatrain_dataset == 'breakhis':
        fpr, tpr, _ = roc_curve(y, auroc_softmax_scores[:,1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='Benign vs. Malignant (area = %0.2f)' % roc_auc,color="purple")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(params.output_path,"TEST_ROC_CURVE_final.pdf"))
        plt.show()


    elif params.metatrain_dataset == 'picai':
        n_classes = params.test_n_way
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y, auroc_softmax_scores[:,i],pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i in range(n_classes): 
            plt.plot(fpr[i], tpr[i], label='ISUP '+str(i+2)+' vs. rest (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(params.output_path,"TEST_ROC_CURVE_final.pdf"))
            plt.show()

if __name__ == '__main__':
    main()    