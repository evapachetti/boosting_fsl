# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti (based on the code of Wang Tan https://github.com/Wangt-CN/IP-IRM)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import utils
from model import IPIRM

# Ensure this environment variable is set after importing os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

# Training according to SimCLR with Contrastive Loss 
def train(net, data_loader, train_optimizer, temperature, debiased, tau_plus):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target, idx in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True) # augmented versions of the same image -> positive examples
        feature_1, out_1 = net(pos_1) # feature is the output of the CNN, out is feature after passing through MLP (both embedding vectors)
        feature_2, out_2 = net(pos_2)

        # Calculates components of contrastive loss

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature) 
        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if debiased:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# ssl training with IP-IRM
def train_env(net, data_loader, train_optimizer, temperature, updated_split):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for batch_index, data_env in enumerate(train_bar):
        # extract all feature
        pos_1_all, pos_2_all, indexs = data_env[0].cuda(), data_env[1].cuda(), data_env[-1].cuda()
        feature_1_all, out_1_all = net(pos_1_all)
        feature_2_all, out_2_all = net(pos_2_all)
        
        if args.keep_cont: # global contrastive loss (1st partition) (keep original contrastive loss of SSL, deafult=False)
            logits_all, labels_all = utils.info_nce_loss(torch.cat([out_1_all, out_2_all], dim=0), out_1_all.size(0), temperature=temperature)
            loss_original = torch.nn.CrossEntropyLoss()(logits_all, labels_all)

        env_contrastive, env_penalty = [], []

        if isinstance(updated_split, list): # if retain previous partitions
            assert args.retain_group
            for updated_split_each in updated_split:
                for env in range(args.env_num):
                    out_1, out_2 = utils.assign_features(out_1_all, out_2_all, indexs, updated_split_each, env)
                    # contrastive loss
                    logits, labels = utils.info_nce_loss(torch.cat([out_1, out_2], dim=0), out_1.size(0), temperature=1.0)
                    logits_cont = logits / temperature

                    loss = torch.nn.CrossEntropyLoss()(logits_cont, labels)
                    # penalty
                    logits_pen = logits / args.irm_temp
                    penalty_score = utils.penalty(logits_pen, labels, torch.nn.CrossEntropyLoss(), mode=args.ours_mode)

                    # collect it into env dict
                    env_contrastive.append(loss)
                    env_penalty.append(penalty_score)
                        

        else:
            for env in range(args.env_num):

                out_1, out_2 = utils.assign_features(out_1_all, out_2_all, indexs, updated_split, env)

                # contrastive loss
                logits, labels = utils.info_nce_loss(torch.cat([out_1, out_2], dim=0), out_1.size(0), temperature=1.0)
                logits_cont = logits / temperature
                logits_pen = logits / args.irm_temp

                loss = torch.nn.CrossEntropyLoss()(logits_cont, labels)
                # penalty
                penalty_score = utils.penalty(logits_pen, labels, torch.nn.CrossEntropyLoss(), mode=args.ours_mode)

                # collect it into env dict
                env_contrastive.append(loss)
                env_penalty.append(penalty_score)

        loss_cont = torch.stack(env_contrastive).mean()
        if args.keep_cont:
            loss_cont += loss_original

        if args.increasing_weight:
            penalty_weight = utils.increasing_weight(0, args.penalty_weight, epoch, args.epochs)
        elif args.penalty_iters < 200:
            penalty_weight = args.penalty_weight if epoch >= args.penalty_iters else 0.
        else:
            penalty_weight = args.penalty_weight
        irm_penalty = torch.stack(env_penalty).mean()
        loss_penalty = irm_penalty
        loss = loss_cont + penalty_weight * loss_penalty

        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] [{trained_samples}/{total_samples}]  Loss: {:.4f}  LR: {:.4f}  PW {:.4f}'
            .format(epoch, epochs, total_loss/total_num, train_optimizer.param_groups[0]['lr'], penalty_weight,
            trained_samples=batch_index * batch_size + len(pos_1_all),
            total_samples=len(data_loader.dataset)))

        if batch_index % 10 == 0:
            utils.write_log('Train Epoch: [{:d}/{:d}] [{:d}/{:d}]  Loss: {:.4f}  LR: {:.4f}  PW {:.4f}'
                            .format(epoch, epochs, batch_index * batch_size + len(pos_1_all), len(data_loader.dataset), total_loss/total_num,
                                    train_optimizer.param_groups[0]['lr'], penalty_weight), log_file=log_file)

    return total_loss / total_num



def train_update_split(net, update_loader, soft_split, random_init=False,root_path=''):
    utils.write_log('Start Maximizing ...', log_file, print_=True)
    if random_init:
        utils.write_log('Give a Random Split:', log_file, print_=True)
        soft_split = torch.randn(soft_split.size(), requires_grad=True, device="cuda")
        utils.write_log('%s' %(soft_split[:3]), log_file, print_=True)
    else:
        utils.write_log('Use Previous Split:', log_file, print_=True)
        soft_split = soft_split.requires_grad_()
        utils.write_log('%s' %(soft_split[:3]), log_file, print_=True)

    if args.offline: # Maximize Step offline, first extract image features
        net.eval()
        feature_bank_1, feature_bank_2 = [], []
        with torch.no_grad():
            # generate feature bank
            for pos_1, pos_2, target, Index in tqdm(update_loader_offline, desc='Feature extracting'):
                feature_1, out_1 = net(pos_1.cuda(non_blocking=True))
                feature_2, out_2 = net(pos_2.cuda(non_blocking=True))
                feature_bank_1.append(out_1.cpu())
                feature_bank_2.append(out_2.cpu())
        feature1 = torch.cat(feature_bank_1, 0)
        feature2 = torch.cat(feature_bank_2, 0)
        updated_split = utils.auto_split_offline(feature1, feature2, soft_split, temperature, args.irm_temp, loss_mode='v2', irm_mode=args.irm_mode,
                                         irm_weight=args.irm_weight_maxim, constrain=args.constrain, cons_relax=args.constrain_relax, nonorm=args.nonorm, log_file=log_file)
    else:
        updated_split = utils.auto_split(net, update_loader, soft_split, temperature, args.irm_temp, loss_mode='v2', irm_mode=args.irm_mode,
                                     irm_weight=args.irm_weight_maxim, constrain=args.constrain, cons_relax=args.constrain_relax, nonorm=args.nonorm, log_file=log_file)
    np.save(os.path.join(root_path,'GroupResults'+ str(epoch)+ ".txt"), updated_split.cpu().numpy())
    return updated_split



# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def evaluate(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_auroc, total_num, feature_bank = 0.0, 0.0, 0, []
    targets = []
    scores = []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        try:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)
        except:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            # To compute AUROC
            targets.append(target.cpu().item())
            scores.append(pred_scores.cpu()[0][1].item())

        total_auroc += roc_auc_score(targets,scores)
        test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.3f} AUROC:{:.3f}%'.format(epoch, epochs, total_top1 / total_num * 100, total_auroc * 100))

    return total_top1 / total_num * 100, total_auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay")
    parser.add_argument('--debiased', default=False, type=bool, help='Debiased contrastive loss or standard loss')
    parser.add_argument('--dataset', type=str, default='picai', choices=['picai','breakhis'], help='experiment dataset')
    parser.add_argument('--baseline', action="store_true", default=False, help='SSL baseline?')
    parser.add_argument("--pretrained_net", choices = ['Resnet18','Resnet50','VGG16','Densenet'],default=None, type=str, help="Pretrained network to fine-tune")
    parser.add_argument("--output_path", default="pretrained_models",type=str, help="Path save model")

    #### ours model param ####
    parser.add_argument('--ours_mode', default='w', type=str, help='what mode to use')
    parser.add_argument('--penalty_weight', default=1, type=float, help='penalty weight')
    parser.add_argument('--penalty_iters', default=0, type=int, help='penalty weight start iteration')
    parser.add_argument('--increasing_weight', action="store_true", default=False, help='increasing the penalty weight?')
    parser.add_argument('--env_num', default=2, type=int, help='num of the environments')

    parser.add_argument('--maximize_iter', default=30, type=int, help='when maximize iteration')
    parser.add_argument('--irm_mode', default='v1', type=str, help='irm mode when maximizing')
    parser.add_argument('--irm_weight_maxim', default=1, type=float, help='irm weight in maximizing')
    parser.add_argument('--irm_temp', default=0.5, type=float, help='irm loss temperature')
    parser.add_argument('--random_init', action="store_true", default=False, help='random initialization before every time update?')
    parser.add_argument('--constrain', action="store_true", default=False, help='make num of 2 group samples similar?')
    parser.add_argument('--constrain_relax', action="store_true", default=False, help='relax the constrain?')
    parser.add_argument('--retain_group', action="store_true", default=False, help='retain the previous group assignments?')
    parser.add_argument('--debug', action="store_true", default=False, help='debug?')
    parser.add_argument('--nonorm', action="store_true", default=False, help='not use norm for contrastive loss when maximizing')
    parser.add_argument('--groupnorm', action="store_true", default=False, help='use group contrastive loss?')
    parser.add_argument('--offline', action="store_true", default=False, help='save feature at the beginning of the maximize?')
    parser.add_argument('--keep_cont', action="store_true", default=False, help='keep original contrastive?')

    ### metadata file paths ###
    parser.add_argument('--csv_train_unsup', default='', type=str, help='unsupervised trainset path')
    parser.add_argument('--csv_val_unsup', default='', type=str, help='unsupervised valset path')
    parser.add_argument('--csv_test_unsup', default='', type=str, help='unsupervised testset path')
    parser.add_argument('--csv_train_sup', default='', type=str, help='supervised trainset path')
   
    # args parse
    args = parser.parse_args()

    for arg in vars(args):
         print (arg, getattr(args, arg))

    # seed
    utils.set_seed(42)

    # learning rate 
    learning_rate = 0.3*args.batch_size/256 # from original SimCLR paper

    # root path to save
    root_path = os.path.join(args.output_path, f"{args.pretrained_net}_{args.batch_size}_{learning_rate}_{args.weight_decay}")
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    batch_size, epochs, debiased = args.batch_size, args.num_epochs,  args.debiased
   
    log_file = os.path.join(root_path,'log.txt')
    
    model = IPIRM(args.pretrained_net, feature_dim=feature_dim)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    
    NUM_GPUS = 1
    device = torch.device("cuda")
    model = model.to(device)
    
    
    # data prepare
    if args.dataset == 'picai':
        # Train in unsupervised way
        train_data = utils.PICAIPair_Index(csv_path=args.csv_train_unsup, transform=utils.train_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4*NUM_GPUS, pin_memory=True,
                                  drop_last=True)
        update_data = utils.PICAIPair_Index(csv_path=args.csv_train_unsup, transform=utils.train_transform)
        update_loader = DataLoader(update_data, batch_size=batch_size, shuffle=True, num_workers=4*NUM_GPUS, pin_memory=True, drop_last=True)
        update_loader_offline = DataLoader(update_data, batch_size=batch_size, shuffle=False, num_workers=4*NUM_GPUS, pin_memory=True)
        # Evaluate in supervised way
        memory_data = utils.PICAIPair(csv_path = args.csv_train_sup, transform=utils.test_transform)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4*NUM_GPUS, pin_memory=True)
        test_data = utils.PICAIPair(csv_path = args.csv_val_sup, transform=utils.test_transform)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4*NUM_GPUS, pin_memory=True)
    elif args.dataset == 'breakhis':
        # Train in unsupervised way
        train_data = utils.BREAKHISPair_Index(csv_path=args.csv_train_unsup, transform=utils.train_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4*NUM_GPUS, pin_memory=True,
                                  drop_last=True)
        update_data = utils.BREAKHISPair_Index(csv_path=args.csv_train_unsup, transform=utils.train_transform)
        update_loader = DataLoader(update_data, batch_size=batch_size, shuffle=True, num_workers=4*NUM_GPUS, pin_memory=True, drop_last=True)
        update_loader_offline = DataLoader(update_data, batch_size=batch_size, shuffle=False, num_workers=4*NUM_GPUS, pin_memory=True)
        # Evaluate in supervised way
        memory_data = utils.BREAKHISPair(csv_path = args.csv_train_sup, cls_type="binary", transform=utils.test_transform)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4*NUM_GPUS, pin_memory=True)
        test_data = utils.BREAKHISPair(csv_path = args.csv_val_sup, cls_type="binary", transform=utils.test_transform)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4*NUM_GPUS, pin_memory=True)
    else:
        raise Exception("Dataset not implemented")

    c = len(memory_data.classes)
    print('# Classes: {}'.format(c))

    
    epoch = 0
    # update partition for the first time
    if not args.baseline:
        updated_split = torch.randn((len(update_data.data), args.env_num), requires_grad=True, device="cuda")
        updated_split = train_update_split(model, update_loader, updated_split, random_init=args.random_init,root_path=root_path)
        updated_split_all = [updated_split.clone().detach()]

    losses = []
    for epoch in range(1, epochs + 1):
        if args.baseline:
            train_loss = train(model, train_loader, optimizer, temperature, debiased, tau_plus)
        else: # Minimize Step
            if args.retain_group: # retain the previous partitions
                train_loss = train_env(model, train_loader, optimizer, temperature, updated_split_all)
                losses.append(train_loss)
            else:
                train_loss = train_env(model, train_loader, optimizer, temperature, updated_split)

            if epoch % args.maximize_iter == 0: # Maximize Step
                updated_split = train_update_split(model, update_loader, updated_split, random_init=args.random_init,root_path=root_path)
                updated_split_all.append(updated_split)

        if epoch % 100 == 0: # eval knn every 100 epochs
            test_acc, test_auroc = evaluate(model, memory_loader, test_loader)
            txt_write = open(os.path.join(root_path, 'knn_result.txt'), 'a')
            txt_write.write('\n Epoch: {} test_acc@1: {} test_auroc:{}'.format(epoch, test_acc, test_auroc))
            torch.save(model.state_dict(), os.path.join(root_path, 'model_{}.pth'.format(epoch)))

    plt.figure()
    plt.plot(range(1,epochs+1),losses,color='navy')
    plt.title('LOSS')
    plt.show()
    plt.savefig(os.path.join(root_path,"LOSS"+str(epochs)+".pdf"))

   
