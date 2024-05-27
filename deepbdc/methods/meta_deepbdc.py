# -*- coding: utf-8 -*-
"""
Author: Eva Pachetti (based on the code of Fei Long https://github.com/Fei-Long121/DeepBDC)
"""

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.autograd import Variable
from .template import MetaTemplate
from .bdc_module import BDC
from libauc.losses import AUCMLoss



class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.params = params
        self.loss_fn = AUCMLoss()

        reduce_dim = params.reduce_dim
        self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
        self.dcov = BDC(is_vec=True, input_dim=self.feature.feat_dim, dimension_reduction=reduce_dim)

    def feature_forward(self, x):
        out = self.dcov(x)
        return out

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.metric(z_query, z_proto)
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.type(torch.LongTensor)
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores = self.set_forward(x)
        _, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)

        scores = torch.nn.Softmax(dim=1)(scores)
        auroc_scores = scores.detach().cpu().numpy()
        auroc = roc_auc_score(
                y_label,
                auroc_scores if self.params.metatest_dataset == 'picai' else auroc_scores[:, 1],
                average='macro' if self.params.metatest_dataset == 'picai' else None,
                multi_class='ovr' if self.params.metatest_dataset == 'picai' else None
                )

        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores, auroc

    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        if self.n_support > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            score = (x * y).sum(2)
        return score
