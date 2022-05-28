# -*- coding: utf-8 -*-
# @License : Apache Licence
# @Time    : 2020/10/4 10:35 上午
# @Author  : 杨树鑫
# @Contact : aspenstars@qq.com
# @FileName: criterion.py
import torch
import torch.nn as nn

# from ..builder import LOSSES


# @LOSSES.register_module()
class fscoreLoss(nn.Module):

    def __init__(self, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.threshold = threshold
        self.activation = activation

    @staticmethod
    def f_score(pr,
                gt,
                beta=1,
                eps=1e-7,
                threshold=None,
                activation='sigmoid'):
        """
        Args:
            pr (torch.Tensor): A list of predicted elements
            gt (torch.Tensor):  A list of elements that are to be predicted
            beta (float): positive constant
            eps (float): epsilon to avoid zero division
            threshold: threshold for outputs binarization
        Returns:
            float: F score
        """

        if activation is None or activation == 'none':
            activation_fn = lambda x: x  # noqa: E731
        elif activation == 'sigmoid':
            activation_fn = torch.nn.Sigmoid()
        elif activation == 'softmax2d':
            activation_fn = torch.nn.Softmax2d()
        else:
            raise NotImplementedError(
                'Activation implemented for sigmoid and softmax2d')

        pr = activation_fn(pr)

        if threshold is not None:
            pr = (pr > threshold).float()

        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp

        score = ((1 + beta**2) * tp + eps) / (
            (1 + beta**2) * tp + beta**2 * fn + fp + eps)

        return score

    def forward(self, outputs, annotation):
        loss = 0
        if type(outputs).__name__ == 'list':
            for output in outputs:
                loss += 1 - self.f_score(output, annotation, self.beta,
                                         self.eps, self.threshold,
                                         self.activation)
        else:
            loss += 1 - self.f_score(outputs, annotation, self.beta, self.eps,
                                     self.threshold, self.activation)

        return loss
