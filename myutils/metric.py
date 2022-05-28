# -*- coding: utf-8 -*-
# @Time    : 2020/6/13 6:01 下午
# @Author  : 杨树鑫
# @FileName: metric.py
eps = 1e-6
def iou(y_pred, y_true):
#     y_pred = (y_pred >= 0.5).astype('int32')
#     print(y_pred.dtype, y_true.dtype)
    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    return inter/(union+eps)


def dice(y_pred, y_true):
    #     y_pred = (y_pred >= 0.5).astype('int32')
    #     print(y_pred.dtype, y_true.dtype)
    inter = (y_true & y_pred).sum()
    mask_sum = y_true.sum() + y_pred.sum()
    return (2 * inter)/(mask_sum+eps)