# -*- coding: utf-8 -*-
# @License : Apache Licence
# @Time    : 2021/2/2 13:10 上午
# @Auther  : Li Xiao,Shuxin Yang
# @Contact : xiaoli@ict.ac.cn
# @FileName: runtest.py
import cv2
import numpy as np
from PIL import Image


def calinter(mito, ER, px, ori):
    # print(mito)
    im1 = mito.astype(np.uint8)
    im1[im1 < 255] = 0
    im = cv2.GaussianBlur(im1, (7, 7), 0)
    ret, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
    im = cv2.Canny(im, 30, 100)
    ret, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)

    mp = ER

    num_component, component = cv2.connectedComponents(im.astype(np.uint8))
    predictions = np.zeros(im.shape, np.int32)
    contourmap = np.zeros(im.shape, np.int32)
    x, y = im.shape
    inter_flag = np.zeros(num_component - 1, np.int32)
    for i in range(x):
        for j in range(y):
            if component[i][j] > 0:
                contourmap[i][j] = 1
                is_inter = False
                for p in range(-1 * (px), px + 1):
                    for q in range(-1 * (px), px + 1):
                        if (i + p < 0 or i + p >= 1024 or j + q < 0 or j + q >= 1024):
                            continue
                        if mp[i + p][j + q]:
                            inter_flag[component[i][j] - 1] = True
                            is_inter = True
                            break
                    if is_inter: break
                if is_inter:
                    predictions[i][j] = 1
    inter = predictions.sum()
    tot_num = inter_flag.sum()
    lengthsum = contourmap.sum()

    img = np.array(ori)
    img[:, :, 2] = np.where(predictions == True, 255, img[:, :, 2])
    img[:, :, 1] = np.where(predictions == True, 0, img[:, :, 1])
    img[:, :, 0] = np.where(predictions == True, 0, img[:, :, 0])

    return num_component - 1, lengthsum, inter, tot_num, img


def calinter_dist(mito, ER, px, ori):
    # print(mito)
    im1 = mito.astype(np.uint8)
    im1[im1 < 255] = 0
    im = cv2.GaussianBlur(im1, (7, 7), 0)
    ret, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
    im = cv2.Canny(im, 30, 100)
    ret, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)

    mp = ER

    num_component, component = cv2.connectedComponents(im.astype(np.uint8))
    predictions = np.zeros(im.shape, np.int32)
    contourmap = np.zeros(im.shape, np.int32)
    distmap = np.zeros(px + 1, np.int32)
    x, y = im.shape
    inter_flag = np.zeros(num_component - 1, np.int32)
    for i in range(x):
        for j in range(y):
            if component[i][j] > 0:
                contourmap[i][j] = 1
                is_inter = False
                dist0 = px
                for p in range(-1 * (px), px + 1):
                    for q in range(-1 * (px), px + 1):
                        if (i + p < 0 or i + p >= 1024 or j + q < 0 or j + q >= 1024):
                            continue
                        if mp[i + p][j + q]:
                            dist = np.sqrt(p * p + q * q)
                            if dist <= px:
                                inter_flag[component[i][j] - 1] = True
                                is_inter = True
                                if dist < dist0:
                                    dist0 = dist
                if is_inter:
                    if dist0 == 0:
                        distmap[0] += 1
                    else:
                        distmap[int(dist0 - 0.001) + 1] += 1
                    predictions[i][j] = 1
    inter = predictions.sum()
    tot_num = inter_flag.sum()
    lengthsum = contourmap.sum()

    dist_ER = 0
    mp = cv2.Canny(ER.astype(np.uint8) * 255, 30, 100)
    num_component, component = cv2.connectedComponents(mp.astype(np.uint8))
    for c in range(1, num_component):
        T = (component == c)
        dist_ER += T.sum()

    img = np.array(ori)
    img[:, :, 2] = np.where(predictions == True, 255, img[:, :, 2])
    img[:, :, 1] = np.where(predictions == True, 0, img[:, :, 1])
    img[:, :, 0] = np.where(predictions == True, 0, img[:, :, 0])

    return num_component - 1, lengthsum, inter, tot_num, img, dist_ER, distmap


def calinter_range(mito, ER, px, py, ori):
    # print(mito)
    im1 = mito.astype(np.uint8)
    im1[im1 < 255] = 0
    im = cv2.GaussianBlur(im1, (7, 7), 0)
    ret, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
    im = cv2.Canny(im, 30, 100)
    ret, im = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)

    mp = ER

    num_component, component = cv2.connectedComponents(im.astype(np.uint8))
    predictions = np.zeros(im.shape, np.int32)
    contourmap = np.zeros(im.shape, np.int32)
    x, y = im.shape
    inter_flag = np.zeros(num_component - 1, np.int32)

    for i in range(x):
        for j in range(y):
            if component[i][j] > 0:
                contourmap[i][j] = 1
                is_inter = False
                for p in range(-1 * (py), py + 1):
                    for q in range(-1 * (py), py + 1):
                        if (i + p < 0 or i + p >= 1024 or j + q < 0 or j + q >= 1024):
                            continue
                        if mp[i + p][j + q]:
                            inter_flag[component[i][j] - 1] = True
                            is_inter = True
                            break
                    if is_inter: break
                if is_inter:
                    for p in range(-1 * (px), px + 1):
                        for q in range(-1 * (px), px + 1):
                            if (i + p < 0 or i + p >= 1024 or j + q < 0 or j + q >= 1024):
                                continue
                            if mp[i + p][j + q]:
                                inter_flag[component[i][j] - 1] = False
                                is_inter = False
                                break
                        if is_inter == False: break
                if is_inter:
                    predictions[i][j] = 1
    inter = predictions.sum()
    tot_num = inter_flag.sum()
    lengthsum = contourmap.sum()

    img = np.array(ori)
    img[:, :, 2] = np.where(predictions == True, 255, img[:, :, 2])
    img[:, :, 1] = np.where(predictions == True, 0, img[:, :, 1])
    img[:, :, 0] = np.where(predictions == True, 0, img[:, :, 0])

    return num_component - 1, lengthsum, inter, tot_num, img


if __name__ == "__main__":
    mito = np.zeros((1024, 1024), dtype=np.uint8)
    er = np.zeros((1024, 1024), dtype=np.uint8)
    ori = np.zeros((1024, 1024, 3), dtype=np.uint8)
    calinter(mito, er, 3, ori)
