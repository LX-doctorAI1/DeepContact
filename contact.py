# -*- coding: utf-8 -*-
# @Time    : 2021/1/31
# @Author  : Li Xiao, Shuxin Yang
# @Contact : aspenstars@qq.com, xiaoli@ict.ac.cn
# @FileName: contact.py
import cv2
import numpy as np
from copy import deepcopy
from PIL import Image
from myutils.visualise import label2rgb, vis_heatmap

def calContact(mito_pred, er_pred, image, px=3):
    """计算contact
        相交区域： C1 = M & E
        Mito的边缘： MM = Canny(M)
        Contact的线： C2 = C1 & MM
    """
    mito_pred = mito_pred.astype(np.uint8)
    num_mito, mitos = cv2.connectedComponents(mito_pred)

    num_cont, cont_len, mito_len = 0, 0, 0
    cont_image = np.zeros(image.shape[:2])
    # 每一个mito分开计算（因为有的会有多个接触点，直接计算可能会计算成多个contact）
    for m in range(1, num_mito):
        # 取出一个mito
        M = (mitos == m)
        # mito和ER的相交部分
        C = M & er_pred
        # Mito的边缘，Canny只能接受uint8数据类型，同时将图片转换为0-255范围
        MM = cv2.Canny(M.astype(np.uint8)*255, 30, 100)
        # Mito边缘和ER相交的部位
        contact = C & MM
        # 刚才转换了255，要转换回来
        mito_len += MM.sum() / 255
        length = contact.sum()
        if length > 0:
            cont_len += length
            num_cont += 1
            cont_image += contact

    cont_image = cont_image.astype(np.bool)
    vis = visualize(image, cont_image)

    return num_mito, mito_len, cont_len, num_cont, vis


def calContactDist(mito_pred, er_pred, image, px=3):
    """Follow Yang Liu and Xiao Li"""
    mito_pred = deepcopy(mito_pred)
    er_pred = deepcopy(er_pred)
    image = deepcopy(image)

    mito = mito_preprocess(mito_pred)
    er = er_pred.astype(np.uint8)

    mito_num, mitos = cv2.connectedComponents(mito)
    contact_map = np.zeros(mito.shape, np.int32)
    distmap = np.zeros(px + 1, np.int32)
    contact_flag = np.zeros(mito_num - 1, np.int32)

    # 遍历Mito图上的每个点
    for i in range(mito.shape[0]):
        for j in range(mito.shape[1]):
            # 存在Mito时搜索周围 px 范围内的点是否存在ER
            if mitos[i][j] > 0:
                is_contact, min_dist = False, px
                for p in range(-1 * (px), px + 1):
                    for q in range(-1 * (px), px + 1):
                        x, y = i + p, j + q
                        # 超出边界
                        if x < 0 or x >= 1024 or y < 0 or y >= 1024:
                            continue
                        # 存在ER时计算最近 ER 的距离
                        if er[x][y]:
                            dist = np.sqrt(p ** 2 + q ** 2)
                            if dist <= px:
                                # 标记此 Mito contact ER
                                contact_flag[mitos[i][j] - 1] = True
                                is_contact = True
                                min_dist = min(dist, min_dist)

                if is_contact:
                    if min_dist == 0:
                        distmap[0] += 1
                    else:
                        distmap[int(min_dist - 0.001) + 1] += 1
                    contact_map[i][j] = 1
    cont_len = contact_map.sum()
    cont_num = contact_flag.sum()
    mito_len = mitos.astype(np.bool).sum()

    er_contour = cv2.Canny(er * 255, 30, 100)
    er_len = er_contour.astype(np.bool).sum()

    vis = visualize(image, contact_map)
    return mito_num - 1, mito_len, cont_len, cont_num, vis, er_len, distmap


def mito_preprocess(mito_pred):
    mito_pred = mito_pred.astype(np.uint8)
    mito_pred[mito_pred < 255] = 0

    mito = cv2.GaussianBlur(mito_pred, (7, 7), 0)
    _, mito = cv2.threshold(mito, 100, 255, cv2.THRESH_BINARY)
    mito = cv2.Canny(mito, 30, 100)
    _, mito = cv2.threshold(mito, 100, 255, cv2.THRESH_BINARY)

    return mito.astype(np.uint8)

def visualize(image, mask):
    vis = deepcopy(image)
    vis[:, :, 0] = np.where(mask == True, 0, vis[:, :, 0])
    vis[:, :, 1] = np.where(mask == True, 0, vis[:, :, 1])
    vis[:, :, 2] = np.where(mask == True, 255, vis[:, :, 2])

    return vis


def calContactDist_range_10pix_min(mito_pred, er_pred, image, px=10):
    """Follow Yang Liu and Xiao Li"""
    """mito_pred: 传入的是原题上剪切的mito，像素值小于255的是真实的mito，等于255的是背景
       er_pred: 传入的是二值图像，0为背景，1为er（输出后转换为255）
    """
    mito_pred = deepcopy(mito_pred[...,0])
    er_pred = deepcopy(er_pred * 255).astype(np.uint8)
    image = deepcopy(image)
    h, w = image.shape[:2]

    # 得到Mito的边缘
    mito_pred[mito_pred < 255] = 0
    # 转换Mito的背景和前景
    mito_pred = (mito_pred == 0).astype(np.uint8) * 255
    mito = cv2.Canny(mito_pred, 30, 100)
    # 得到Mito的数量、分离每个Mito
    mito_num, mitos = cv2.connectedComponents(mito)

    # ER在计算最近距离时使用边缘
    # er = er_pred
    er = cv2.Canny(er_pred, 30, 100)

    er_mito_min_dist = np.ones((h, w)) * 100
    contact_map = np.ones((h, w), np.int32) * 11

    distmap = np.zeros(px + 1, np.int32)
    contact_flag = np.zeros(mito_num - 1, np.int32)

    # 计算mito与ER直接相交的点，判断overlap的时候使用整个
    # 加mito边缘是因为canny算子提取的边缘会出现1-2个像素的偏移
    overlap = ((mito_pred + mito) & er_pred)

    # 计算ER最近的mito距离
    for i in range(h):
        for j in range(w):
            # 排除ER和Mito Overlap的点
            if er[i][j] > 0 and not overlap[i][j]:
                for p in range(-1 * (px), px + 1):
                    for q in range(-1 * (px), px + 1):
                        x, y = i + p, j + q
                        # 超出边界
                        if x < 0 or x >= 1024 or y < 0 or y >= 1024:
                            continue
                        # 存在Mito时计算最近 Mito 的距离
                        if mito[x][y] and overlap[x][y] == 0:
                            dist = np.sqrt(p ** 2 + q ** 2)
                            er_mito_min_dist[i][j] = min(er_mito_min_dist[i][j], dist)

    # 遍历Mito图上的每个点
    for i in range(mito.shape[0]):
        for j in range(mito.shape[1]):
            # 存在Mito时搜索周围 px 范围内的点是否存在ER
            if mitos[i][j] > 0 and not overlap[i][j]:
                is_contact, min_dist = False, 11
                for p in range(-1 * (px), px + 1):
                    for q in range(-1 * (px), px + 1):
                        x, y = i + p, j + q
                        # 超出边界
                        if x < 0 or x >= 1024 or y < 0 or y >= 1024:
                            continue
                        # 存在ER时计算最近 ER 的距离
                        if er[x][y] and overlap[x][y] == 0:
                            dist = np.sqrt(p ** 2 + q ** 2)
                            if dist <= px and dist <= er_mito_min_dist[x][y] + 1:
                                # 标记此 Mito contact ER
                                is_contact = True
                                contact_flag[mitos[i][j] - 1] = True
                                min_dist = min(dist, min_dist)

                if is_contact:
                    if min_dist == 0:
                        distmap[0] += 1
                    else:
                        distmap[int(min_dist - 0.001) + 1] += 1
                    contact_map[i][j] = int(min_dist)

    distmap[0] = ((overlap > 0) & (mitos > 0)).sum()
    contact_map = np.where(overlap & mitos, 0, contact_map)
    cont_len = (contact_map < 11).sum()
    cont_num = contact_flag.sum()
    mito_len = mitos.astype(np.bool).sum()
    er_len = er.astype(np.bool).sum()

    vis = vis_heatmap(contact_map, image)
    return mito_num - 1, mito_len, cont_len, cont_num, vis, er_len, distmap


def calContactDist_er_elongation(mito_pred, er_pred, image, px=10):
    """Follow Yang Liu and Xiao Li"""
    """增加ER elongation
       mito_pred: 传入的是原题上剪切的mito，像素值小于255的是真实的mito，等于255的是背景
       er_pred: 传入的是二值图像，0为背景，1为er（输出后转换为255）
    """
    mito_pred = deepcopy(mito_pred[...,0])
    er_pred = deepcopy(er_pred * 255).astype(np.uint8)
    image = deepcopy(image)
    h, w = image.shape[:2]

    # 得到Mito的边缘
    mito_pred[mito_pred < 255] = 0
    # 转换Mito的背景和前景
    mito_pred = (mito_pred == 0).astype(np.uint8) * 255
    mito = cv2.Canny(mito_pred, 30, 100)
    # 得到Mito的数量、分离每个Mito
    mito_num, mitos = cv2.connectedComponents(mito)

    # ER在计算最近距离时使用边缘
    er = cv2.Canny(er_pred, 30, 100)

    er_mito_min_dist = np.ones((h, w)) * 100
    contact_map = np.ones((h, w), np.int32) * 11

    distmap = np.zeros(px + 1, np.int32)
    contact_flag = np.zeros(mito_num - 1, np.int32)

    # 计算mito与ER直接相交的点，判断overlap的时候使用整个
    # 加mito边缘是因为canny算子提取的边缘会出现1-2个像素的偏移
    overlap = ((mito_pred + mito) & er_pred)

    # 计算ER最近的mito距离
    for i in range(h):
        for j in range(w):
            # 排除ER和Mito Overlap的点
            if er[i][j] > 0 and not overlap[i][j]:
                for p in range(-1 * (px), px + 1):
                    for q in range(-1 * (px), px + 1):
                        x, y = i + p, j + q
                        # 超出边界
                        if x < 0 or x >= 1024 or y < 0 or y >= 1024:
                            continue
                        # 存在Mito时计算最近 Mito 的距离
                        if mito[x][y] and overlap[x][y] == 0:
                            dist = np.sqrt(p ** 2 + q ** 2)
                            er_mito_min_dist[i][j] = min(er_mito_min_dist[i][j], dist)

    # 遍历Mito图上的每个点
    for i in range(mito.shape[0]):
        for j in range(mito.shape[1]):
            # 存在Mito时搜索周围 px 范围内的点是否存在ER
            if mitos[i][j] > 0 and not overlap[i][j]:
                is_contact, min_dist = False, 11
                for p in range(-1 * (px), px + 1):
                    for q in range(-1 * (px), px + 1):
                        x, y = i + p, j + q
                        # 超出边界
                        if x < 0 or x >= 1024 or y < 0 or y >= 1024:
                            continue
                        # 存在ER时计算最近 ER 的距离
                        if er[x][y] and overlap[x][y] == 0:
                            dist = np.sqrt(p ** 2 + q ** 2)
                            if dist <= px and dist <= er_mito_min_dist[x][y] + 1:
                                # 标记此 Mito contact ER
                                is_contact = True
                                contact_flag[mitos[i][j] - 1] = True
                                min_dist = min(dist, min_dist)

                if is_contact:
                    if min_dist == 0:
                        distmap[0] += 1
                    else:
                        distmap[int(min_dist - 0.001) + 1] += 1
                    contact_map[i][j] = int(min_dist)

    distmap[0] = ((overlap > 0) & (mitos > 0)).sum()
    contact_map = np.where(overlap & mitos, 0, contact_map)
    cont_len = (contact_map < 11).sum()
    cont_num = contact_flag.sum()
    mito_len = mitos.astype(np.bool).sum()
    er_len = er.astype(np.bool).sum()

    # ER elongation
    num_er, ers = cv2.connectedComponents(er_pred)
    er_elongs = []
    for i in range(1, num_er):
        tmp = (ers == i)
        contour = cv2.Canny(tmp.astype(np.uint8) * 255, 30, 100)
        perimeter = contour.sum() / 255
        area = tmp.sum()
        er_elongs.append((perimeter ** 2) / (12.56 * area))
    er_elong = np.mean(er_elongs)

    vis = vis_heatmap(contact_map, image)
    return [mito_num - 1, mito_len, cont_len, cont_num, er_len, er_elong, vis, distmap]


if __name__ == "__main__":
    mito_pred = np.load("/Users/aspenstars/Downloads/results/mito_pred.npy")
    er_pred = np.load("/Users/aspenstars/Downloads/results/er_pred.npy")
    image = np.load("/Users/aspenstars/Downloads/results/aug_image.npy")

    mito_num, mito_len, cont_len, cont_num, vis, er_len, distmap = calContactDist_range_10pix_min(mito_pred, er_pred, image)
    # mito_num1, mito_len1, cont_len1, cont_num1, vis1, er_len1, distmap1 = calContactDist_vertical(mito_pred, er_pred, image)

    cv2.imwrite("debug.png", vis)
    # cv2.imwrite("debug2.png", vis1)
