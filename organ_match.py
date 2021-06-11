import cv2
import numpy as np
from copy import deepcopy
from PIL import Image

def matching(organ, pred, threshold=0.6):
    """计算手工标注的细胞器和预测结果之间的符合度
        先计算连通域，遍历所有手工标注的细胞器，计算与之
        相交的预测结果，大于threshold的计为符合
        返回： 符合的数量， 手工标注的总数， 预测的总数
    """
    assert organ.shape == pred.shape, "organ and pred's shape must match!"
    organ = organ.astype(np.uint8)
    pred = pred.astype(np.uint8)
    row, col = organ.shape

    n_organ, organs = cv2.connectedComponents(organ)
    n_pred, preds = cv2.connectedComponents(pred)

    # 记录已经计算过的，避免重复计算
    done = np.zeros((n_organ, n_pred))
    done_organ = np.zeros(n_organ)
    done_pred = np.zeros(n_pred)

    num_match, num_organ, num_pred = 0, 0, 0

    for x in range(row):
        for y in range(col):
            oid, pid = organs[x, y], preds[x, y]

            # 计数match个数
            if oid > 0 and pid > 0 and done[oid, pid] == 0:
                done[oid, pid] = 1

                T_organ = (organs == oid)
                if T_organ.sum() < 500:
                    done[oid, :] = 1
                    continue

                T_pred = (preds == pid)
                union = T_organ & T_pred
                match_score = union.sum() / T_organ.sum()

                if match_score >= threshold:
                    num_match += 1
                    done[oid, :] = 1
                    done[:, pid] = 1

            # 计数手工标注的个数
            if oid > 0 and done_organ[oid] == 0:
                done_organ[oid] = 1
                T_organ = (organs == oid)
                if T_organ.sum() >= 500:
                    num_organ += 1

            # 计数模型预测的个数
            if pid > 0 and done_pred[pid] == 0:
                done_pred[pid] = 1
                T_pred = (preds == pid)
                if T_pred.sum() >= 500:
                    num_pred += 1


    return num_match, num_organ, num_pred




