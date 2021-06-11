eps = 1e-6
def iou(y_pred, y_true):
#     y_pred = (y_pred >= 0.5).astype('int32')
#     print(y_pred.dtype, y_true.dtype)
    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    return inter/(union+eps)