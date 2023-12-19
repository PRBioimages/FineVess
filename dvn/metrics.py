from keras.datasets import mnist
from keras.utils import np_utils
from keras import layers as KL
from keras import backend as K
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_recall_curve

def dice_score(y_true, y_pred):
    return f1_score(y_true.flatten(), y_pred.flatten())

def dice_information(y_true, y_pred):
    prec, rec, thres = precision_recall_curve(y_true.flatten(), y_pred.flatten())
    f1 = (2. * prec * rec) / (prec + rec)
    ind = np.argmax(f1)
    return prec[ind], rec[ind], f1[ind], thres[ind]

def threshold_accuracy(threshold=0.5):
    def metric(y_true, y_pred):
        pred = K.cast(K.greater_equal(y_pred, threshold),'int32')   #功能：将张量转换到不同的 dtype 并返回; >>> np.greater_equal([4, 2, 1], [2, 2, 2])  array([ True, True, False])
        return K.mean(K.equal(K.cast(y_true, 'int32'), pred))
    return metric

def categorical_accuracy(axis=-1):
    def accuracy(y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=axis),K.argmax(y_pred, axis=axis)))   #argmax函数通常用于对softmax层的输出进行后处理,predict(some_random_image)的输出是[0.02, 0.90, 0.06, 0.02]。然后，argmax([0.02, 0.90, 0.06, 0.02])立即给您这个类（1）。在
    return accuracy

# 我加的
# def dice(smooth=1):
#     def metric(y_true, y_pred):
#         y_true_f = K.flatten(y_true)  # 将 y_true 拉伸为一维.
#         y_pred_f = K.flatten(y_pred)
#         intersection = K.sum((y_true_f * y_pred_f))
#         union = K.sum(y_true_f) + K.sum(y_pred_f)
#         return K.mean((2. * intersection + smooth) / (union + smooth))  #这个不合适的原因可能是不该算均值
#     return metric


"""""""""#这个简单的dice会让loss变负数
def dice(smooth=1):  #我加的
    def metric(y_true, y_pred):
        y_true_f = K.flatten(y_true)  # 将 y_true 拉伸为一维.
        y_pred_f = K.flatten(y_pred)
       #  if y_true_f.shape[0] !="?":
       #  #if y_true_f.shape[0] is not None:
       # #  if y_true_f.dtype != NoneType:
       #      index = []  # 剔除掉标签为2的值，不参与计算。
       #      for i in range(y_true_f.shape[0]):
       #          if y_true_f[i] != 2:
       #              print(i)
       #              index.append(i)
       #      y_true_f = y_true_f[index]
       #      y_pred_f = y_pred_f[index]
       #  wh = K.tf.where(K.tf.not_equal(y_true_f, 2))
        intersection = K.sum((y_true_f * y_pred_f))
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        return K.mean((2. * intersection + smooth) / (union + smooth))
    return metric
"""""""""""

#有待研究：
# def label_wise_dice_coefficient(y_true, y_pred, label_index):
#     return dice(y_true[:, :,label_index,:,:], y_pred[:,:,label_index,:,:])



# 原来的
# def dice(smooth=1):
#     def metric(y_true, y_pred):
#         intersection = K.sum((y_true * y_pred)[:,1:,:,:,:])
#         union = K.sum(y_true[:,1:,:,:,:]) + K.sum(y_pred[:,1:,:,:,:])
#         return K.mean((2. * intersection + smooth) / (union + smooth))
#     return metric

def dice(smooth=1):  #适用于不同维度不同通道数的数据，如果是多个类别，那么softmax会输出多个通道，每个像素点对应三个通道（分别是（0.3，0.1，0，6）和（1，0，0）照样求覆盖情况
    def metric(y_true, y_pred):
        # ''''稀疏注释
        c_true = K.cast(K.not_equal(y_true, 2), K.dtype(y_pred))
        y_true = y_true * c_true
        y_pred = y_pred * c_true
        # '''''
        intersection = K.sum(y_true * y_pred, axis=list(range(1, K.ndim(y_true))))    #只要对于没有标注的层，intersection和union为0，那么就是不做贡献。 不管是loss还是dice都一样；计算完之后数组就变成一维的了。
        print('y_true',y_true.shape)   #y_true (?, ?, ?, ?, ?)  #第一维是batch数量
        print('y_pred',y_pred.shape)    #(?, 2, ?, ?, ?)
        print('intersection',intersection.shape)   #(?,)
        union = K.sum(y_true, axis=list(range(1, K.ndim(y_true)))) + K.sum(y_pred, axis=list(range(1, K.ndim(y_true))))
        print('union', union.shape)     #(?,)
        return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)  #应该是对所有volume（一个批次的）求平均
    return metric


def metric_dice(y_true, y_pred):
    smooth = 1
    # intersection = K.sum(y_true * y_pred, axis=list(range(1, K.ndim(y_true))))
    # print('y_true',y_true.shape)   #y_true (?, ?, ?, ?, ?)
    # print('y_pred',y_pred.shape)    #(?, 2, ?, ?, ?)
    # print('intersection',intersection.shape)   #(?,)
    # union = K.sum(y_true, axis=list(range(1, K.ndim(y_true)))) + K.sum(y_pred, axis=list(range(1, K.ndim(y_true))))
    # print('union', union.shape)     #(?,)

    # y_true_f = K.flatten(y_true)  # 将 y_true 拉伸为一维.
    # y_pred_f = K.flatten(y_pred)
    # intersection = K.sum((y_true_f * y_pred_f))
    # union = K.sum(y_true_f) + K.sum(y_pred_f)
    # return (2. * intersection + smooth) / (union + smooth)  #返回的是一个张量
    #return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    ##用numpy计算：
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)