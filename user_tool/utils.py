import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def save_img_ch1(img, path):  # w*h, 0-1
    img = np.expand_dims(img, -1)*255   # w*h*1, 0-255
    img = np.concatenate([img, img, img], -1)  # w*h*3, 0-255
    cv2.imwrite(path, img)
def save_matrix_heatmap_visual(similar_distance_map,save_change_map_dir):

    from matplotlib import cm
    cmap = cm.get_cmap('jet', 30)
    plt.set_cmap(cmap)
    plt.imsave(save_change_map_dir,similar_distance_map)

def save_img_ch3(img, path):  # 3*w*h, 0-1
    img = np.transpose(img, (1, 2, 0)) * 255  # w*h*3, 0-255
    cv2.imwrite(path, img)

# 将图像二值化
def pic2binary(data):
    datasize = data.shape
    datax = np.zeros(datasize)
    for i in range(datasize[0]):
        for j in range(datasize[1]):
            if data[i][j] >= 128:
                datax[i][j] = 1
    return datax


def bilary2pic(data):  # 将0-1数值转化为二值化图像（可视化过程）
    datasize = data.shape
    for i in range(datasize[0]):
        for j in range(datasize[1]):
            if data[i][j] == 1:
                data[i][j] = 255
    data = np.expand_dims(data, -1)
    result = np.concatenate([data, data, data], -1)
    return result

def minmaxscaler(data):
    data = data.detach().cpu().numpy()
    min = np.amin(data)
    max = np.amax(data)
    result = torch.from_numpy((data - min) / (max - min))
    return result



def lr_schedule(epoch):
    lr = 0.01
    if epoch > 25:
        lr = 0.01
    elif epoch > 20:
        lr = 0.005
    else:
        if epoch > 15:
            lr = 0.001
        else:
            if epoch > 10:
                lr = 0.005
            else:
                if epoch > 5:
                    lr = 0.01
    print('Learning rate: ', lr)
    return lr

def otsu(data, num=400, get_bcm=False):
    """
    generate binary change map based on otsu
    :param data: cluster data
    :param num: intensity number
    :param get_bcm: bool, get bcm or not
    :return:
        binary change map
        selected threshold
    """
    max_value = np.max(data)
    min_value = np.min(data)

    total_num = data.shape[1]
    step_value = (max_value - min_value) / num
    value = min_value + step_value
    best_threshold = min_value
    best_inter_class_var = 0
    while value <= max_value:
        data_1 = data[data <= value]
        data_2 = data[data > value]
        if data_1.shape[0] == 0 or data_2.shape[0] == 0:
            value += step_value
            continue
        w1 = data_1.shape[0] / total_num
        w2 = data_2.shape[0] / total_num

        mean_1 = data_1.mean()
        mean_2 = data_2.mean()

        inter_class_var = w1 * w2 * np.power((mean_1 - mean_2), 2)
        if best_inter_class_var < inter_class_var:
            best_inter_class_var = inter_class_var
            best_threshold = value
        value += step_value
    if get_bcm:
        bwp = np.zeros(data.shape)
        bwp[data <= best_threshold] = 0
        bwp[data > best_threshold] = 255
        print('otsu is done')
        return bwp, best_threshold
    else:
        print('otsu is done')
        print('otsu==', best_threshold)
        return best_threshold

def evaluate(pred_label_data, true_label_data):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_label_data = np.reshape(pred_label_data, (1, -1))
    true_label_data = np.reshape(true_label_data, (1, -1))
    all_num = pred_label_data.shape[1]

    for i in range(all_num):
        if pred_label_data[0][i] == 255 and true_label_data[0][i] == 255:
            TP += 1
        elif pred_label_data[0][i] == 255 and true_label_data[0][i] == 0:
            FP += 1
        elif pred_label_data[0][i] == 0 and true_label_data[0][i] == 255:
            FN += 1
        else:
            TN += 1

    FPR = FP/(TP+FP+TN+FN)
    FNR = FN/(TP+FP+TN+FN)
    OE = FN+FP
    OER= FNR+FPR
    PCC = (TP+TN)/(TP+FP+TN+FN)
    PRE = ((TP+FP)*(TP+FN))/((TP+TN+FP+FN)**2) + ((FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2)
    Kappa = (PCC-PRE)/(1-PRE)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F = 2 * (Recall * Precision) / (Recall + Precision)

    print('FP:' + str(FP))
    print('FN:' + str(FN))
    print('OE:' + str(OE))
    print('FPR: %.4f' % (FPR))
    print('FNR: %.4f' % (FNR))
    print('OER:: %.4f' % (OER))
    print('PCC: %.4f' % (PCC))
    print('Kappa: %.4f' % (Kappa))
    print('F:  %.4f' % (F))
    return FP, FN, OE, FPR, FNR,OER, PCC, Kappa, F

def select_eva(pred_label_data, true_label_data):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_label_data = np.reshape(pred_label_data, (1, -1))
    true_label_data = np.reshape(true_label_data, (1, -1))
    all_num = pred_label_data.shape[1]

    for i in range(all_num):
        if pred_label_data[0][i] == 255 and true_label_data[0][i] == 255:
            TP += 1
        elif pred_label_data[0][i] == 255 and true_label_data[0][i] == 0:
            FP += 1
        elif pred_label_data[0][i] == 0 and true_label_data[0][i] == 255:
            FN += 1
        else:
            TN += 1

    Precision = TP / (TP + FP+ 1e-10)
    Recall = TP / (TP + FN)
    F = 2 * (Recall * Precision) / (Recall + Precision+ 1e-10)
    return F