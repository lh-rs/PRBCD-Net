import numpy as np
def evaluate2(pred_label_data, true_label_data):
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

    FP_rate = FP/(TP+FP+TN+FN)
    FN_rate = FN/(TP+FP+TN+FN)
    # OE = FP_rate+FN_rate
    OE = FP + FN
    PCC = (TP+TN)/(TP+FP+TN+FN)
    PRE = ((TP+FP)*(TP+FN))/((TP+TN+FP+FN)**2) + ((FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2)
    Kappa = (PCC-PRE)/(1-PRE)

    # 绘制PR曲线：Precision纵坐标，Recall横坐标  #PR曲线下的面积称为AP
    Precision = TP / (TP + FP) #分类准确率
    Recall = TP / (TP + FN)  #真正类率

    # 绘制roc曲线：TPR纵坐标，FPR横坐标    #ROC曲线下的面积称为AUC
    TPR = Recall  # 真正类率 TPR=TP / (TP + FN)
    FPR = FP / (TN + FP)  # 伪正类率 FPR=FP/(TN+FP)

    F1 = 2 * (Recall * Precision) / (Recall + Precision)  # a=0.5

    print('FP:' + str(FP))
    print('FN:' + str(FN))
    print('TP:' + str(TP))
    print('TN:' + str(TN))
    print('FN_rate:%.4f' %(FN_rate))
    print('FP_rate: %.4f' % (FP_rate))
    print('OE: %.4f' % (OE))
    print('PCC:%.4f' %(PCC))
    print('PRE:%.4f' %(PRE))
    print('Precision: ' + str(Precision))
    print('Recall: ' + str(Recall))
    print('Kappa: %.4f'%(Kappa))
    print('F1: %.4f' % (F1))

    return FP, FN, FN_rate, FP_rate,OE, PCC, PRE, Kappa, Precision,Recall,F1,TPR,FPR

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

def metric(img, chg_ref):

    chg_ref = np.array(chg_ref, dtype=np.float32)

    chg_ref = chg_ref / np.max(chg_ref) #[0,255]--->[0,1]
    img = img / np.max(img)

    img = np.reshape(img, [-1])
    chg_ref = np.reshape(chg_ref, [-1])

    loc1 = np.where(chg_ref == 1)[0]
    num1 = np.sum(img[loc1] == 1)
    acc_chg = np.divide(float(num1), float(np.shape(loc1)[0]))

    loc2 = np.where(chg_ref == 0)[0]
    num2 = np.sum(img[loc2] == 0)
    acc_un = np.divide(float(num2), float(np.shape(loc2)[0]))

    acc_all = np.divide(float(num1 + num2), float(np.shape(loc1)[0] + np.shape(loc2)[0]))

    loc3 = np.where(img == 1)[0]
    num3 = np.sum(chg_ref[loc3] == 1)
    acc_tp = np.divide(float(num3), float(np.shape(loc3)[0]))

    print('Accuracy of Unchanged Regions is: %.4f' % (acc_un))
    print('Accuracy of Changed Regions is:   %.4f' % (acc_chg))
    print('The True Positive ratio is:       %.4f' % (acc_tp))
    print('Accuracy of all testing sets is : %.4f' % (acc_all))
    print('')

    return acc_un, acc_chg, acc_all, acc_tp

