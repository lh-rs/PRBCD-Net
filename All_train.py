import torch
import numpy as np
from user_tool.dataload import getRGBData, getP0Data
from models.MCAE import MCAE, clNet
from user_tool.utils import save_img_ch1, save_img_ch3, minmaxscaler, otsu,save_matrix_heatmap_visual
import cv2
from user_tool.evaluate import select_eva, evaluate, metric
from models.loss import ContrastiveLoss, Smooth_contrastive
import xlwt
import os
import matplotlib.pyplot as plt
import random
import datetime

def showresult(data):  # 将0-1数值转化为二值化图像（可视化过程）
    data = np.expand_dims(data, -1)
    result = np.concatenate([data, data, data], -1)
    return result

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.deterministic = True

def main():

    date_object = datetime.date.today()
    fea_type = 'Fea_Con'  # 选择模式： Fea_Con,Fea_Diff
    Recon_type = 'Cross'  # 选择模式：Cross , none

    for dataset in ['ShuguangVillage2','Italy','Tianhe','#4Gloucester']:

        if dataset == 'Tianhe':
            setup_seed(2021)
            img1_path = 'dataset/Tianhe/im1.bmp'
            img2_path = 'dataset/Tianhe/im2.bmp'
            ref_path = 'dataset/Tianhe/im3.bmp'
            pse_path = 'dataset/Tianhe/pse_data/ae_br.png'
            m = 0.1
        elif dataset == 'ShuguangVillage2':
            setup_seed(2021)
            img1_path = 'dataset/ShuguangVillage2/im1.png'
            img2_path = 'dataset/ShuguangVillage2/im2.png'
            ref_path = 'dataset/ShuguangVillage2/im3.png'
            pse_path = 'dataset/ShuguangVillage2/pse_data/ae_br.png'
            m = 0.1
        elif dataset == '#4Gloucester':
            setup_seed(2022)
            img1_path = 'dataset/#4Gloucester/im1.jpg'
            img2_path = 'dataset/#4Gloucester/im2.jpg'
            ref_path = 'dataset/#4Gloucester/im3.jpg'
            pse_path = 'dataset/#4Gloucester/pse_data/ae_br.png'
            m = 0.1 # m = 0.1 or 0.2
        elif dataset == 'Italy':
            setup_seed(2023)
            img1_path = 'dataset/Italy/im1.bmp'
            img2_path = 'dataset/Italy/im2.bmp'
            ref_path = 'dataset/Italy/im3.bmp'
            pse_path = 'dataset/Italy/pse_data/ae_br.png'
            m = 0.3

        result_path0 = 'results2/' + dataset + '_' + str(date_object) + '/'
        result_path = os.path.join(result_path0)

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        check_dir(result_path)

        Channels = 20

        img1_data = getRGBData(img1_path)
        img2_data = getRGBData(img2_path)

        ref_data = cv2.imread(ref_path)[..., 0]
        sz_img1 = img1_data.shape
        sz_img2 = img2_data.shape

        N = sz_img1[1] * sz_img1[2] * sz_img1[3]
        img1_data = img1_data.cuda()
        img2_data = img2_data.cuda()

        # Randomly initializing
        pcx_2 = torch.rand(sz_img1[1], sz_img1[2], sz_img1[3]).cuda()
        pcx_1 = torch.rand(sz_img1[1], sz_img1[2], sz_img1[3]).cuda()

        print('Randomly initializing Pc: {}'.format(torch.sum(pcx_1) / pcx_1.numel()))

        model_1 = MCAE(sz_img1[1], Channels, sz_img2[1], Channels).cuda()
        model_2 = MCAE(sz_img1[1], Channels, sz_img2[1], Channels).cuda()

        ##########****************
        if fea_type == "Fea_Con":
            net_1 = clNet(2 * Channels, 1).cuda()  # fea_con
            net_2 = clNet(2 * Channels, 1).cuda()  # fea_con
        elif fea_type == 'Fea_Diff':
            net_1 = clNet(Channels, 1).cuda()  # fea_diff
            net_2 = clNet(Channels, 1).cuda()  # fea_diff
        else:
            print(" clNet error")

        CL = ContrastiveLoss().cuda()
        ReconstructionLoss = torch.nn.MSELoss().cuda()

        '''双分类器'''
        optimizer1 = torch.optim.RMSprop([{'params': model_1.parameters(), 'lr': 0.001},
                                          {'params': model_2.parameters(), 'lr': 0.001},
                                          {'params': net_1.parameters(), 'lr': 0.001},
                                          {'params': net_2.parameters(), 'lr': 0.001}
                                          ])

        epochs = 50
        iters = 100

        i = 1
        wk = xlwt.Workbook()
        ws = wk.add_sheet('analysis')
        ws.write(0, 0, "epoch")
        ws.write(0, 1, "m")
        ws.write(0, 2, "FP")
        ws.write(0, 3, "FN")
        ws.write(0, 4, "OE")
        ws.write(0, 5, "PCC")
        ws.write(0, 6, "Kappa")
        ws.write(0, 7, "F")

        ws.write(0, 9, "acc_un")
        ws.write(0, 10, "acc_chg")
        ws.write(0, 11, "acc_all")
        ws.write(0, 12, "acc_tp")

        GlobalMask = torch.ones(img1_data.shape[-2:]).cuda()
        last_pse_data_1 = torch.zeros_like(getP0Data(pse_path).unsqueeze(0)).cuda()
        last_pse_data_2 = torch.zeros_like(getP0Data(pse_path).unsqueeze(0)).cuda()
        for epoch in range(epochs):
            if epoch == 0:
                pse_path_c = pse_path
                pse_data_c = getP0Data(pse_path).unsqueeze(0).cuda()
            else:
                pse_path=result_path + str(epoch - 1) + '_' + 'bestresult.png'
                pse_path_c = result_path + str(epoch - 1) + '_' + 'bestresult.png'
                pse_data_c = getP0Data(pse_path).unsqueeze(0).cuda()

            loss_best = 9999999
            model_1.train()
            model_2.train()
            for iter in range(iters):
                if 'model_1':
                    F1_brand1, F2_brand1, x1_recon_brand1, x2_recon_brand1 = model_1(img1_data, img2_data)
                    if Recon_type=='Cross':
                        reconLoss_1 = ReconstructionLoss(x1_recon_brand1, img2_data)
                        reconLoss_2 = ReconstructionLoss(x2_recon_brand1, img1_data)
                    else:
                        reconLoss_1 = ReconstructionLoss(x1_recon_brand1, img1_data)
                        reconLoss_2 = ReconstructionLoss(x2_recon_brand1, img2_data)
                    assert pse_data_c.shape[1] == 1

                    Loss_Reconstruct_1 = (reconLoss_1 + reconLoss_2) / 2

                    diff_1 = torch.sqrt(torch.mean((F1_brand1.detach() - F2_brand1) ** 2, dim=1))

                    unchanged_weithts = pse_data_c.sum() / (sz_img1[2] * sz_img1[3])

                    weight = pse_data_c * 1 + (1 - pse_data_c) * m

                    Loss_CL_1 = CL(diff_1, pcx_1, unchanged_weithts)

                    ##########****************#
                    if fea_type == "Fea_Con":
                        pos_fea_1 = torch.cat([F1_brand1, F2_brand1], dim=1)  # fea_con

                    elif fea_type == 'Fea_Diff':
                        pos_fea_1 = torch.sqrt((F1_brand1 - F2_brand1) ** 2 + 1e-15)  # fea_diff
                    else:
                        print(" pos_fea_1 error")

                    pre_data_1 = net_1(pos_fea_1)  # [1,1,w,h]

                    Loss_Classify_1 = torch.mean((pre_data_1 - pse_data_c) ** 2 * weight)

                if 'model_2':
                    F1_brand2, F2_brand2, x1_recon_brand2, x2_recon_brand2 = model_2(img1_data, img2_data)
                    if Recon_type == 'Cross':
                        reconLoss_1 = ReconstructionLoss(x1_recon_brand2, img2_data)
                        reconLoss_2 = ReconstructionLoss(x2_recon_brand2, img1_data)
                    else:
                        reconLoss_1 = ReconstructionLoss(x1_recon_brand2, img1_data)
                        reconLoss_2 = ReconstructionLoss(x2_recon_brand2, img2_data)
                    assert pse_data_c.shape[1] == 1

                    Loss_Reconstruct_2 =(reconLoss_1 + reconLoss_2) / 2
                    diff_2 = torch.sqrt(torch.mean((F1_brand2 - F2_brand2.detach()) ** 2, dim=1))

                    unchanged_weithts = pse_data_c.sum() / (sz_img1[2] * sz_img1[3])

                    weight = pse_data_c * 1 + (1 - pse_data_c) * m

                    Loss_CL_2 = CL(diff_2, pcx_2, unchanged_weithts)

                    ##########****************#
                    if fea_type == "Fea_Con":
                        pos_fea_2 = torch.cat([F1_brand2, F2_brand2], dim=1)  # fea_con

                    elif fea_type == 'Fea_Diff':
                        pos_fea_2 = torch.sqrt((F1_brand2 - F2_brand2) ** 2 + 1e-15)  # fea_diff
                    else:
                        print(" pos_fea error")

                    pre_data_2 = net_2(pos_fea_2)  # [1,1,w,h]

                    Loss_Classify_2 = torch.mean((pre_data_2 - pse_data_c) ** 2 * weight)

                if iter <= 20:
                    loss = 10000 + Loss_Reconstruct_1 + Loss_Classify_1 +Loss_Reconstruct_2 + Loss_Classify_2
                else:
                    loss =1*( Loss_Reconstruct_1 + Loss_Reconstruct_2) + 3 * (Loss_CL_1+Loss_CL_2) +Loss_Classify_1 +Loss_Classify_2

                optimizer1.zero_grad()
                loss.backward(retain_graph=False)
                optimizer1.step()

                info = ''
                if loss_best > loss:
                    loss_best = loss
                    pre_data_best_1 = pre_data_1
                    pre_data_best_2 = pre_data_2
                    F1_brand1_best, F2_brand1_best = F1_brand1, F2_brand1
                    F1_brand2_best, F2_brand2_bst = F1_brand2, F2_brand2
                    diff_best_1 = diff_1
                    diff_best_2 = diff_2
                    pos1_fea_best = pos_fea_1
                    pos2_fea_best = pos_fea_2
                    x1_recon_best_1 = x1_recon_brand1
                    x1_recon_best_2 = x1_recon_brand2
                    x2_recon_best_1 = x2_recon_brand1
                    x2_recon_best_2 = x2_recon_brand2
                    info = '\tsave best results....'
                else:
                    info = ''

                if iter % 10 == 0 or len(info) > 0:
                    print('epoch:{},iter: {}, diff:{:.8f}-{:.8f}, pcx:{:.8f}-{:.8f},  CL:{:.8f}-{:.8f}, classify:{:.8f}-{:.8f}, reconstruct:{:.8f}-{:.8f}, total_loss: {}, {}'.
                          format(epoch, iter,
                                 torch.sum(diff_best_1) / diff_best_1.numel(),torch.sum(diff_best_2) / diff_best_2.numel(),
                                 torch.sum(pcx_1) / pcx_1.numel(), torch.sum(pcx_2) / pcx_2.numel(),
                                 Loss_CL_1,Loss_CL_2,
                                 Loss_Classify_1,Loss_Classify_2,
                                 Loss_Reconstruct_1,Loss_Reconstruct_2,
                                 loss,
                                 info))
            '''第一路生成差异图'''
            pre_data_1 = pre_data_best_1.squeeze(0).detach()
            pcx_1 = pre_data_1

            last_pcx_1 = pcx_1 if epoch == 0 else last_pcx_1 + pcx_1
            img2_1 = (last_pcx_1 / (epoch + 1)).squeeze().detach().cpu().numpy()

            '''第二路生成差异图'''
            pre_data_2 = pre_data_best_2.squeeze(0).detach()
            pcx_2 = pre_data_2

            img1_2 = diff_best_2.squeeze().cpu().detach().numpy()
            img2_2 = pcx_2.squeeze().cpu().detach().numpy()

            last_pcx_2 = pcx_2 if epoch == 0 else last_pcx_2 + pcx_2    #历史的平均值，滑动平均
            img2_2 = (last_pcx_2 / (epoch + 1)).squeeze().detach().cpu().numpy()

            # ## 融合策略
            bestPc = 0.5 * img2_1 + 0.5 * img2_2
            last_pc = cv2.imread(pse_path_c)[..., 0]

            Fbefore = 1e15
            thresh = np.array(list(range(80, 200))) / 255
            # thresh = np.array(list(range(1, 255))) / 255
            best_th = 0
            for th in thresh:
                seclect_result = np.where(bestPc > th, 255, 0)
                F0 =  np.mean((seclect_result - last_pc) ** 2 )
                # F0 = 1 / np.sum((seclect_result - last_pc) ** 2 + 1e-15)
                if F0 < Fbefore:
                    cv2.imwrite(result_path + str(epoch) + '_' + 'bestresult.png', showresult(seclect_result))
                    Fbefore = F0
                    best_th = th
            print('\nbest_th=======================', best_th)
            bestresult = cv2.imread(result_path + str(epoch) + '_' + 'bestresult.png')[..., 0]

            FP, FN, OE, FPR, FNR, OER, PCC, Kappa, F = evaluate(bestresult, ref_data)
            acc_un, acc_chg, acc_all, acc_tp = metric(bestresult, ref_data)

        ws.write(i, 0, epoch)
        ws.write(i, 1, m)
        ws.write(i, 2, FP)
        ws.write(i, 3, FN)
        ws.write(i, 4, OE)
        ws.write(i, 5, PCC)
        ws.write(i, 6, Kappa)
        ws.write(i, 7, F)

        ws.write(i, 9, acc_un)
        ws.write(i, 10, acc_chg)
        ws.write(i, 11, acc_all)
        ws.write(i, 12, acc_tp)

        i += 1

        wk.save(result_path + dataset + '_Eval.xls')


if __name__ == '__main__':
    main()

