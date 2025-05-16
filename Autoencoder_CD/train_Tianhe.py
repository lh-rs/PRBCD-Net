import torch
import numpy as np
from dataload import getSARData, getRGBData, getP0Data,getP02Data,getSARData0,data_Norm
from net import AEnet
from utils import save_img_ch1, save_img_ch3, minmaxscaler,  otsu
import cv2
import torch.nn as nn
from evaluate import select_eva,metric,evaluate
import itertools
import xlwt


def showresult(data):  # 将0-1数值转化为二值化图像（可视化过程）
    data = np.expand_dims(data, -1)
    result = np.concatenate([data, data, data], -1)
    return result

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
import os
def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

if __name__ == '__main__':

    img1_path = 'dataset/Tianhe/im1.bmp'
    img2_path = 'dataset/Tianhe/im2v.png'
    pretarin_model = torch.load('result/Tianhe/AE_result/CP2_val_0.002815.pth')

    import datetime
    date_object = datetime.date.today()

    newset_path =  'result/Tianhe/' + str(date_object) + '_item/'

    result_path = newset_path

    if not os.path.exists(newset_path):
        os.makedirs(newset_path)
    check_dir(newset_path)

    Channels = 32
    img1_data = getSARData0(img1_path)/255
    img2_data = getSARData0(img2_path)/255

    sz_img1 = img1_data.shape
    sz_img2 = img2_data.shape

    N = sz_img1[1] * sz_img1[2] * sz_img1[3]

    img1_data = img1_data.cuda()
    img2_data = img2_data.cuda()

    model = AEnet(sz_img1[1], Channels, sz_img1[1], Channels).cuda()
    model.load_state_dict(pretarin_model)

    for name, param in model.named_parameters():
        if name[5] == "2":
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)

    epochs = 1000

    i = 1
    wk = xlwt.Workbook()
    ws = wk.add_sheet('analysis')
    ws.write(0, 0, "epoch")
    ws.write(0, 1, "Channels")
    ws.write(0, 2, "FP")
    ws.write(0, 3, "FN")
    ws.write(0, 4, "OE")
    ws.write(0, 5, "PCC")
    ws.write(0, 6, "Kappa")
    ws.write(0, 7, "F")

    model.train()
    for epoch in range(epochs):

        F1_1, F2_1 = model(img1_data, img2_data)

        diff = torch.sqrt(torch.sum((F1_1 - F2_1) ** 2, dim=1))

        loss = torch.sum(diff) / N

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch: {}, loss: {}, diff: {}'.format(epoch, loss, torch.sum(diff) / diff.numel()))

        DI = diff.squeeze().cpu().detach().numpy()

        DI = data_Norm(DI)

    save_img_ch1(DI, result_path + 'ae_di_{}.png'.format(epoch))

    thresh = np.array(list(range(50, 240)))
    for th in thresh:
        seclect_result = np.where(DI*255 > th, 255, 0)

        cv2.imwrite(result_path + str(th) + 'ae_br.png', showresult(seclect_result))



