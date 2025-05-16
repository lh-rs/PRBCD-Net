import torch
import torch.nn as nn
from dataload import getSARData, getRGBData,ennoisegaussian
import numpy as np
from utils import save_img_ch1, save_img_ch3
from net import AEnet

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     # random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# setup_seed(0)

def ennoise1(x):  # SAR, 高斯散斑噪声
    sz_x = x.shape
    noise = torch.randn(sz_x)
    x = x.cpu()
    x += x * noise
    x = torch.where(torch.BoolTensor(x > 1), torch.ones(sz_x), x)
    x = torch.where(torch.BoolTensor(x < -1), -torch.ones(sz_x), x)
    return x.detach().cuda()


def pretrain():

    dataname = 'Tianhe'
    img1_path = 'dataset/Tianhe/im1.bmp'
    img2_path = 'dataset/Tianhe/im2v.png'

    Channels = 32

    img1_data = getSARData(img1_path)
    img2_data = getSARData(img2_path)
    sz_img1 = img1_data.shape

    img1_data = img1_data.cuda()
    img2_data = img2_data.cuda()

    img1_ennoise = ennoise1(img1_data)
    img2_ennoise = ennoisegaussian(img2_data)

    model = AEnet(sz_img1[1],Channels,sz_img1[1],Channels).cuda()

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 1000

    model.train()
    for epoch in range(epochs):
        x1_recon,x2_recon = model(img1_ennoise, img2_ennoise,pretraining=True)

        loss = criterion(x1_recon, img1_data) +criterion(x2_recon, img2_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss))

    item = 2
    x1_recon,x2_recon = model(img1_data, img2_data, pretraining=True)


    x1_recon = x1_recon.cpu().detach().numpy()[0, 0, :, :]
    save_img_ch1(x1_recon, 'result/{}/AE_result/recon{}_1.png'.format(dataname, item))


    x2_recon = x2_recon.cpu().detach().numpy()[0,0, :, :]
    save_img_ch1(x2_recon, 'result/{}/AE_result/recon{}_2.png'.format(dataname, item))

    torch.save(model.state_dict(), 'result/{}/AE_result/CP{}_val_{}.pth'.format(dataname, item, '%4f' % loss))


if __name__ == '__main__':
    pretrain()


