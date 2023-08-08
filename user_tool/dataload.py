import torch
import cv2
import numpy as np
from user_tool.utils import save_img_ch1

def data_Norm(data):
    # data = data.detach().cpu().numpy()
    min = np.amin(data)
    max = np.amax(data)
    result = (data - min) / (max - min) #torch.from_numpy()
    return result

def getSARData(img_path):

    img_data = np.expand_dims(data_Norm(cv2.imread(img_path)[..., 0]), -1)  # w*h*1
    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # 1*w*h tensor #permute换顺序
    img_data = img_data.unsqueeze(0)  # 1*1*w*h tensor
    print(img_data.shape)
    return img_data
def getSARData0(img_path):

    img_data = np.expand_dims(cv2.imread(img_path)[..., 0], -1)  # w*h*1
    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # 1*w*h tensor #permute换顺序
    img_data = img_data.unsqueeze(0)  # 1*1*w*h tensor
    print(img_data.shape)
    return img_data
def getRData(img_path):
    img_data =cv2.imread(img_path)   # w*h*c
    img_data= cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
    img_data = np.expand_dims(data_Norm(img_data), -1)
    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # c*w*h tensor
    img_data = img_data.unsqueeze(0)  # 1*c*w*h tensor

    return img_data
def getMSIData(img_data):
    img_data = img_data/255 # w*h*c
    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # c*w*h tensor
    img_data = img_data.unsqueeze(0)  # 1*c*w*h tensor
    print(img_data.shape)
    return img_data
def getRGBData(img_path):

    img_data = cv2.imread(img_path)/255 # w*h*c
    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # c*w*h tensor
    img_data = img_data.unsqueeze(0)  # 1*c*w*h tensor
    print(img_data.shape)
    return img_data
def getRGBData0(img_path):

    img_data = cv2.imread(img_path)# w*h*c
    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # c*w*h tensor
    img_data = img_data.unsqueeze(0)  # 1*c*w*h tensor
    print(img_data.shape)
    return img_data

def getP0Data(p0_path):
    p0_data = np.expand_dims(data_Norm(cv2.imread(p0_path))[..., 0], -1)  # w*h*1
    p0_data = torch.FloatTensor(p0_data).permute(2, 0, 1)
    return p0_data

def getP02Data(p0_path):
    p0_data = np.expand_dims(cv2.imread(p0_path)[..., 0], -1)  # w*h*1
    p0_data = torch.FloatTensor(p0_data).permute(2, 0, 1)
    return p0_data

def ennoisesar(x):
    sz_x = x.shape
    noise = torch.randn(sz_x)
    x = x.cpu()
    x += x * noise
    x = torch.where(torch.BoolTensor(x > 1), torch.ones(sz_x), x)
    x = torch.where(torch.BoolTensor(x < -1), -torch.ones(sz_x), x)
    return x.detach().cuda()

def ennoisegaussian(x,var=0.001):
    sz_x = x.shape
    x = x.cpu()
    noise = torch.randn(sz_x)
    nosiy = x + (var**0.5)*noise

    return nosiy.detach().cuda()

def ennoise1(x):
    sz_x = x.shape
    noise = torch.randn(sz_x)
    x = x.cpu()
    x += x * noise
    x = torch.where(torch.BoolTensor(x > 1), torch.ones(sz_x), x)
    x = torch.where(torch.BoolTensor(x < -1), -torch.ones(sz_x), x)
    return x.detach().cuda()



def liu_showresult(data,norm=False):
    data = np.expand_dims(data, -1)
    if norm:
        data_ = np.reshape(data, [-1])
        data_max = np.max(data_)
        data_min = np.min(data_)
        data = (data - data_min) / (data_max - data_min)
    result = np.concatenate([data, data, data], -1)
    return result

