import torch
import torch.nn as nn


def paramsInit(net):
    if isinstance(net, nn.Conv2d):
        nn.init.xavier_uniform_(net.weight.data)
        nn.init.constant_(net.bias.data, 0.01)
    if isinstance(net, nn.Linear):
        nn.init.xavier_uniform_(net.weight.data)
        nn.init.constant_(net.bias.data, 0.01)

class Encoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()

        self.forward_pass1 = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        paramsInit(self)

    def forward(self, x):
        F = self.forward_pass1(x)
        return F

class Feature(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Feature, self).__init__()
        self.forward_pass1 = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1,groups=output_size),
            nn.GELU(),
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1,groups=output_size),
            nn.GELU(),
            nn.Conv2d(output_size, output_size, kernel_size=1, padding=0),
        )
        paramsInit(self)
    
    def forward(self, x):
        F = self.forward_pass1(x)
        return F

class clNet(nn.Module):
    def __init__(self, input_size,output_size):
        super(clNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, input_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(input_size, input_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(input_size, input_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(input_size, output_size, kernel_size=1),
            nn.Sigmoid()
        )
        paramsInit(self)

    def forward(self, x):
        cov1 = self.conv1(x)
        return cov1


class Decoder(nn.Module):

    def __init__(self, output_size,input_size):
        super(Decoder, self).__init__()
        self.backward_pass2 = nn.Sequential(
            nn.ConvTranspose2d(output_size, output_size, kernel_size=3, padding=1),
            nn.LeakyReLU( ),
            nn.ConvTranspose2d(output_size, output_size, kernel_size=3, padding=1),
            nn.LeakyReLU( ),
            nn.ConvTranspose2d(output_size, output_size, kernel_size=3, padding=1),
            nn.LeakyReLU( ),
            nn.ConvTranspose2d(output_size, input_size, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        df2 = self.backward_pass2(x)
        return df2


class MCAE(nn.Module):
    def __init__(self, in_ch1, out_ch1, in_ch2, out_ch2):  # c, 20, c, 20
        super(MCAE, self).__init__()
        self.enc_x1 = Encoder(in_ch1, out_ch1)
        self.enc_x2 = Encoder(in_ch2, out_ch2)

        self.dec_x1 = Decoder(out_ch1,in_ch1)
        self.dec_x2 = Decoder(out_ch2, in_ch2)
        
        self.Feature=Feature(out_ch1, out_ch1)

        paramsInit(self)

    def forward(self, x1, x2):

        F1 = self.enc_x1(x1)
        F2= self.enc_x2(x2)
        
        F1=self.Feature(F1)
        F2=self.Feature(F2)
        
        x1_recon = self.dec_x1(F1)
        x2_recon = self.dec_x2(F2)
        return F1, F2, x1_recon, x2_recon

#
# if __name__ == '__main__':
#     x = torch.randn(1,20,235,233)
#     net = clNet(20,1)
#     y= net(x)
#     print(y.shape)