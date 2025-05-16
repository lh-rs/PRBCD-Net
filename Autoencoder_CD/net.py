import torch
import torch.nn as nn


class Coupling(nn.Module):
    def __init__(self, in_channels, out_channels):  # c, Channels
        super(Coupling, self).__init__()
        self.coupling = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.coupling(x)


class deCoupling(nn.Module):
    def __init__(self, in_channels, out_channels):  # 20, Channels
        super(deCoupling, self).__init__()
        self.decoupling = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoupling(x)


class AEnet(nn.Module):
    def __init__(self, in_ch1, out_ch1, in_ch2, out_ch2):  # c, Channels, c, Channels
        super(AEnet, self).__init__()
        #
        # for p in self.parameters():
        #     p.requires_grad = False
        self.enc_x1 = Coupling(in_ch1, out_ch1)
        self.enc_x2 = Coupling(in_ch2, out_ch2)

        self.dec_x1 = deCoupling(out_ch1, in_ch1)
        self.dec_x2 = deCoupling(out_ch2, in_ch2)

    def forward(self, x1, x2, pretraining=False):
        x1_fea = self.enc_x1(x1)
        x2_fea = self.enc_x2(x2)

        if pretraining:
            x1_recon = self.dec_x1(x1_fea)
            x2_recon = self.dec_x2(x2_fea)
            return x1_recon, x2_recon
        else:
            return x1_fea, x2_fea


# if __name__ == '__main__':
#     model = AEnet(1, 20, 1, 20)
#     x1=torch.randn(1,1, 811,921)
#     x2 = torch.randn(1, 1, 811,921)
#     v1,v2=model(x1, x2, pretraining=False)
#     print(v1.shape)