import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class PrintModule(nn.Module):

    def __init__(self):
        super(PrintModule, self).__init__()

    def forward(self, x):
        print(x.size())
        return x


class netG(nn.Module):

    def __init__(self, args):
        super(netG, self).__init__()
        nz = args.nz
        ngf = args.ngf
        nc = args.nc
        self.main = nn.Sequential(nn.ConvTranspose2d(nz,
                                                     ngf * 8,
                                                     4,
                                                     1,
                                                     0,
                                                     bias=False),
                                  nn.BatchNorm2d(ngf * 8),
                                  nn.ReLU(True),

                                  nn.ConvTranspose2d(ngf * 8,
                                                     ngf * 4,
                                                     4,
                                                     2,
                                                     1,
                                                     bias=False),
                                  nn.BatchNorm2d(ngf * 4),
                                  nn.ReLU(True),

                                  nn.ConvTranspose2d(ngf * 4,
                                                     ngf * 2,
                                                     4,
                                                     2,
                                                     1,
                                                     bias=False),
                                  nn.BatchNorm2d(ngf * 2),
                                  nn.ReLU(True),

                                  nn.ConvTranspose2d(ngf * 2,
                                                     ngf,
                                                     4,
                                                     2,
                                                     1,
                                                     bias=False),
                                  nn.BatchNorm2d(ngf),
                                  nn.ReLU(True),

                                  nn.ConvTranspose2d(ngf,
                                                     nc,
                                                     4,
                                                     2,
                                                     1,
                                                     bias=False),
                                  nn.Tanh()
                                  )

        weights_init(self)

    def forward(self, x):
        y = self.main(x)
        return y


class netD(nn.Module):

    def __init__(self, args):
        super(netD, self).__init__()
        ndf = args.ndf
        nc = args.nc
        self.main = nn.Sequential(
                nn.Conv2d(nc,
                          ndf,
                          4,
                          2,
                          1,
                          bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf,
                          ndf * 2,
                          4,
                          2,
                          1,
                          bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2,
                          ndf * 4,
                          4,
                          2,
                          1,
                          bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4,
                          ndf * 8,
                          4,
                          2,
                          1,
                          bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8,
                          1,
                          4,
                          1,
                          0,
                          bias=False),
                nn.Sigmoid()
            )

        weights_init(self)

    def forward(self, x):
        y = self.main(x)
        return y.view(-1, 1).squeeze(1)
