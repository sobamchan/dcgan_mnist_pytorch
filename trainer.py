import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import model


def get_data_loader(args):
    tf = transforms.Compose([transforms.Resize(64),
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307, ),
                                                  (0.3081, ))])
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/',
                                                              train=True,
                                                              download=True,
                                                              transform=tf),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)
    return train_loader


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.data_loader = get_data_loader(args)
        netG = model.netG(args)
        netD = model.netD(args)
        if args.use_cuda:
            netG = netG.cuda()
            netD = netD.cuda()
        self.netG = netG
        self.netD = netD
        self.optimizer_D = optim.Adam(self.netD.parameters(),
                                      lr=args.lr,
                                      betas=(args.beta1, 0.999))
        self.optimizer_G = optim.Adam(self.netG.parameters(),
                                      lr=args.lr,
                                      betas=(args.beta1, 0.999))
        self.criterion = nn.BCELoss()
        self.real_label = 1
        self.fake_label = 0

        fixed_noise_np = np.random.normal(0.0, 1.0,
                                          size=(args.batch_size,
                                                args.nz,
                                                1,
                                                1))
        fixed_noise = torch.from_numpy(fixed_noise_np).type(torch.FloatTensor)
        if args.use_cuda:
            fixed_noise = fixed_noise.cuda()
        self.fixed_noise = Variable(fixed_noise)

    def train_D(self, real_cpu):
        # train with real data
        args = self.args
        self.netD.zero_grad()
        batch_size = real_cpu.size()[0]

        label = torch.FloatTensor(batch_size).fill_(self.real_label)
        if args.use_cuda:
            real_cpu = real_cpu.cuda()
            label = label.cuda()
        x = Variable(real_cpu)
        label = Variable(label)
        output = self.netD(x)
        error_D_real = self.criterion(output, label)
        error_D_real.backward()
        # D_x = output.data.mean()

        # train with fake data
        nz = args.nz
        noise_np = np.random.normal(0.0, 1.0, size=(batch_size, nz, 1, 1))
        noise = torch.from_numpy(noise_np).type(torch.FloatTensor)
        label = torch.FloatTensor(batch_size).fill_(self.fake_label)
        if args.use_cuda:
            noise = noise.cuda()
            label = label.cuda()
        noise = Variable(noise)
        fake = self.netG(noise)
        label = Variable(label)
        output = self.netD(fake.detach())
        error_D_fake = self.criterion(output, label)
        error_D_fake.backward()
        D_G_z1 = output.data.mean()
        error_D = error_D_real + error_D_fake
        self.optimizer_D.step()

        return D_G_z1, error_D.data[0]

    def train_G(self, batch_size):
        args = self.args
        nz = args.nz
        self.netG.zero_grad()
        label = torch.FloatTensor(batch_size).fill_(self.real_label)
        if args.use_cuda:
            label = label.cuda()
        label = Variable(label)
        noise_np = np.random.normal(0.0, 1.0, size=(batch_size, nz, 1, 1))
        noise = torch.from_numpy(noise_np).type(torch.FloatTensor)
        if args.use_cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = self.netG(noise)
        output = self.netD(fake)
        error_D_fake = self.criterion(output, label)
        error_D_fake.backward()
        D_G_z2 = output.data.mean()
        self.optimizer_G.step()

        return D_G_z2, error_D_fake.data[0]

    def train(self):
        args = self.args
        for i_epoch in range(1, args.epoch + 1):
            D_G_z1_list = []
            D_G_z2_list = []
            error_D_list = []
            error_G_list = []
            for i, (data, _) in enumerate(self.data_loader):
                batch_size = data.size()[0]
                D_G_z1, error_D = self.train_D(data)
                D_G_z2, error_G = self.train_G(batch_size)
                D_G_z1_list.append(D_G_z1)
                D_G_z2_list.append(D_G_z2)
                error_D_list.append(error_D)
                error_G_list.append(error_G)
            print('%d th epoch: loss d: %.4f, loss g: %.4f' %
                  (i_epoch,
                   np.mean(error_D_list),
                   np.mean(error_G_list)))

            self.generate(i_epoch)

    def generate(self, i_epoch):
        fake = self.netG(self.fixed_noise)
        vutils.save_image(fake.data,
                          './fake_samples_eppch_%03d.png' % (i_epoch),
                          normalize=True)
