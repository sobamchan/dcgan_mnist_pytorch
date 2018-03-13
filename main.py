import torch
import argparse
from trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--use-cuda', type=bool, default=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
