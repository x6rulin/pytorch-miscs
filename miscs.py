"""framework and utils for network training process. """
import os
import argparse
import shutil
import torch
from torch.utils.data import DataLoader


def argparser():
    """default argparse, please customize it by yourself. """
    parser = argparse.ArgumentParser(description="base class for network training")
    parser.add_argument("-r", "--resume", type=str, default='', help="if specified starts from checkpoint")
    parser.add_argument("-t", "--transfer", type=str, default='', help="specify the path of weights for transfer learning")
    parser.add_argument("-e", "--epochs", type=int, default=128, help="number of epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="mini-batch size")
    parser.add_argument("-n", "--ncpu", type=int, default=8, help="number of cpu threads used during batch generation")
    parser.add_argument("-l", "--lr", type=float, default=1e-3, help="learning rate for gradient descent")
    parser.add_argument("-p", "--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("-c", "--chkpt-dir", type=str, default="checkpoints/", help="directory saved checkpoints")
    parser.add_argument("-i", "--evaluation-interval", type=int, default=1, help="interval between evaluations on validation set")

    return parser.parse_args()


class Trainer:
    """base class for network training, its instance variables and functions requires implemented while used. """
    def __init__(self, train_dataset, val_dataset, model, args=argparser):
        self.args = args()
        if not os.path.exists(self.args.chkpt_dir):
            os.mkdir(self.args.chkpt_dir)

        is_cuda = torch.cuda.is_available()
        self.train_loader = DataLoader(train_dataset, self.args.batch_size, shuffle=True, num_workers=self.args.ncpu, pin_memory=is_cuda)
        self.val_loader = DataLoader(val_dataset, self.args.batch_size, shuffle=False, num_workers=self.args.ncpu, pin_memory=is_cuda)

        self.device = torch.device(["cpu", "cuda:0"][is_cuda])
        self.net = model().to(self.device)
        self.optimizer = torch.optim.Optimizer(self.net.parameters(), defaults={})
        self.loss_fn = NotImplemented
        self.value = NotImplemented
        self.epoch = 0

    def main(self):
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("loading chekpoint '{}' ...".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.net.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epoch = checkpoint['epoch']
                self.value = checkpoint['rate']
        elif self.args.transfer:
            if os.path.isfile(self.args.transfer):
                print("tranfer learning from weights '{}' ...".format(self.args.transfer))
                weights = torch.load(self.args.transfer)
                self.transfer(weights)

        while self.epoch < self.args.epochs:
            self.epoch += 1
            self.train()

            if self.epoch % self.args.evaluation_interval == 0:
                rate = self.validate()
                self.checkpoint({'state_dict': self.net.state_dict(),
                                  'optimizer': self.optimizer.state_dict(),
                                  'epoch': self.epoch,
                                  'rate': rate},
                                 rate > self.value)
                self.value = max(rate, self.value)

    def transfer(self, weights):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def checkpoint(self, state, is_best):
        save_pth = os.path.join(self.args.chkpt_dir, "current.pth.tar")
        torch.save(state, save_pth)
        if is_best:
            shutil.copyfile(save_pth, os.path.join(self.args.chkpt_dir, "best.pth.tar"))
