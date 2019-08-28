"""Framework and utils for network training process. """
import os
import argparse
import shutil
import torch
from torch.utils.data import DataLoader


class ArgParse:
    """Default argparser, please customize it by yourself. """
    def __init__(self, description="base class for network trainer, additional customizations required if necessary"):
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument("-r", "--resume", type=str, default='', help="if specified starts from checkpoint")
        self.parser.add_argument("-t", "--transfer", type=str, default='', help="specify the path of weights for transfer learning")
        self.parser.add_argument("-e", "--epochs", type=int, default=128, help="number of epochs")
        self.parser.add_argument("-b", "--batch-size", type=int, default=64, help="mini-batch size")
        self.parser.add_argument("-n", "--ncpu", type=int, default=8, help="number of cpu threads used during batch generation")
        self.parser.add_argument("-l", "--lr", type=float, default=1e-3, help="learning rate for gradient descent")
        self.parser.add_argument("-p", "--print-freq", type=int, default=10, help="print frequency")
        self.parser.add_argument("-c", "--chkpt-dir", type=str, default="checkpoints/", help="directory saved checkpoints")
        self.parser.add_argument("-i", "--evaluation-interval", type=int, default=1, help="interval between evaluations on validation set")

    def __call__(self):
        return self.parser.parse_args()


class Trainer:
    """Base class for network training,
       its instance variables and functions require implemented while used.
    """
    def __init__(self, train_dataset, val_dataset, model, args=ArgParse()):
        self.__cell = ['net', 'optimizer', 'value', 'epoch']
        self.args = args()

        is_cuda = torch.cuda.is_available()
        self.train_loader = DataLoader(train_dataset, self.args.batch_size, shuffle=True, num_workers=self.args.ncpu, pin_memory=is_cuda)
        self.val_loader = DataLoader(val_dataset, self.args.batch_size, shuffle=False, num_workers=self.args.ncpu, pin_memory=is_cuda)

        self.device = torch.device(["cpu", "cuda:0"][is_cuda])
        self.net = model.to(self.device)
        self.optimizer = torch.optim.Optimizer(self.net.parameters(), defaults={})
        self.criterion = NotImplemented
        self.value = NotImplemented
        self.epoch = 0

    def _setcell(self, cells):
        """Set trainer cells for checkout. """
        self.__cell.extend(cells)

    def state_dict(self):
        """Return dictionary of states for instance varibales listed in self.__cell. """
        return {k: self._get_state(v) for k, v in self.__dict__.items() if k in self.__cell}

    @staticmethod
    def _get_state(obj):
        return obj.state_dict() if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)) else obj

    def load_state_dict(self, checkpoint, strict=True):
        """Resume cells of trainer from checkpoint. """
        for k, v in checkpoint.items():
            if k not in self.__dict__: continue

            if isinstance(self.__dict__[k], torch.nn.Module):
                self.__dict__[k].load_state_dict(v, strict)
            elif isinstance(self.__dict__[k], torch.optim.Optimizer):
                self.__dict__[k].load_state_dict(v)
            else:
                self.__dict__[k] = v

    def main(self):
        """Main cycle of training and validation. """
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("loading checkpoint '{}' ...".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.load_state_dict(checkpoint)
        elif self.args.transfer:
            if os.path.isfile(self.args.transfer):
                print("transfer learning from weights '{}' ...".format(self.args.transfer))
                pre_trained = torch.load(self.args.transfer)
                checkpoint = {'net': self.transfer(pre_trained)}
                self.load_state_dict(checkpoint, strict=False)

        while self.epoch < self.args.epochs:
            self.epoch += 1
            self.train()

            if self.epoch % self.args.evaluation_interval == 0:
                value = self.validate()
                self.checkpoint(value)
                self.value = max(value, self.value)

    def transfer(self, weights):
        """Return weights transferring to aim model. """
        raise NotImplementedError

    def train(self):
        """Self-defined model training. """
        raise NotImplementedError

    def validate(self):
        """Self-defined model validation. """
        raise NotImplementedError

    def checkpoint(self, value):
        """Save checkpoint for the training process. """
        if not os.path.exists(self.args.chkpt_dir):
            os.mkdir(self.args.chkpt_dir)
        save_pth = os.path.join(self.args.chkpt_dir, "current.pth.tar")

        torch.save(self.state_dict(), save_pth)
        if value > self.value:
            shutil.copyfile(save_pth, os.path.join(self.args.chkpt_dir, "best.pth.tar"))
            print("upgrade model successfully!")
