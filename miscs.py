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
        self.parser.add_argument("-c", "--chkpt-dir", type=str, default="checkpoints/", help="directory saving checkpoints")
        self.parser.add_argument("-i", "--evaluation-interval", type=int, default=1, help="interval between evaluations on validation set")

    def __call__(self):
        return self.parser.parse_args()


class Trainer:
    """Base class for network training,
       its instance variables and functions require implemented while used.

       :p for multiple models training, that's suggested packing the models and their
          optimizers by dictionarys.
    """
    def __init__(self, train_dataset, val_dataset=None, args=ArgParse()):
        self.__cell = ['net', 'optimizer', 'value', 'epoch']
        self.args = args()

        is_cuda = torch.cuda.is_available()
        self.train_loader = DataLoader(train_dataset, self.args.batch_size, shuffle=True, num_workers=self.args.ncpu, pin_memory=is_cuda)
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, self.args.batch_size, shuffle=False, num_workers=self.args.ncpu, pin_memory=is_cuda)

        self.device = torch.device(["cpu", "cuda:0"][is_cuda])
        self.net = NotImplemented
        self.optimizer = NotImplemented
        self.criterion = NotImplemented
        self.value = NotImplemented
        self.epoch = 0

    def _appendcell(self, cells):
        """Set trainer cells for checkout. """
        self.__cell.extend(cells)

    def state_dict(self):
        """Return dictionary of states for instance varibales listed in self.__cell. """
        return {k: self._get_state(self.__dict__[k]) for k in self.__cell}

    def load_state_dict(self, checkpoint, strict=True):
        """Resume cells of trainer from checkpoint. """
        self._load_state(self.__dict__, checkpoint, strict)

    @staticmethod
    def _get_state(obj):
        if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
            _state = obj.state_dict()
        elif isinstance(obj, dict):
            _state = {k: Trainer._get_state(obj[k]) for k in obj.keys()}
        else:
            _state = obj

        return _state

    @staticmethod
    def _load_state(_dict, checkpoint, strict=True):
        for k, v in checkpoint.items():
            if k not in _dict: continue

            if isinstance(_dict[k], torch.nn.Module):
                _dict[k].load_state_dict(v, strict)
            elif isinstance(_dict[k], torch.optim.Optimizer):
                _dict[k].load_state_dict(v)
            elif isinstance(_dict[k], dict):
                Trainer._load_state(_dict[k], v, strict)
            else:
                _dict[k] = v

    def __call__(self):
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
        """Return (dict of) weights transferring to aim model(s). """
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
            os.makedirs(self.args.chkpt_dir, 0o775)
        save_pth = os.path.join(self.args.chkpt_dir, "current.pth.tar")

        torch.save(self.state_dict(), save_pth)
        if value > self.value:
            shutil.copyfile(save_pth, os.path.join(self.args.chkpt_dir, "best.pth.tar"))
            print("upgrade model successfully!")
