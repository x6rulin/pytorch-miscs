"""Miscellaneous utility classes and functions. """

import sys

from typing import Any


class Logger:
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing
       on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = 'w', should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Writing text to stdout (and a file) and optionally flush. """
        if len(text) == 0:
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open. """
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring. """
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


class EMA:
    """Weighted Moving Average. """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.back_up = {}

        self.register_shadow()

    def register_shadow(self):
        """Registers model's parameters into shadow. """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow from model's parameters. """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def state_dict(self):
        """Returns a dictionary containing a whole state of the EMA weights. """
        return self.shadow

    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from :attr:`state_dict` into shadow. """
        for name, data in state_dict.items():
            assert name in self.shadow
            self.shadow[name] = data

    def apply_shadow(self):
        """Loads shadow to model. """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.back_up[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restores model's parameters. """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.back_up
                param.data = self.back_up[name]
        self.back_up = {}
