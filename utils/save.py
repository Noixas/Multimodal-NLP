import json
import os
from os.path import abspath, dirname, exists, join
import subprocess

import torch

from utils.logger import LOGGER


class ModelSaver():
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def save(self, model, optimizer=None):
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
        dump = {'model_state_dict': state_dict}
        if optimizer is not None:
            dump['optimizer_state_dict'] = optimizer.state_dict()
            if hasattr(optimizer, '_amp_stash'):
                pass  # TODO fp16 optimizer
        torch.save(dump, self.output_dir)