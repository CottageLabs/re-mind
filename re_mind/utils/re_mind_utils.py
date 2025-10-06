import os
import torch


def get_global_device():
    env_device = os.environ.get('LMT_DEVICE')
    if env_device:
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_device(device: str):
    os.environ['LMT_DEVICE'] = device
    return device



def get_sys_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
