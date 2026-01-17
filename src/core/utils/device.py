import torch

DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")


def use_mps():
    global DTYPE, DEVICE
    DTYPE = torch.float16
    DEVICE = torch.device("mps")
