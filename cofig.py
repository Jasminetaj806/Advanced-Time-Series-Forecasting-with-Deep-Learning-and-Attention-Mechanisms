import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 5        # total features including target
TARGET_DIM = 1
ENC_HIDDEN = 64
DEC_HIDDEN = 64
HORIZON = 10