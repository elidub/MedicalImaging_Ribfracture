import lightning.pytorch as pl
import torch

def set_seed_and_precision(args):
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('medium')