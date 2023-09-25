import argparse, sys
import lightning.pytorch as pl
import torch

sys.path.insert(1, sys.path[0] + '/..')
from src.data.datamodule import DataModule
from src.model.setup import setup_model

def parse_option(notebook = False):
    parser = argparse.ArgumentParser(description="RibFrac")

    # Model
    parser.add_argument('--net', type=str, default='unet3d', help='Network architecture')

    # Training 
    parser.add_argument('--max_epochs', type=int, default=3, help='Max number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    # Logging
    parser.add_argument('--ckpt_path', type=str, default='test', help='Path to save checkpoint')
    parser.add_argument('--version', default='version_0', help='Version of the model to load')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    return args

def main(args):
    pl.seed_everything(args.seed, workers=True)

    datamodule = DataModule(dir = '../data_dev', num_workers=args.num_workers, batch_size=args.batch_size)
    model = setup_model(net = args.net)

    trainer = pl.Trainer(
        logger = pl.loggers.TensorBoardLogger('../logs', name = 'test', version = args.version),
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
        callbacks = [
                pl.callbacks.TQDMProgressBar(refresh_rate = 1000)
                ],
        deterministic = True,
    )

    trainer.fit(model,  datamodule=datamodule)

if __name__ == "__main__":
    args = parse_option()
    print(args)
    main(args)