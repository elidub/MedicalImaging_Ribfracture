import os
import sys
import torch
import pickle
import shutil
import argparse
import numpy as np
import lightning.pytorch as pl

from matplotlib import pyplot as plt

sys.path.insert(1, sys.path[0] + "/..")
from src.data.datamodule import DataModule
from src.model.setup import setup_model
from src.misc.utils import set_seed_and_precision


def find_npz_files(directory):
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(root, file))
    return sorted(npz_files)


def parse_option(notebook=False):
    parser = argparse.ArgumentParser(description="RibFrac")

    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--predict", action="store_true", help="Predict model")

    # Model
    parser.add_argument(
        "--net", type=str, default="unet3d", help="Network architecture"
    )
    parser.add_argument(
        "--data_dir", type=str, default="../data", help="Path to data directory"
    )
    parser.add_argument("--dataset", type=str, default="boxes", help="Dataset type")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to use",
    )

    # Training
    parser.add_argument(
        "--max_epochs", type=int, default=3, help="Max number of training epochs"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for dataloader"
    )
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size")

    # Logging
    parser.add_argument(
        "--log_dir", type=str, default="../logs", help="Path to save logs"
    )
    parser.add_argument(
        "--version", default="version_0", help="Version of the model to load"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    return args


def main(args):
    assert args.train is not args.predict, "Must train or predict"
    set_seed_and_precision(args)

    datamodule = DataModule(
        dir=args.data_dir,
        dataset=args.dataset,
        splits=args.splits,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    model = setup_model(args)

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            args.log_dir, name=args.net, version=args.version
        ),
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=1000)],
        deterministic=False,  # Set to False for max_pool3d_with_indices_backward_cuda
    )

    if args.train:
        trainer.fit(model, datamodule=datamodule)
    elif args.predict:
        for split in args.splits:
            dataloader = f"{split}_dataloader"
            if split == "train":
                dataloader = f"{split}_no_shuffle_dataloader"
            datamodule.predict_dataloader = getattr(
                datamodule, dataloader, "predict_dataloader"
            )

            preds = trainer.predict(
                model, datamodule=datamodule
            )  # i think this is a (list [#batches], tuple [prediction ???], tensor [batchsize, x, y, z])

            pred_dir = os.path.join(args.log_dir, args.net, args.version)

            if args.net == "unet3d":
                data_dir = os.path.join(pred_dir, "segmentations", split)
                if os.path.exists(data_dir):
                    shutil.rmtree(data_dir)
                os.makedirs(data_dir)

                # Recreate directory structure and dumy npz files
                for img_id in sorted(
                    os.listdir(os.path.join(args.data_dir, "boxes", split, "images"))
                ):
                    os.makedirs(os.path.join(data_dir, img_id))
                    for patch in sorted(
                        os.listdir(
                            os.path.join(
                                args.data_dir, "boxes", split, "images", img_id
                            )
                        )
                    ):
                        if patch != "metadata.csv":
                            os.makedirs(os.path.join(data_dir, img_id, patch))
                            num_boxes_in_patch = len(
                                sorted(
                                    os.listdir(
                                        os.path.join(
                                            args.data_dir,
                                            "boxes",
                                            split,
                                            "images",
                                            img_id,
                                            patch,
                                        )
                                    )
                                )
                            )
                            for i in range(num_boxes_in_patch):
                                np.savez_compressed(
                                    os.path.join(
                                        data_dir, img_id, patch, f"box{i}.npz"
                                    ),
                                    np.zeros(1),
                                )
                        # copy metadata.csv
                        else:
                            shutil.copy(
                                os.path.join(
                                    args.data_dir,
                                    "boxes",
                                    split,
                                    "images",
                                    img_id,
                                    patch,
                                ),
                                os.path.join(data_dir, img_id, patch),
                            )

                # Save predictions to npz files
                npz_files = find_npz_files(
                    os.path.join(data_dir)
                )
                for batch in preds:
                    batch_segs, _ = batch
                    for seg in batch_segs:
                        path = npz_files.pop(0)
                        np.savez_compressed(path, seg)

            elif args.net == "retinanet":
                data_dir = os.path.join(pred_dir, "boxes", split)
                if os.path.exists(data_dir):
                    shutil.rmtree(data_dir)
                os.makedirs(data_dir)

                for batch in preds:
                    batch_boxes, batch_infos = batch
                    for boxes, info in zip(batch_boxes, batch_infos):
                        patch = f"patch{info['patch']}"
                        patch_dir = os.path.join(
                            data_dir, "images", info["file"], patch
                        )
                        label_dir = os.path.join(
                            data_dir, "labels", info["file"], patch
                        )
                        os.makedirs(patch_dir)
                        os.makedirs(label_dir)

                        file = f"{info['file']}.npz"
                        patch_img = np.load(
                            os.path.join(
                                args.data_dir, "patches", split, "images", file
                            )
                        )[info["patch"]]

                        if split != "test":
                            patch_lab = np.load(
                                os.path.join(
                                    args.data_dir, "patches", split, "labels", file
                                )
                            )[info["patch"]]

                        for box in boxes:
                            res_img = patch_img[
                                box[0] : box[0] + box[3],
                                box[1] : box[1] + box[4],
                                box[2] : box[2] + box[5],
                            ]

                            if split != "test":
                                res_lab = patch_lab[
                                    box[0] : box[0] + box[3],
                                    box[1] : box[1] + box[4],
                                    box[2] : box[2] + box[5],
                                ]

                            box_id = len(os.listdir(patch_dir))
                            np.savez_compressed(
                                os.path.join(patch_dir, f"box{box_id}.npz"), res_img
                            )
                            if split != "test":
                                np.savez_compressed(
                                    os.path.join(label_dir, f"box{box_id}.npz"), res_lab
                                )


if __name__ == "__main__":
    args = parse_option()
    print(args)
    main(args)
