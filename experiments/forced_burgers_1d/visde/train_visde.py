import torch
from torch.utils.data import DataLoader
#from pytorch_lightning.profilers import SimpleProfiler

import pickle as pkl
import os
import pathlib
import shutil

import pytorch_lightning as pl
from pytorch_lightning import loggers
#from pytorch_lightning.callbacks import EarlyStopping

import visde
from experiments.forced_burgers_1d.visde.def_model import create_latent_sde
from datasets.forced_burgers_1d.load_data import load_data

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def get_dataloader(
    n_win: int,
    n_batch: int,
    data_file: str,
    train_val_test: str = "train",
) -> DataLoader:
    data = load_data(data_file)

    data_tensors = visde.MultiEvenlySpacedTensors(
        data[f"{train_val_test}_mu"],
        data[f"{train_val_test}_t"],
        data[f"{train_val_test}_x"],
        data[f"{train_val_test}_f"],
        n_win
    )

    sampler = visde.MultiTemporalSampler(data_tensors, n_batch, n_repeats=1)

    dataloader = DataLoader(
        data_tensors,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=sampler,
        pin_memory=True
    )

    return dataloader

def main(overwrite: bool = False,
        data_file: str = "data_bcf_100_10_10.pkl",
        dim_z = 3,
) -> None:
    n_win = 1
    n_batch_train = 64
    n_batch_val = 256
    print("CUDA:", torch.cuda.is_available())

    train_dataloader = get_dataloader(n_win, n_batch_train, data_file, "train")
    val_dataloader = get_dataloader(n_win, n_batch_val, data_file, "val")

    model = create_latent_sde(dim_z, n_batch_train, n_win, data_file, device)

    version = "_".join([data_file.split(".")[0], str(dim_z), str(n_batch_train), str(n_batch_val)])
    if os.path.exists(os.path.join(CURR_DIR, "logs_visde", version)):
        if overwrite:
            print(f"Version {version} already exists. Overwriting...", flush=True)
            while os.path.exists(os.path.join(CURR_DIR, "logs_visde", version)):
                try:
                    shutil.rmtree(os.path.join(CURR_DIR, "logs_visde", version))
                except OSError as e:
                    continue
        else:
            print(f"Version {version} already exists. Skipping...", flush=True)
            return

    tensorboard = loggers.TensorBoardLogger(CURR_DIR, name="logs_visde", version=version)
    #profiler = SimpleProfiler(dirpath=".", filename="perf_logs")

    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=50,
        logger=tensorboard,
        check_val_every_n_epoch=5,
        #profiler=profiler,
        #callbacks=[EarlyStopping(monitor="val/norm_rmse", mode="min")]
    )
    # ---------------------- training ---------------------- #
    trainer.fit(model, train_dataloader, val_dataloader)
    #print(profiler.summary())

if __name__ == "__main__":
    main()