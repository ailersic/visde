import torch
from torch.utils.data import DataLoader

import pickle as pkl
import os
import pathlib
import pysindy as ps
import matplotlib.pyplot as plt
import numpy as np
import shutil
import time

import pytorch_lightning as pl
from pytorch_lightning import loggers

import visde
from experiments.reaction_diffusion_2d.sindy_2step.def_model import create_sindy_model, create_autoencoder, TestAutoencoder
from datasets.reaction_diffusion_2d.load_data import load_data

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def get_dataloaders(n_win: int,
                    n_batch: int,
                    data_file: str,
) -> tuple[DataLoader, DataLoader]:
    data = load_data(data_file)

    train_data = visde.MultiEvenlySpacedTensors(data["train_mu"], data["train_t"], data["train_x"], data["train_f"], n_win)
    val_data = visde.MultiEvenlySpacedTensors(data["val_mu"], data["val_t"], data["val_x"], data["val_f"], n_win)

    train_sampler = visde.MultiTemporalSampler(train_data, n_batch, n_repeats=1)
    train_dataloader = DataLoader(
        train_data,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=train_sampler,
        pin_memory=True
    )
    val_sampler = visde.MultiTemporalSampler(val_data, n_batch, n_repeats=1)
    val_dataloader = DataLoader(
        val_data,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=val_sampler,
        pin_memory=True
    )

    return train_dataloader, val_dataloader

def train_pod(data_file: str, dim_z: int, overwrite: bool = False):
    data = load_data(data_file)
    snapshots = data["train_x"].flatten(0, 1).flatten(1)
    U, S, VH = torch.linalg.svd(snapshots.T, full_matrices=False)
    modes = U[:, :dim_z]
    latent_snapshots = torch.einsum("ij,jk->ik", [snapshots, modes])
    latent_trajs = latent_snapshots.reshape(data["train_x"].shape[0], data["train_x"].shape[1], dim_z)

    return modes, latent_trajs

def train_autoencoder(data_file: str, dim_z: int, overwrite: bool = False, device: torch.device = device):
    n_win = 1
    n_batch = 128

    train_dataloader, val_dataloader = get_dataloaders(n_win, n_batch, data_file)
    config, encoder, decoder = create_autoencoder(dim_z, n_win, data_file)
    model = TestAutoencoder(config=config, encoder=encoder, decoder=decoder).to(device)

    version = "_".join([data_file.split(".")[0], str(dim_z)])
    if os.path.exists(os.path.join(CURR_DIR, "logs_psindy_ae", version)):
        if overwrite:
            print(f"Version {version} already exists. Overwriting...", flush=True)
            shutil.rmtree(os.path.join(CURR_DIR, "logs_psindy_ae", version))
        else:
            print(f"Version {version} already exists. Skipping...", flush=True)
            ckpt_dir = os.path.join(CURR_DIR, "logs_psindy_ae", version, "checkpoints")
            for file in os.listdir(ckpt_dir):
                if file.endswith(".ckpt"):
                    ckpt_file = file
            
            model = TestAutoencoder.load_from_checkpoint(os.path.join(ckpt_dir, ckpt_file),
                                                        config=config,
                                                        encoder=encoder,
                                                        decoder=decoder).to(device)

            return model

    tensorboard = loggers.TensorBoardLogger(CURR_DIR, name="logs_psindy_ae", version=version)
    #profiler = SimpleProfiler(dirpath=".", filename="perf_logs")

    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=1000,
        logger=tensorboard,
        check_val_every_n_epoch=100,
        #profiler=profiler,
        #callbacks=[EarlyStopping(monitor="val/norm_rmse", mode="min")]
    )
    # ---------------------- training ---------------------- #
    trainer.fit(model, train_dataloader, val_dataloader)
    #print(profiler.summary())

    return model.to(device)

def main(overwrite: bool = False,
        data_file: str = "data.pkl",
        dim_z: int = 2,
        compression: str = "pod", # "pod" or "ae"
        threshold: float = 1e-3,
        degree: int = 3,
) -> None:
    print("CUDA:", torch.cuda.is_available())
    start_time = time.time()

    data = load_data(data_file)

    n_traj = data["train_mu"].shape[0]
    n_tsteps = data["train_x"].shape[1]
    n_win = 1

    dim_mu = data["train_mu"].shape[1]
    dim_f = data["train_f"].shape[1]

    mu = data["train_mu"]
    x = data["train_x"]
    f = data["train_f"]
    t = data["train_t"]
    dt = (data["train_t"][0, 1] - data["train_t"][0, 0]).item()

    # train latent map

    if compression == "pod":
        modes, latent_trajs = train_pod(data_file, dim_z, overwrite)
    elif compression == "ae":
        aenc_model = train_autoencoder(data_file, dim_z, overwrite, device=device)
        latent_trajs = torch.zeros((n_traj, n_tsteps, dim_z), device=device)
        for j in range(n_tsteps):
            with torch.no_grad():
                latent_trajs[:, j], _ = aenc_model.encoder(mu.to(device), x[:, j:(j+n_win)].to(device))

    # train latent sindy model
    
    z_np = latent_trajs.cpu().detach().numpy()

    expanded_mu = mu.unsqueeze(1).expand(-1, n_tsteps, -1)
    u_np = torch.cat((expanded_mu, f), dim=-1).cpu().detach().numpy()

    z = [zi.copy() for zi in z_np]
    u = [ui.copy() for ui in u_np]
    dz = [np.diff(zi, axis=0)/dt for zi in z]

    z = [zi[:-1] for zi in z]
    u = [ui[:-1] for ui in u]

    model = create_sindy_model(z, dz, u, dt, threshold, degree)
    model.print()
    print(model.score(z, u=u, t=dt, x_dot=dz, multiple_trajectories=True))

    version = "_".join([data_file.split(".")[0], str(dim_z), compression, str(threshold), str(degree)])
    version_dir = os.path.join(CURR_DIR, "logs_psindy", version)
    pathlib.Path(version_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(version_dir, "sindy_model.pkl"), "wb") as f:
        pkl.dump(model, f)
    if compression == "pod":
        with open(os.path.join(version_dir, "modes.pkl"), "wb") as f:
            pkl.dump(modes.cpu().detach().numpy(), f)
    
    with open(os.path.join(version_dir, "train_time.txt"), "w") as f:
        f.write(f"{time.time() - start_time:.2f} seconds")
    print("Training time:", time.time() - start_time)

if __name__ == "__main__":
    main()