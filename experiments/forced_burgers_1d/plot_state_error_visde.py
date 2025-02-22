import torch
from torch import Tensor
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
import numpy as np

import visde
from experiments.forced_burgers_1d.def_model import create_latent_sde

plt.rcParams.update({'font.size': 14})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
OUT_DIR = os.path.join(CURR_DIR, "postproc_visde")
N_TRAIN = 200
DATA_FILE = f"data_ll_{N_TRAIN}_{int(N_TRAIN*0.2)}_50.pkl"

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(train_val_test: str = "test"):
    with open(os.path.join(CURR_DIR, DATA_FILE), "rb") as f:
        data = pkl.load(f)
    
    mu = data[f"{train_val_test}_mu"].to(device)
    t = data[f"{train_val_test}_t"].to(device)
    x = data[f"{train_val_test}_x"].to(device)
    f = data[f"{train_val_test}_f"].to(device)

    dim_x = x.shape[-1]
    dim_z = 9
    i_traj = 0
    n_win = 1
    n_batch = 32
    n_dec_samp = 64
    n_tsteps = t.shape[1]

    abs_err = np.zeros((dim_x, n_tsteps))
    x_pred = np.zeros((dim_x, n_tsteps))
    x_pred_std = np.zeros((dim_x, n_tsteps))

    sde_options = {
        'method': 'srk',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-4,
        'atol': 1e-6
    }

    dummy_model = create_latent_sde(dim_z, n_batch, n_win, DATA_FILE)
    model = visde.LatentSDE.load_from_checkpoint(f"experiments/forced_burgers_1d/logs_visde/version_ref_{N_TRAIN}_ll_2/checkpoints/epoch=49-step={400*N_TRAIN}.ckpt",
                                                 config=dummy_model.config,
                                                 encoder=dummy_model.encoder,
                                                 decoder=dummy_model.decoder,
                                                 drift=dummy_model.drift,
                                                 dispersion=dummy_model.dispersion,
                                                 loglikelihood=dummy_model.loglikelihood,
                                                 latentvar=dummy_model.latentvar).to(device)
    model.eval()
    model.encoder.resample_params()
    model.decoder.resample_params()
    model.drift.resample_params()
    model.dispersion.resample_params()

    print([p for p in model.dispersion.parameters()])

    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    print(f"Integrating SDE for trajectory {train_val_test} {i_traj}...", flush=True)

    mu_i = mu[i_traj].unsqueeze(0)
    mu_i_batch = mu_i.repeat((n_batch, 1))
    t_i = t[i_traj]
    x0_i = x[i_traj, :n_win, :].unsqueeze(0)
    f_i = f[i_traj]

    z0_i = model.encoder.sample(n_batch, mu_i, x0_i)
    sde = visde.sde.SDE(model.drift, model.dispersion, mu_i, t_i, f_i)
    with torch.no_grad():
        zs = torchsde.sdeint(sde, z0_i, t_i, **sde_options)
    print("done", flush=True)

    assert isinstance(zs, Tensor), "zs is expected to be a single tensor"

    print("Decoding trajectory...", flush=True)
    for j_t in range(n_tsteps):
        if j_t % 100 == 0:
            print(f"{j_t}...", flush=True)

        xs = model.decoder.sample(n_dec_samp, mu_i_batch, zs[j_t]).detach()
        x_mean = xs.mean(dim=0)
        x_std = xs.std(dim=0)
        x_err = x_mean - x[i_traj, j_t]

        x_pred[:, j_t] = x_mean.cpu().detach().numpy()
        x_pred_std[:, j_t] = x_std.cpu().detach().numpy()
        abs_err[:, j_t] = x_err.abs().cpu().detach().numpy()
    print("done", flush=True)

    fig, axs = plt.subplots(1, 4, figsize=(6.3, 3), layout='constrained')
    
    # state
    plot0 = axs[0].pcolorfast((-3, 3), (0, 1), x[i_traj].cpu().numpy(), cmap='coolwarm', vmin=0, vmax=1.5)
    axs[0].set_title("True\nSolution")

    axs[0].set_ylabel(r"Time $t$")
    axs[0].set_ylim([0, 1])
    axs[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    axs[0].set_xlabel(r"$x$")
    axs[0].set_xlim([-3, 3])
    axs[0].set_xticks([-3, 0, 3])
    
    # prediction
    plot1 = axs[1].pcolorfast((-3, 3), (0, 1), x_pred.T, cmap='coolwarm', vmin=0, vmax=1.5)
    axs[1].set_title("Prediction\nMean")

    #axs[1].set_ylabel(r"$t$")
    axs[1].set_ylim([0, 1])
    axs[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[1].set_yticklabels([])

    axs[1].set_xlabel(r"$x$")
    axs[1].set_xlim([-3, 3])
    axs[1].set_xticks([-3, 0, 3])

    fig.colorbar(plot1, ax=axs[[0, 1]], ticks=[0, 0.5, 1, 1.5], location='bottom')#, label=r"QoI $u$")

    # error
    plot2 = axs[2].pcolorfast((-3, 3), (0, 1), abs_err.T, cmap='afmhot', vmin=0, vmax=0.1)
    axs[2].set_title("Absolute\nError")

    #axs[2].set_ylabel(r"$t$")
    axs[2].set_ylim([0, 1])
    axs[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[2].set_yticklabels([])

    axs[2].set_xlabel(r"$x$")
    axs[2].set_xlim([-3, 3])
    axs[2].set_xticks([-3, 0, 3])

    # standard deviation
    plot3 = axs[3].pcolorfast((-3, 3), (0, 1), x_pred_std.T, cmap='afmhot', vmin=0, vmax=0.1)
    axs[3].set_title("Prediction\nStd. Dev.")

    #axs[3].set_ylabel(r"$t$")
    axs[3].set_ylim([0, 1])
    axs[3].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[3].set_yticklabels([])

    axs[3].set_xlabel(r"$x$")
    axs[3].set_xlim([-3, 3])
    axs[3].set_xticks([-3, 0, 3])

    fig.colorbar(plot3, ax=axs[[2, 3]], ticks=[0, 0.05, 0.1], location='bottom')

    #fig.tight_layout()

    fig.savefig(os.path.join(OUT_DIR, f"{train_val_test}_{N_TRAIN}_abs_error.pdf"), format='pdf')
    fig.show()

if __name__ == "__main__":
    #main("train")
    main("test")