import torch
from torch import Tensor
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
import numpy as np

import visde
from experiments.flowcontrol_2d.def_model import create_latent_sde

plt.rcParams.update({'font.size': 14})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
OUT_DIR = os.path.join(CURR_DIR, "postproc_visde")

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(train_val_test: str = "test"):
    with open(os.path.join(CURR_DIR, "data.pkl"), "rb") as f:
        data = pkl.load(f)
    
    mu = data[f"{train_val_test}_mu"].to(device)
    t = data[f"{train_val_test}_t"].to(device)
    x = data[f"{train_val_test}_x"].to(device)
    f = data[f"{train_val_test}_f"].to(device)

    dim_z = 9
    n_traj = mu.shape[0]
    n_win = 1
    n_batch = 32
    n_dec_samp = 64
    n_tsteps = t.shape[1]

    rmsre = np.zeros(n_traj)
    rel_err = np.zeros((n_traj, n_tsteps))

    sde_options = {
        'method': 'srk',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-4,
        'atol': 1e-6
    }

    dummy_model = create_latent_sde(dim_z, n_batch, n_win)
    model = visde.LatentSDE.load_from_checkpoint("experiments/flowcontrol_2d/logs_visde/version_ref_4/checkpoints/epoch=499-step=150000.ckpt",
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
    
    model.dispersion.disp = torch.exp(model.dispersion.logdisp_mean)
    model.decoder.dec_var = torch.exp(model.decoder.dec_logvar_mean)

    print([p for p in model.dispersion.parameters()])

    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    for i_traj in range(n_traj):
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

        sqerr = np.zeros(n_tsteps)
        norm_sqerr = np.zeros(n_tsteps)

        print("Decoding trajectory...", flush=True)
        for j_t in range(n_tsteps):
            if j_t % 100 == 0:
                print(f"{j_t}...", flush=True)

            xs = model.decoder.sample(n_dec_samp, mu_i_batch, zs[j_t]).detach()
            x_mean = xs.mean(dim=0)
            x_err = x_mean - x[i_traj, j_t]

            sqerr[j_t] = x_err.pow(2).sum().item()
            norm_sqerr[j_t] = sqerr[j_t] / x[i_traj, j_t].pow(2).sum().item()
            rel_err[i_traj, j_t] = np.sqrt(norm_sqerr[j_t])
        print("done", flush=True)

        rmsre[i_traj] = np.sqrt(np.mean(norm_sqerr))
        
        print(f"Mean Normalized RMSE: {rmsre[i_traj]}", flush=True)

    figrmse, ax = plt.subplots(figsize=(12, 3))

    periods = [0, 66, 133, 200, 266, 333]
    t_plot = t.cpu().detach().numpy()[0, :]

    ax.plot(t_plot, np.mean(rel_err, axis=0))
    ax.boxplot(rel_err[:, periods], positions=t_plot[periods], showfliers=False)
    ax.set_xticks(t_plot[periods], [r"$0$", r"$T_v$", r"$2T_v$", r"$3T_v$", r"$4T_v$", r"$5T_v$"])
    ax.set_xlabel("Time")
    ax.set_xlim([0.1667, 2.1667])
    ax.set_ylabel(r"$\|\bar{u}(t) - u(t)\|/\|u(t)\|$")
    ax.grid()
    figrmse.tight_layout()
    figrmse.savefig(os.path.join(OUT_DIR, f"{train_val_test}_rmsre_traj_mean.pdf"), format='pdf')
    figrmse.show()

    print(f"RMSRE Mean: {np.mean(rmsre)}, Std Dev: {np.std(rmsre)}", flush=True)

if __name__ == "__main__":
    #main("train")
    main("test")