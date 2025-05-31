import torch
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

import visde
from experiments.forced_burgers_1d.visde.def_model import create_latent_sde
from datasets.forced_burgers_1d.load_data import load_data

plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(data_file: str = "data_bcf_100_10_10.pkl", dim_z: int = 5):
    data = load_data(data_file)
    
    mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    dim_x = x.shape[-1]
    #n_traj = mu.shape[0]
    n_win = 1
    n_batch = 32
    n_batch_decoder = 64
    n_tsteps = t.shape[1]
    i_traj = 2

    sde_options = {
        'method': 'euler',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-3,
        'atol': 1e-5
    }

    dummy_model = create_latent_sde(dim_z, n_batch, n_win, data_file)
    version = "_".join([data_file.split(".")[0], str(dim_z), "64", "256"])
    ckpt_dir = os.path.join(CURR_DIR, "logs_visde", version, "checkpoints")
    out_dir = os.path.join(CURR_DIR, "postproc_visde", version)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(ckpt_dir):
        if file.endswith(".ckpt"):
            ckpt_file = file
    
    model = visde.LatentSDE.load_from_checkpoint(os.path.join(ckpt_dir, ckpt_file),
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
    
    model.decoder.dec_var = model.decoder.dec_logvar_mean.exp()
    
    traj_dir = os.path.join(out_dir, f"{TRAIN_VAL_TEST}_traj_{i_traj}")
    pathlib.Path(traj_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 4))
    axgrid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 4),
                    axes_pad=0.20,
                    share_all=True,
                    direction="column"
                    )

    print(f"Integrating SDE for trajectory {TRAIN_VAL_TEST} {i_traj}...", flush=True)

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
    
    def update(j):
        if j % 10 == 0:
            print(f"Frame {j}/{n_tsteps}", flush=True)
        # j is time index
        xs = model.decoder.sample(n_batch_decoder, mu_i_batch, zs[j]).detach()
        x_mean = xs.mean(dim=0).cpu().detach().numpy()
        x_std = xs.std(dim=0).cpu().detach().numpy()
        x_true = x[i_traj].cpu().detach().numpy()

        axgrid[0].cla()
        axgrid[0].plot(np.linspace(0, 5, dim_x), x_true[j], color="blue")
        axgrid[0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        axgrid[0].set_ylim(-0.2, 5.2)
        
        axgrid[1].cla()
        axgrid[1].plot(np.linspace(0, 5, dim_x), x_mean, color="blue")
        axgrid[1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        axgrid[1].set_ylim(-0.2, 5.2)
        
        axgrid[2].cla()
        axgrid[2].plot(np.linspace(0, 5, dim_x), np.abs(x_mean - x_true[j]), color="red")
        axgrid[2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        axgrid[2].set_ylim(-0.2, 5.2)

        axgrid[3].cla()
        axgrid[3].plot(np.linspace(0, 5, dim_x), x_std, color="red")
        axgrid[3].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        axgrid[3].set_ylim(-0.2, 5.2)

        axgrid[0].set_title("True Solution")
        axgrid[1].set_title("Prediction Mean")
        axgrid[2].set_title("Absolute Error")
        axgrid[3].set_title("Prediction Std. Dev.")
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_tsteps, interval=30)
    ani.save(filename=os.path.join(traj_dir, "pred_vs_true.gif"), writer="pillow")
    fig.show()
    print("Animation saved to", os.path.join(traj_dir, "pred_vs_true.gif"), flush=True)

if __name__ == "__main__":
    main()