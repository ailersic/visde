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
from experiments.forced_burgers_1d.def_model import create_latent_sde

plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
OUT_DIR = os.path.join(CURR_DIR, "postproc_visde")
N_TRAIN = 50
DATA_FILE = f"data_ll_{N_TRAIN}_{int(N_TRAIN/5)}_50.pkl"

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

    dim_z = 9
    dim_x = x.shape[-1]
    #n_traj = mu.shape[0]
    n_win = 1
    n_batch = 32
    n_tsteps = t.shape[1]

    sde_options = {
        'method': 'srk',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-4,
        'atol': 1e-6
    }

    dummy_model = create_latent_sde(dim_z, n_batch, n_win, DATA_FILE)
    model = visde.LatentSDE.load_from_checkpoint(f"experiments/forced_burgers_1d/logs_visde/version_ref_{N_TRAIN}_ll_2/checkpoints/epoch=49-step={N_TRAIN*400}.ckpt",
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
    
    i_traj = 0

    fig = plt.figure(figsize=(16, 7))
    axgrid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 4),
                    axes_pad=0.20,
                    share_all=True,
                    direction="column"
                    )

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
    
    def update(j):
        if j % 10 == 0:
            print(f"Frame {j}/{n_tsteps}", flush=True)
        # j is time index
        xs = model.decoder.sample(1, mu_i_batch, zs[j]).detach()
        x_mean = xs.mean(dim=0).cpu().detach().numpy()
        x_std = xs.std(dim=0).cpu().detach().numpy()
        x_true = x[i_traj].cpu().detach().numpy()

        axgrid[0].cla()
        axgrid[0].plot(np.linspace(0, 1, dim_x), x_true[j], color="blue")
        axgrid[0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        axgrid[0].set_ylim(-0.2, 1.8)
        
        axgrid[1].cla()
        axgrid[1].plot(np.linspace(0, 1, dim_x), x_mean, color="blue")
        axgrid[1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        axgrid[1].set_ylim(-0.2, 1.8)
        
        axgrid[2].cla()
        axgrid[2].plot(np.linspace(0, 1, dim_x), np.abs(x_mean - x_true[j]), color="red")
        axgrid[2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        axgrid[2].set_ylim(-0.2, 1.8)

        axgrid[3].cla()
        axgrid[3].plot(np.linspace(0, 1, dim_x), x_std, color="red")
        axgrid[3].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        axgrid[3].set_ylim(-0.2, 1.8)

        axgrid[0].set_title("True Solution")
        axgrid[1].set_title("Prediction Mean")
        axgrid[2].set_title("Absolute Error")
        axgrid[3].set_title("Prediction Std. Dev.")
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_tsteps, interval=30)
    ani.save(filename=os.path.join(OUT_DIR, f"{train_val_test}_{N_TRAIN}_pred_vs_true.gif"), writer="pillow")
    fig.show()

if __name__ == "__main__":
    #main("train")
    main("test")