import torch
import os
import pickle as pkl
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
OUT_DIR = os.path.join(CURR_DIR, "postprocess")

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def main():
    with open(os.path.join(CURR_DIR, "data.pkl"), "rb") as f:
        data = pkl.load(f)
    
    train_x = data["train_x"]
    print(train_x.shape)
    snapshots = torch.flatten(torch.flatten(train_x, start_dim=0, end_dim=1), start_dim=1)
    n_snap = snapshots.shape[0]
    print(snapshots.shape)
    S = torch.linalg.svdvals(snapshots - snapshots.mean(0, keepdim=True))
    print(S)
    energy_frac = S.square() / S.square().sum()

    plt.figure()
    plt.plot(S.numpy())
    plt.yscale("log")
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value")
    plt.show()
    plt.savefig(os.path.join(OUT_DIR, "pca_svals.png"))
    
    plt.figure()
    plt.plot(np.linspace(1, n_snap, n_snap), torch.cumsum(energy_frac, 0).numpy())
    plt.xscale("log")
    plt.xlabel("Number of modes")
    plt.ylabel("Energy fraction captured")
    plt.show()
    plt.savefig(os.path.join(OUT_DIR, "pca_energy.png"))

if __name__ == "__main__":
    main()