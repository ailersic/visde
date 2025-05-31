from jaxtyping import Float
import torch
from torch import Tensor
import pickle as pkl
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
# ruff: noqa: F821, F722

from datasets.flow_control_2d.load_data import load_data

#import visde

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
plt.rcParams.update({'font.size': 16})
device = "cpu"

def main(train_val_test: str = "train") -> None:
    data = load_data()
    
    mu = data[f"{train_val_test}_mu"].to(device).numpy()
    t = data[f"{train_val_test}_t"].to(device).numpy()
    x = data[f"{train_val_test}_x"].to(device).numpy()
    f = data[f"{train_val_test}_f"].to(device).numpy()
    n_traj = mu.shape[0]

    plt.figure(figsize=(12, 3))
    for i in range(n_traj):
        if i < 5:
            plt.plot(t[i], f[i, :, 0])
    plt.xlabel("Time")
    plt.ylabel("Forcing")
    plt.xticks(np.arange(0.0, 0.6666667, 0.1666666), [r"$0$", r"$T_v/2$", r"$T_v$", r"$3T_v/2$", r"$2T_v$"])
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(CURR_DIR, "forcing.pdf"), format="pdf")

if __name__ == "__main__":
    main()