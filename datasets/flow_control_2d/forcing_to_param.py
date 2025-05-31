from datasets.flow_control_2d.load_data import load_data

import torch
from matplotlib import pyplot as plt
import pickle as pkl
import os
import pathlib

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def main(data_file: str = "data.pkl"):
    data = load_data(data_file)

    n_traj_train = data["train_mu"].shape[0]
    n_traj_val = data["val_mu"].shape[0]
    n_traj_test = data["test_mu"].shape[0]

    train_f_snapshots = data["train_f"][:, :, 0]
    U, S, VH = torch.linalg.svd(train_f_snapshots.T, full_matrices=False)
    cumul_energy = torch.cumsum(S**2, dim=0)/torch.sum(S**2)
    n_modes_99 = torch.sum(cumul_energy < 0.99) + 1
    print("Number of modes to capture 99% energy:", n_modes_99.item())
    modes = U[:, :n_modes_99]

    train_f_params = torch.einsum("ij,jk->ik", [train_f_snapshots, modes])
    val_f_params = torch.einsum("ij,jk->ik", [data["val_f"][:, :, 0], modes])
    test_f_params = torch.einsum("ij,jk->ik", [data["test_f"][:, :, 0], modes])

    data["train_mu"] = torch.cat([data["train_mu"], train_f_params], dim=1)
    data["val_mu"] = torch.cat([data["val_mu"], val_f_params], dim=1)
    data["test_mu"] = torch.cat([data["test_mu"], test_f_params], dim=1)

    data["train_f"] = torch.zeros_like(data["train_f"])
    data["val_f"] = torch.zeros_like(data["val_f"])
    data["test_f"] = torch.zeros_like(data["test_f"])
    
    print("Train mu:", data["train_mu"])
    print("Val mu:", data["val_mu"])
    print("Test mu:", data["test_mu"])
    
    # Save the modified data
    with open(os.path.join(CURR_DIR, data_file.split(".")[0] + "_param.pkl"), "wb") as f:
        pkl.dump(data, f)
    
if __name__ == "__main__":
    main()