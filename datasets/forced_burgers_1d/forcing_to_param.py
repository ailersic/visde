from datasets.forced_burgers_1d.load_data import load_data

import torch
from matplotlib import pyplot as plt
import pickle as pkl
import os
import pathlib

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def estimate_sine_frequency(y: torch.Tensor) -> float:
    amp = torch.max(torch.abs(y))/(5*torch.pi)
    # amplitude is coupled to the frequency in this dataset

    return amp.item()

def main(data_file: str = "data_bcf_100_10_10.pkl"):
    data = load_data(data_file)

    n_traj_train = data["train_mu"].shape[0]
    n_traj_val = data["val_mu"].shape[0]
    n_traj_test = data["test_mu"].shape[0]

    data["train_mu"] = torch.cat([data["train_mu"], torch.zeros((n_traj_train, 1))], dim=1)
    data["val_mu"] = torch.cat([data["val_mu"], torch.zeros((n_traj_val, 1))], dim=1)
    data["test_mu"] = torch.cat([data["test_mu"], torch.zeros((n_traj_test, 1))], dim=1)

    for i in range(n_traj_train):
        print(f"Trajectory {i}:")
        freq = estimate_sine_frequency(data["train_f"][i, :, 0])
        data["train_f"][i, :, 0] = torch.zeros_like(data["train_f"][i, :, 0])
        data["train_mu"][i, -1] = freq

    for i in range(n_traj_val):
        print(f"Trajectory {i}:")
        freq = estimate_sine_frequency(data["val_f"][i, :, 0])
        data["val_f"][i, :, 0] = torch.zeros_like(data["val_f"][i, :, 0])
        data["val_mu"][i, -1] = freq

    for i in range(n_traj_test):
        print(f"Trajectory {i}:")
        freq = estimate_sine_frequency(data["test_f"][i, :, 0])
        data["test_f"][i, :, 0] = torch.zeros_like(data["test_f"][i, :, 0])
        data["test_mu"][i, -1] = freq
    
    print("Train mu:", data["train_mu"])
    print("Val mu:", data["val_mu"])
    print("Test mu:", data["test_mu"])
    
    # Save the modified data
    with open(os.path.join(CURR_DIR, data_file.split(".")[0] + "_param.pkl"), "wb") as f:
        pkl.dump(data, f)
    
if __name__ == "__main__":
    main()