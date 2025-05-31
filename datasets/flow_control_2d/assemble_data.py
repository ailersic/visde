import pyvista as pv
import numpy as np
import torch
import pickle as pkl
import matplotlib.pyplot as plt

n_traj_train = 60
n_traj_val = 5
n_traj_test = 5
n_per_traj = 2

t_llim = 66 # up to 400
t_ulim = 333 # up to 400
x_ulim = 440 # up to 440
t_per_traj = (t_ulim - t_llim)//n_per_traj

assert t_llim + t_per_traj*n_per_traj <= t_ulim, "t_llim and t_per_traj do not match"

def load_partitioned_trajectories(start_traj, end_traj, color):
    x = torch.zeros((end_traj - start_traj)*n_per_traj, t_per_traj, 3, 80, x_ulim)
    f = torch.zeros((end_traj - start_traj)*n_per_traj, t_per_traj, 2)

    for i in range(start_traj, end_traj):
        for k in range(n_per_traj):
            print(f'Processing run {i}, partition {k} ... ', end='', flush=True)
            for j in range(t_llim + k*t_per_traj, t_llim + (k+1)*t_per_traj):
                if (j - t_llim - k*t_per_traj) % (t_per_traj//4) == 0:
                    print(f't = {j} ... ', end='', flush=True)
                u_filename = f'results_{i}/resample_u/resample_u_0_{j}.vti'
                u_vti_file = pv.read(u_filename)
                vel_field = u_vti_file.point_data['velocity'].reshape(80, 440, 3).transpose(2, 0, 1)[0:2, :, :x_ulim]

                p_filename = f'results_{i}/resample_p/resample_p_0_{j}.vti'
                p_vti_file = pv.read(p_filename)
                pres_field = p_vti_file.point_data['pressure'].reshape(80, 440, 1).transpose(2, 0, 1)[:, :, :x_ulim]

                x[(i - start_traj)*n_per_traj + k, j - t_llim - k*t_per_traj] = torch.cat((torch.tensor(vel_field), torch.tensor(pres_field)), dim=0)
            print('done', flush=True)

            csv = np.loadtxt(open(f'results_{i}/test_strategy.csv', "rb"), delimiter=";", skiprows=1, usecols=range(1,7))
            f[(i - start_traj)*n_per_traj + k] = torch.tensor(csv[::10, -2:])[(t_llim + k*t_per_traj):(t_llim + (k+1)*t_per_traj)]
            plt.plot(np.arange(f.shape[1]), f[(i - start_traj)*n_per_traj + k, :, 0].numpy(), color=color, alpha=0.2+0.8*(i - start_traj)/(end_traj - start_traj))
    
    return x, f

def main():
    true_t = torch.linspace(0, 2, 400)

    train_x = torch.zeros(n_traj_train*n_per_traj, t_per_traj, 3, 80, x_ulim)
    train_f = torch.zeros(n_traj_train*n_per_traj, t_per_traj, 2)
    train_mu = torch.zeros(n_traj_train*n_per_traj, 1)
    train_t = torch.linspace(true_t[0], true_t[t_per_traj-1], t_per_traj).unsqueeze(0).repeat(n_traj_train*n_per_traj, 1)

    val_x = torch.zeros(n_traj_val*n_per_traj, t_per_traj, 3, 80, x_ulim)
    val_f = torch.zeros(n_traj_val*n_per_traj, t_per_traj, 2)
    val_mu = torch.zeros(n_traj_val*n_per_traj, 1)
    val_t = torch.linspace(true_t[0], true_t[t_per_traj-1], t_per_traj).unsqueeze(0).repeat(n_traj_val*n_per_traj, 1)

    test_x = torch.zeros(n_traj_test*n_per_traj, t_per_traj, 3, 80, x_ulim)
    test_f = torch.zeros(n_traj_test*n_per_traj, t_per_traj, 2)
    test_mu = torch.zeros(n_traj_test*n_per_traj, 1)
    test_t = torch.linspace(true_t[0], true_t[t_per_traj-1], t_per_traj).unsqueeze(0).repeat(n_traj_test*n_per_traj, 1)

    fig = plt.figure(figsize=(10, 5))

    train_x, train_f = load_partitioned_trajectories(0, n_traj_train, 'blue')
    val_x, val_f = load_partitioned_trajectories(n_traj_train, n_traj_train + n_traj_val, 'red')
    test_x, test_f = load_partitioned_trajectories(n_traj_train + n_traj_val, n_traj_train + n_traj_val + n_traj_test, 'green')
    plt.savefig("forcing.png")

    '''
    all_f = np.concatenate((train_f[:,:,0], val_f[:,:,0], test_f[:,:,0]), axis=0)
    #mu = np.mean(all_f, axis=0, keepdims=True)
    #delta_f = all_f - mu
    #print(delta_f.shape)
    svdvals_f = np.linalg.svd(all_f, compute_uv=False)
    cum_energy = np.cumsum(svdvals_f ** 2) / np.sum(svdvals_f ** 2)
    n_modes_99 = np.sum(cum_energy < 0.99) + 1
    print(f"{n_modes_99} modes needed for 99% energy captured")
    '''

    data = {"train_mu": train_mu,
            "train_t": train_t,
            "train_x": train_x,
            "train_f": train_f,
            "val_mu": val_mu,
            "val_t": val_t,
            "val_x": val_x,
            "val_f": val_f,
            "test_mu": test_mu,
            "test_t": test_t,
            "test_x": test_x,
            "test_f": test_f}

    with open("data.pkl", "wb") as f:
        pkl.dump(data, f, protocol=4)

    print("Data assembled and saved.")

if __name__ == "__main__":
    main()

'''
with open("data.pkl", "rb") as f:
    data_in = pkl.load(f)
    assert data_in["train_mu"].shape == train_mu.shape
    assert data_in["train_t"].shape == train_t.shape
    assert data_in["train_x"].shape == train_x.shape
    assert data_in["train_f"].shape == train_f.shape

    assert data_in["val_mu"].shape == val_mu.shape
    assert data_in["val_t"].shape == val_t.shape
    assert data_in["val_x"].shape == val_x.shape
    assert data_in["val_f"].shape == val_f.shape

    assert data_in["test_mu"].shape == test_mu.shape
    assert data_in["test_t"].shape == test_t.shape
    assert data_in["test_x"].shape == test_x.shape
    assert data_in["test_f"].shape == test_f.shape
    print("Data verified.")'''