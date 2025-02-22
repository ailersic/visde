# Variational Inference for Stochastic Differential Equations (VISDE)

PyTorch Lightning package for learning stochastic differential systems by variational inference. This package accompanies [this paper](https://blah.blah) by Andrew F. Ilersich, Kevin Course, and Prasanth B. Nair. The code for all test cases is provided in the `experiments` directory.

This package is based on the [arlatentsde](https://github.com/coursekevin/arlatentsde) package that accompanies [this paper](https://neurips.cc/virtual/2023/poster/72781) by Kevin Course and Prasanth B. Nair.

<p align="center">
  <img align="middle" src="./images/flowcontrol.gif" alt="Example with Flow Control over Cylinder" width="700"/>
</p>

If you find this work useful, please cite the following:

Ilersich, Course, Nair. Paper coming soon.

```
@article{comingsoon,
    title={Really cool paper that everyone will absolutely love}
}
```

## Contents

1. [Installation](#1-installation)
2. [Framework](#2-framework)
3. [Usage](#3-usage)
    1. [Training](#3i-training)
    2. [Prediction](#3ii-prediction)
4. [Modules](#4-modules)
    1. [VarEncoder](#4i-varencoder)
    2. [VarDecoder](#4ii-vardecoder)
    3. [LatentDrift](#4iii-latentdrift)
    4. [LatentDispersion](#4iv-latentdispersion)
    5. [LatentVar](#4v-latentvar)
    6. [AmortizedLatentVar](#4vi-amortizedlatentvar)
    7. [LogLike](#4vii-loglike)
    8. [Kernel](#4viii-kernel)

## 1. Installation

We ran all experiments on a Linux cluster with CUDA 12.4, using [poetry](https://github.com/python-poetry/poetry) to manage dependencies. All dependencies are listed in the pyproject.toml. Installation via pip coming soon.

## 2. Framework

This package learns latent SDE models for forced, parametrized dynamical systems. The dataset contains $N_T$ trajectories, each with associated parameters $\mu \in \mathbb{R}^{N_{\mu}}$, time-varying forcing $f(t) \in \mathbb{R}^{N_f}$, and time-varying quantities of interest (QoI) $u(t) \in \mathbb{R}^{D}$. The QoI may also have non-vector shape (e.g. $N_{\rm channels}\times D_x \times D_y$ for multi-channel 2D spatial field). Each trajectory contains $N$ time samples. There are four tensors that comprise this multi-trajectory dataset:

1. Parameter tensor: $M \in \mathbb{R}^{N_T\times N_{\mu}}$
2. Time tensor: $T \in \mathbb{R}^{N_T\times N}$
3. QoI tensor: $U \in \mathbb{R}^{N_T\times N \times D}$ (or for non-vector QoI, $\mathbb{R}^{N_T\times N \times D_1\times D_2\times \dots}$)
4. Forcing tensor: $F \in \mathbb{R}^{N_T\times N \times N_f}$

The model is then trained to capture the dependence of the QoI on latent dynamics, parameters, and forcing. It has three main components:
1. A probabilistic encoder $p_{\theta}^{\rm enc}(z(t)\ |\ u(t))$, assumed to be Gaussian, where $z(t) \in \mathbb{R}^{d}$,
2. A latent SDE model $\text{d}z = \psi_{\theta} (z, t; \mu, f(t)) \text{d}t + \Psi_{\theta}(t; \mu) \text{d}\beta$, where $\beta$ is a Wiener process,
3. A probabilistic decoder $p_{\theta}^{\rm dec}(u(t)\ |\ z(t))$, also assumed to be Gaussian.

After the model is trained, predictions are made given new parameters, forcing, and initial condition. We then make the predictions by a three-step process:

1. Sample the latent initial condition from the encoder, $z(0) \sim p_{\theta}^{\rm enc}(z_0\ |\ u_0)$,
2. Integrate the latent SDE to obtain a latent solution, $z(T) \sim \int_0^T {\psi_\theta}(z, t; \mu, f(t)) \text{d}t + \int_0^T {\Psi_\theta}(t; \mu) \text{d}{\beta}$,
3. Sample the QoI solution from the decoder, $u(T) \sim p_{\theta}^{\rm dec}(u(T)\ |\ z(T))$.

There is a minor nuance to the definition of the encoder that was ignored above for clarity, but we explain here for completeness. The encoder may be chosen to require a short window of $N_{\rm win}$ subequent time samples of the QoI to evaluate the latent state. For example, if $N_{\rm win} = 2$, then the variational encoder takes two subsequent time steps of $u(t)$ to form the latent state $z(t)$. This is useful for fundamentally second-order systems, such as pendulums, since our framework uses a first-order SDE as the latent model. The latent state must therefore encode both position and velocity, the latter of which can only be calculated from multiple time samples.

## 3. Usage

This section explains how to train latent SDE models and how to make predictions with a trained model.

### 3.i. Training

This package provides the `LatentSDE` class, which inherits `LightningModule`. This class is set up to use `MultiEvenlySpacedTensors` and `MultiTemporalSampler` defined in `data.py` to form the dataset and sampler respectively. The four data tensors defined above are used to form the dataset as follows:

```
from torch.utils.data import DataLoader
import visde

data = visde.MultiEvenlySpacedTensors(M, T, U, F, n_win)
sampler = visde.MultiTemporalSampler(data, n_batch, n_repeats=1)
loader = DataLoader(data, batch_sampler=sampler, ...)
```

The training process is then the same as with any other `LightningModule`. See the following short example:

```
import pytorch_lightning as pl

config = visde.LatentSDEConfig(n_totaldata,
                                n_samples,
                                n_tquad,
                                n_warmup,
                                n_transition,
                                lr,
                                lr_sched_freq)
model = visde.LatentSDE(config, *modules)
trainer = pl.Trainer(...)
trainer.fit(model, loader, ...)
```

The `LatentSDEConfig` is a dataclass with attributes:
- `n_totaldata`: total number of datapoints in dataset, i.e., $N_T\times N$
- `n_samples`: number of Monte Carlo samples of $z(t)$ when evaluating ELBO
- `n_tquad`: number of quadrature samples of $t$ when evaluating ELBO
- `n_warmup`: number of optimization steps where only log-likelihood term of ELBO is optimized (can usually keep this zero)
- `n_transition`: number of optimization steps to transition between log-likelihood and full ELBO
- `lr`: learning rate for default Adam optimizer
- `lr_sched_freq`: sets exponential decay rate for learning rate

The default optimizer is Adam with an exponential decay on the learning rate. The decay rate is controlled by `lr_sched_freq`, which is how many optimizer steps it takes for the learning rate to decrease by 10%. The user may define a custom optimizer by subclassing `LatentSDE` and overloading the `configure_optimizers` function. See `sde.py` for reference.

The `modules` are detailed in the Section 4.

### 3.ii. Prediction

With a trained `LatentSDE` model, we can make predictions as follows. We assume we are given an initial condition `u0` with shape `(1, N_win, D)`, time steps `t` with shape `(N,)`, parameters `mu` with shape `(1, N_mu)`, and forcing `f` with shape `(N, N_f)`.

As described in Section 2, we encode the initial condition, integrate the latent SDE using the `torchsde` package, and decode the solution as follows:

```
z0 = model.encoder.sample(1, mu, x0)

sde = visde.SDE(model.drift, model.dispersion, mu, t, f)
with torch.no_grad():
    zs = torchsde.sdeint(sde, z0, t, **sde_options)

xf = model.decoder.sample(1, mu, zs[-1])
```

You may also find a batch of `n_batch` solutions as follows:

```
z0 = model.encoder.sample(n_batch, mu, x0)

sde = visde.SDE(model.drift, model.dispersion, mu, t, f)
with torch.no_grad():
    zs = torchsde.sdeint(sde, z0, t, **sde_options)

xf = model.decoder.sample(1, mu.repeat((n_batch, 1)), zs[-1])
```

Note that in the decoder argument, `mu` is repeated along its batch dimension to match `zs`.

## 4. Modules

Instantiating `LatentSDE` requires the following components, defined as Protocols:

### 4.i. VarEncoder
This module is used to obtain mean and variance of the latent state distribution.

#### General Use:
A `VarEncoder` is defined in terms of a Protocol in `autoencoder.py` and shown below. The user may define any `nn.Module` that conforms to this Protocol.

<details>
    <summary>VarEncoder Protocol</summary>

    class VarEncoder(Protocol):
        def resample_params(self) -> None:
            ...
        
        def kl_divergence(self) -> Tensor:
            ...

        def __call__(self,
                    mu: Float[Tensor, "n_batch dim_mu"],
                    x_win: Float[Tensor, "n_batch n_win ..."]
        ) -> tuple[Float[Tensor, "n_batch dim_z"],
                Float[Tensor, "n_batch dim_z"]
        ]:
            ...

        def sample(self,
                n_samples: int,
                mu: Float[Tensor, "n_batch dim_mu"],
                x_win: Float[Tensor, "n_batch n_win ..."]
        ) -> Float[Tensor, "n_batch*n_samples dim_z"]:
            ...
</details>

If a statistical prior is placed on the parameters of a `VarEncoder`, the functions `resample_params` and `kl_divergence` must be appropriately defined.

#### Default with no prior:

The default class `VarEncoderNoPrior` is provided; it parameterizes the variational encoder as an `nn.Module` with no prior on its parameters. Because there is no prior, the functions `resample_params` and `kl_divergence` are `pass` and `return torch.tensor(0.0)` respectively. The class is instantiated as follows:

```
config = visde.VarAutoencoderConfig(dim_mu, dim_x, dim_z, shape_x, n_win)
encoder = visde.VarEncoderNoPrior(config, mean_net, var_net)
```
where `mean_net` and `var_net` are arbitrary `nn.Module` classes used to evaluate the mean and variance of the encoder respectively. The `VarAutoencoderConfig` is a dataclass with attributes:
- `dim_mu`: dimension of dataset parameters, $N_{\mu}$
- `dim_x`: dimension of QoI, $D$
- `dim_z`: dimension of latent state, $d$
- `shape_x`: if QoI is not a vector, this is its shape; otherwise this is $(D,)$
- `n_win`: number of consecutive state samples in time needed to obtain the latent state

### 4.ii. VarDecoder
This module is used to obtain mean and variance of the QoI distribution given a latent state.

#### General Use:
A `VarDecoder` is defined in terms of a Protocol in `autoencoder.py` and shown below. The user may define any `nn.Module` that conforms to this Protocol.

<details>
    <summary>VarDecoder Protocol</summary>

    class VarDecoder(Protocol):
        def resample_params(self) -> None:
            ...
        
        def kl_divergence(self) -> Tensor:
            ...
        
        def __call__(self,
                    mu: Float[Tensor, "n_batch dim_mu"],
                    z: Float[Tensor, "n_batch dim_z"]
        ) -> tuple[Float[Tensor, "n_batch *shape_x"],
                Float[Tensor, "n_batch *shape_x"]
        ]:
            ...

        def sample(self,
                n_samples: int,
                mu: Float[Tensor, "n_batch dim_mu"],
                z: Float[Tensor, "n_batch dim_z"]
        ) -> Float[Tensor, "..."]:
            ...
</details>

If a statistical prior is placed on the parameters of a `VarDecoder`, the functions `resample_params` and `kl_divergence` must be appropriately defined.

#### Default with no prior:

The default class `VarDecoderNoPrior` is provided; it parameterizes the variational decoder as an `nn.Module` with no prior on its parameters. Because there is no prior, the functions `resample_params` and `kl_divergence` are `pass` and `return torch.tensor(0.0)` respectively. The class is instantiated as follows:

```
config = visde.VarAutoencoderConfig(dim_mu, dim_x, dim_z, shape_x, n_win)
decoder = visde.VarDecoderNoPrior(config, mean_net, var_net)
```
where `mean_net` and `var_net` are arbitrary `nn.Module` classes used to evaluate the mean and variance of the decoder respectively. The `VarAutoencoderConfig` dataclass is detailed in the previous section.

### 4.iii. LatentDrift
This module is the drift function of the latent SDE.

#### General Use:
A `LatentDrift` is defined in terms of a Protocol in `sdeprior.py` and shown below. The user may define any `nn.Module` that conforms to this Protocol.

<details>
    <summary>LatentDrift Protocol</summary>

    class LatentDrift(Protocol):
        def resample_params(self) -> None:
            ...
        
        def kl_divergence(self) -> Tensor:
            ...
        
        def __call__(self,
                    mu: Float[Tensor, "n_batch dim_mu"],
                    t: Float[Tensor, "n_batch 1"],
                    z: Float[Tensor, "n_batch dim_z"],
                    f: Float[Tensor, "n_batch dim_f"]
        ) -> Float[Tensor, "n_batch dim_z"]:
            ...
</details>

If a statistical prior is placed on the parameters of a `LatentDrift`, the functions `resample_params` and `kl_divergence` must be appropriately defined.

#### Default with no prior:

The default class `LatentDriftNoPrior` is provided; it parameterizes the variational decoder as an `nn.Module` with no prior on its parameters. Because there is no prior, the functions `resample_params` and `kl_divergence` are `pass` and `return torch.tensor(0.0)` respectively. The class is instantiated as follows:

```
driftconfig = visde.LatentDriftConfig(dim_mu, dim_z, dim_f)
drift = visde.LatentDriftNoPrior(driftconfig, drift_net)
```
where `drift_net` is an arbitrary `nn.Module` class used to evaluate the drift function. The `LatentDriftConfig` is a dataclass with attributes:
- `dim_mu`: dimension of dataset parameters, $N_{\mu}$
- `dim_z`: dimension of latent state, $d$
- `dim_f`: dimension of dataset forcing, $N_f$

### 4.iv. LatentDispersion
This module is the dispersion matrix (diagonal only) of the latent SDE.

#### General Use:
A `LatentDispersion` is defined in terms of a Protocol in `sdeprior.py` and shown below. The user may define any `nn.Module` that conforms to this Protocol.

<details>
    <summary>LatentDispersion Protocol</summary>

    class LatentDispersion(Protocol):
        def resample_params(self) -> None:
            ...
        
        def kl_divergence(self) -> Tensor:
            ...
        
        def __call__(self,
                    mu: Float[Tensor, "n_batch dim_mu"],
                    t: Float[Tensor, "n_batch 1"]
        ) -> Float[Tensor, "n_batch dim_z"]:
            ...
</details>

If a statistical prior is placed on the parameters of a `LatentDispersion`, the functions `resample_params` and `kl_divergence` must be appropriately defined.

#### Default with no prior:

The default class `LatentDispersionNoPrior` is provided; it parameterizes the variational decoder as an `nn.Module` with no prior on its parameters. Because there is no prior, the functions `resample_params` and `kl_divergence` are `pass` and `return torch.tensor(0.0)` respectively. The class is instantiated as follows:

```
dispconfig = visde.LatentDispersionConfig(dim_mu, dim_z)
dispersion = visde.LatentDispersionNoPrior(dispconfig, disp_net)
```
where `disp_net` is an arbitrary `nn.Module` class used to evaluate the diagonal of the dispersion matrix. The `LatentDispersionConfig` is a dataclass with attributes:
- `dim_mu`: dimension of dataset parameters, $N_{\mu}$
- `dim_z`: dimension of latent state, $d$

### 4.v. LatentVar
This module is the directly-parametrized variational distribution over the latent state. Either this or `AmortizedLatentVar` is necessary, but not both.

#### General Use:
A `LatentVar` is defined in terms of a Protocol in `sdevar.py` and shown below. The user may define any `nn.Module` that conforms to this Protocol.

<details>
    <summary>LatentVar Protocol</summary>

    class LatentVar(Protocol):
        def __call__(self,
                    mu: Float[Tensor, "n_batch dim_mu"],
                    t: Float[Tensor, "n_batch 1"]
        ) -> tuple[Float[Tensor, "n_batch dim_z"],
                Float[Tensor, "n_batch dim_z"],
                Float[Tensor, "n_batch dim_z"],
                Float[Tensor, "n_batch dim_z"]
        ]:
            ...
        
        def sample(self,
                n_samples: int,
                mu: Float[Tensor, "n_batch dim_mu"],
                t: Float[Tensor, "n_batch 1"]
        ) -> Float[Tensor, "... dim_z"]:
            ...
</details>

#### Default with latent Gaussian process:

The default class `LatentVarGP` is provided; it parameterizes the variational distribution as a Gaussian process. The class is instantiated as follows:

```
varconfig = visde.LatentVarConfig(dim_mu, dim_z)
latentvar = visde.LatentVarGP(varconfig, z_mean_net, z_var_net)
```
where `z_mean_net` and `z_var_net` are arbitrary `nn.Module` classes used to evaluate the mean and variance of the Gaussian process respectively. The `LatentVarConfig` is a dataclass with attributes:
- `dim_mu`: dimension of dataset parameters, $N_{\mu}$
- `dim_z`: dimension of latent state, $d$

### 4.vi. AmortizedLatentVar
This module is the amortized variational distribution over the latent state. Either this or `LatentVar` is necessary, but not both.

#### General Use:
A `AmortizedLatentVar` is defined in terms of a Protocol in `sdevar.py` and shown below. The user may define any `nn.Module` that conforms to this Protocol.

<details>
    <summary>AmortizedLatentVar Protocol</summary>

    class AmortizedLatentVar(Protocol):
        def __call__(self,
                    mu: Float[Tensor, "n_batch dim_mu"],
                    t: Float[Tensor, "n_batch 1"],
        ) -> tuple[Float[Tensor, "n_batch dim_z"],
                    Float[Tensor, "n_batch dim_z"],
                    Float[Tensor, "n_batch dim_z"],
                    Float[Tensor, "n_batch dim_z"]
        ]:
            ...
        
        def form_window(self,
                        mu: Float[Tensor, "n_batch dim_mu"],
                        t: Float[Tensor, "n_batch 1"],
                        x_win: Float[Tensor, "n_batch n_win *shape_x"]
        ) -> None:
            ...
        
        def sample(self,
                n_samples: int,
                mu: Float[Tensor, "n_batch dim_mu"],
                t: Float[Tensor, "n_batch 1"]
        ) -> Float[Tensor, "... dim_z"]:
            ...
</details>

#### Default with latent Gaussian process:

The class `AmortizedLatentVarGP` is provided; it defines the variational distribution as a Gaussian process over encoded samples of the true state. It requires a `Kernel` and `VarEncoder` to already be instantiated. The class is instantiated as follows:

```
varconfig = visde.LatentVarConfig(dim_mu, dim_z)
latentvargp = visde.AmortizedLatentVarGP(varconfig, kernel, encoder)
```
The `LatentVarConfig` dataclass is detailed in the previous section.

### 4.vii. LogLike
This module calculates the log-likelihood of samples from the dataset given predictions from the decoded latent state.

#### General Use:
A `LogLike` is defined in terms of a Protocol in `likelihood.py` and shown below. The user may define any `nn.Module` that conforms to this Protocol.

<details>
    <summary>LogLike Protocol</summary>

    class LogLike(Protocol):
        def __call__(self,
                    x_true: Float[Tensor, "n_batch dim_x"],
                    x_mean: Float[Tensor, "n_batch n_samples dim_x"],
                    x_var: Float[Tensor, "n_batch n_samples dim_x"]
        ) -> Float[Tensor, "n_batch n_samples"]:
            ...
</details>

#### Default with Gaussian likelihood:

The class `LogLikeGaussian` is provided; it defines the likelihood function as a simple Gaussian. The class is instantiated as follows:

```
loglike = visde.LogLikeGaussian()
```
No config dataclass or other arguments are necessary.

### 4.viii. Kernel
This module is used by `AmortizedLatentVar` to form a distribution over the latent state given a time-window of QoI.

#### General Use:
A `Kernel` is defined in terms of a Protocol in `kernel.py` and shown below. The user may define any `nn.Module` that conforms to this Protocol.

<details>
    <summary>Kernel Protocol</summary>

    class Kernel(Protocol):
        def __call__(
            self,
            t1: Float[Tensor, "n 1"],
            t2: Float[Tensor, "m 1"]
        ) -> Float[Tensor, "n m"]:
            ...

        @property
        def var(self) -> Tensor:
            ...
</details>

#### Default with deep Gaussian kernel:

The class `DeepGaussianKernel` is provided; it defines a Gaussian kernel with a user-defined `nn.Module` applied to each of its arguments. The class is instantiated as follows:

```
kernel = visde.DeepGaussianKernel(kernel_net, n_batch, dt)
```
where `kernel_net` is an arbitrary `nn.Module` class applied to each input of the Gaussian kernel. The other arguments, used for input scaling, are:
- `n_batch`: batch size used by dataloader
- `dt`: time step size in dataset
