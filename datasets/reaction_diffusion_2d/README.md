# Preparing the reaction-diffusion dataset

To prepare this dataset, first obtain the data file `reaction_diffusion.mat` [provided on Github](https://github.com/kpchamp/SindyAutoencoders/) by Champion et al. and place it in `datasets/reaction_diffusion_2d`. Run the script `gen_data.py` with the `noisy` bool set to either true or false. This will create a file named either `data.pkl` or `data_noisy.pkl` respectively. You are now ready to begin training.
