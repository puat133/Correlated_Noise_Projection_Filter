





# Correlated Automatic Projection Filter
This repository contains codes related to paper "Efficient Projection Filter Algorithm for Stochastic Dynamical Systems with Correlated Noises and State-Dependent Measurement Covariance" which can be downloaded from this [Paper Link](https://doi.org/10.1016/j.sigpro.2024.109383)

To reproduce the simulation results in the paper, you can execute the following script

Van der Pol two dimension
```bash
python ./Examples/signal_processing_correlated/Heteroscedastic_Van_Der_Pol_Simulation.py --slevel=5 --nparticle=50000 --nt=400 --dt=2.5e-3 --corrstrength=0.5 --maxgrid=8 --compare  --scale=1.3 --order=4 --moment_iterations=4 --seed=7 --compare  --plot --save
```

Van der Pol four dimension
```bash
python ./Examples/signal_processing_correlated/hetero_VDP_dim_4.py --slevel=6 --nparticle=50000 --nt=400 --dt=2.5e-3 --corrstrength=0.5 --maxgrid=8 --scale=1.3 --order=4 --moment_iterations=4 --seed=7 --integrator=spg --corrstrength=0.5 --plot --animate --save
```

Heston Model: Open `./Examples/signal_processing_correlated/heston_4.ipynb`