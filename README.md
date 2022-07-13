# GBQ.jl
Generalized Bayesian Quadrature Using Spectral Kernels

## Instructions

Currently, GBQ is not implemented in compiled package form and relies and manually loading several scripts into an environment to execute an experiment. This will be changed with future updates.

After making sure the project dependencies in `Project.toml` are installed, the current process to run an experiment is:
1. Open a julia console.
2. Run all code in `utils.jl`
3. Run all code in `sampling.jl`
4. Run all code in `dists.jl`
5. Run all code in `rff.jl`
6. Run all code in `baselines.jl`
7. Run all code in `generalizedbq.jl`
8. Run all code in `train.jl`
9. Run all code of chosen experiment in `experiments` directory.

Note: certain experiments in the camera-ready paper are run under different package iterations, which can affect the random seeds. To appropriately reproduce results, find the associated CSV of results in the `results/` folder for the experiment dimensionality (ie. 1D, 2D, ND), rewind to the commit when the CSV of results was last changed, and then run the experimental code.

To produce figures and tables used in the paper from generated data, you can run `plots.jl`.