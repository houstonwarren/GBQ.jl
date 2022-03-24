using Distributions
using LinearAlgebra
using Random
using Quadrature
using Plots
using KernelFunctions
using IterTools
using SymPy
using DataFrames
using CSV

########################################## SETUP ###########################################
global_rng = 2022
Random.seed!(global_rng)
n_full_data = 1000   

###################################### GENERATE DATA #######################################
function polynomial_1d(x)
    out = (x-4)^2 * .2*x^3 - 3x -3
    return out
end

function generate_poly1d_data(n, lb, ub, rng)
    n_dim = length(lb)
    dim_delta = ub - lb
    halton_samples = halton_sampler(n, n_dim, rng)
    deltas = halton_samples .* dim_delta'
    X = lb' .+ deltas
    z = map(i -> polynomial_1d(X[i, :]...), 1:size(X, 1))
    return X, z
end

# full data
X, y = generate_poly1d_data(n_full_data, [0], [5], global_rng)
data = DataFrame(hcat(X, y), ["x", "y"])

# plot
scatter(X, y)

# analytical solution
@syms x_ana
expr = (x_ana-4)^2 * .2*x_ana^3 - 3x_ana -3
simplify(expr)
analytical = integrate(expr, 0, 5)  # -31.66

# quadrature solution
quadrature(polynomial_1d, [0], [5], 0).u # -31.66

################################## EXPERIMENT DEFINITION ###################################
function run_once(;
    n_train,
    lb, ub,
    jitter_val, noise_sd,
    ls, λ,
    n_fourier_feats,
    μₓ, σₓ,
    rng)
    
    ############### GENERATE DATA ################
    X_train, y_train =  generate_data(generate_poly1d_data, n_train, lb, ub, noise_sd, rng)

    ############# GENERATE FEATURES ##############
    dim = length(lb)
    ff, fb, ffₖ, fbₖ, ff_m12, ff_m32, ff_m52 = generate_features(n_fourier_feats, dim, rng)

    ############### CREATE KERNELS ###############
    # rbf
    rbf = with_lengthscale(SqExponentialKernel(), ls)
    K_rbf = kernelmatrix(rbf, RowVecs(X_train))
    K_rbf = add_jitter(K_rbf, jitter_val)
    noisy_K_rbf = K_rbf + I * noise_sd^2
    
    # gaussian rff
    rffk = RandomFeatureKernel(ff, fb, ls, λ, true)
    K_rff = kernelmatrix(rffk, RowVecs(X_train))
    K_rff = add_jitter(K_rff, jitter_val)
    noisy_K_rff = K_rff + I * noise_sd^2

    # matern
    rff_m12 = RandomFeatureKernel(ff_m12, fb, ls, λ, true)
    K_m12 = add_jitter(kernelmatrix(rff_m12, RowVecs(X_train)), jitter_val)
    noisy_Km12 = K_m12 + I * noise_sd^2

    rff_m32 = RandomFeatureKernel(ff_m32, fb, ls, λ, true)
    K_m32 = add_jitter(kernelmatrix(rff_m32, RowVecs(X_train)), jitter_val)
    noisy_Km32 = K_m32 + I * noise_sd^2

    rff_m52 = RandomFeatureKernel(ff_m52, fb, ls, λ, true)
    K_m52 = add_jitter(kernelmatrix(rff_m52, RowVecs(X_train)), jitter_val)
    noisy_Km52 = K_m52 + I * noise_sd^2

    ################# BASELINES ##################
    ### analytical
    # done outside scope

    ### quadrature
    quad_est = quadrature(polynomial_1d, lb, ub, noise_sd, n_train).u

    ### mc integration
    mc_est, mc_sd = mc_quadrature(polynomial_1d, lb, ub, n_train, noise_sd, rng)

    # quasi mc integration
    qmc_est, qmc_sd = mc_quadrature_with_data(polynomial_1d, lb, ub, X_train, noise_sd, rng)

    ## bq
    pₓ = Normal(μₓ, σₓ)
    bq_est = bayesian_quadrature(X_train, ls[1], pₓ, noisy_K_rbf, y_train, lb[1], ub[1])

    #################### gbq #####################
    ### uniform
    gbq_uni_rff = gbq_uni_1d_μ(rffk, X_train, noisy_K_rff, y_train, lb[1], ub[1])
    gbq_uni_m12 = gbq_uni_1d_μ(rff_m12, X_train, noisy_Km12, y_train, lb[1], ub[1])
    gbq_uni_m32 = gbq_uni_1d_μ(rff_m32, X_train, noisy_Km32, y_train, lb[1], ub[1])
    gbq_uni_m52 = gbq_uni_1d_μ(rff_m52, X_train, noisy_Km52, y_train, lb[1], ub[1])

    ### gaussian
    rff_pₓ = RandomFeatureGaussian(ffₖ, pₓ.μ, pₓ.σ, false)
    gbq_gauss = gbq_gauss_μ_1d(rffk, rff_pₓ, X_train, noisy_K_rff, y_train, lb[1], ub[1])
    gbq_m12 = gbq_gauss_μ_1d(rff_m12, rff_pₓ, X_train, noisy_Km12, y_train, lb[1], ub[1])
    gbq_m32 = gbq_gauss_μ_1d(rff_m32, rff_pₓ, X_train, noisy_Km32, y_train, lb[1], ub[1])
    gbq_m52 = gbq_gauss_μ_1d(rff_m52, rff_pₓ, X_train, noisy_Km52, y_train, lb[1], ub[1])

    ################## RESULTS ###################
    results = [
        analytical, quad_est, mc_est, qmc_est, bq_est,
        gbq_uni_rff, gbq_uni_m12, gbq_uni_m32, gbq_uni_m52,
        gbq_gauss, gbq_m12, gbq_m32, gbq_m52
    ]
    print(results, "\n")
    return results
end

######################################## EXPERIMENT ########################################
exp_params = Dict([
    :n_train => 100,
    :lb => [0.0],
    :ub => [5.0],
    :noise_sd => 0.1,
    :rng => global_rng,
    :jitter_val => 1e-7,
    :n_fourier_feats => 100,
    :μₓ => 2.5,
    :σₓ => 2,
    :jitter_val => 1e-7,
    :ls => [1.0],
    :λ => 1.0,
])

# res = run_once(;exp_params...)

ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
runs_per_n = 10
res_means, res_stds, err_means, err_σ = exp_runs_over_n(ns, runs_per_n, run_once, exp_params)

######################################### RESULTS ##########################################
# results
nms = [
    "n", "true", "quad", "mc", "qmc", "bq", "gbq_uni", "gbq_uni_m12", "gbq_uni_m32", "gbq_uni_m52",
    "gbq_gauss", "gbq_gauss_m12", "gbq_gauss_m32", "gbq_gauss_m52",
]
mean_df = DataFrame(res_means, nms)
std_df = DataFrame(res_stds, nms)

# pct error
err_nms = [
    "n", "quad", "mc", "qmc", "bq", "gbq_uni", "gbq_uni_m12", "gbq_uni_m32", "gbq_uni_m52",
    "gbq_gauss", "gbq_gauss_m12", "gbq_gauss_m32", "gbq_gauss_m52"
]
err_df = DataFrame(err_means, err_nms)
err_σ_df = DataFrame(err_σ, err_nms)

# plot
# plot
plot_labels = [
    "quad" "mc" "qmc" "bq" "gbq_uni" "gbq_uni_m12" "gbq_uni_m32" "gbq_uni_m52" "gbq_gauss" "gbq_gauss_m12" "gbq_gauss_m32" "gbq_gauss_m52"
]

plot_err(err_df, plot_labels)
  
# save
# save_results(mean_df, std_df, err_df, runs_per_n, "experiments/2D/disjoint_1d.hdf5")
CSV.write("experiments/1D/results/poly_1d_means.csv", mean_df)
CSV.write("experiments/1D/results/poly_1d_stds.csv", std_df)
CSV.write("experiments/1D/results/poly_1d_data.csv", data)
CSV.write("experiments/1D/results/poly_1d_err_stds.csv", err_σ_df)
CSV.write("experiments/1D/results/poly_1d_err_means.csv", err_df)
