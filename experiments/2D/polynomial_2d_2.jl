using Distributions
using LinearAlgebra
using Random
using Quadrature, Cubature
using Plots
using KernelFunctions
using IterTools
using SymPy
using DataFrames 
using CSV

########################################## SETUP ###########################################
global_rng = 2022
Random.seed!(global_rng)
n_full_data = 2500

###################################### GENERATE DATA #######################################
p2d(x, y) = -0.005x^4 * 0.1x^3 + y^5 * (0.02x - 0.08) - 0.001y^2 + 0.2y + 0.5

function generate_p2d_data(n, lb, ub, rng)
    n_dim = length(lb)
    dim_delta = ub - lb
    halton_samples = halton_sampler(n, n_dim, rng)
    deltas = halton_samples .* dim_delta'
    X = lb' .+ deltas
    z = map(i -> p2d(X[i, :]...), 1:size(X, 1))
    return X, z
end

X, y = generate_p2d_data(n_full_data, [-4, -2.5], [4, 2.5], global_rng)
data = hcat(X, y)

# plot
surface(-4:0.1:4, -2.5:0.1:2.5, p2d)

# analytical solution
@syms x_ana, y_ana
expr = -0.005x_ana^4 * 0.1x_ana^3 + y_ana^5 * (0.02x_ana - 0.08) - 0.001y_ana^2 + 0.2y_ana + 0.5
analytical = integrate(expr, (x_ana, -4, 4), (y_ana, -2.5, 2.5))  # 19.91666

# quadrature solution
quadrature(p2d, [-4, -2.5], [4, 2.5], 0)  # 19.916666

################################## EXPERIMENT DEFINITION ###################################
function run_once(;
    n_train,
    lb, ub,
    jitter_val, noise_sd,
    ls, λ,
    n_fourier_feats,
    μₓ, Σₓ,
    rng)
    
    ############### GENERATE DATA ################
    X_train, y_train = generate_data(generate_p2d_data, n_train, lb, ub, noise_sd, rng)

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

    ### quadrature
    quad_est = quadrature(p2d, lb, ub, noise_sd, n_train, rng).u

    ### mc integration
    mc_est, mc_sd = mc_quadrature(p2d, lb, ub, n_train, noise_sd, rng)

    # quasi mc integration
    qmc_est, qmc_sd = mc_quadrature_with_data(p2d, lb, ub, X_train, noise_sd, rng)

    ## bq
    pₓ = MvNormal(μₓ, diagm(Σₓ))
    bq_est = bayesian_quadrature(X_train, diagm(ls).^2, pₓ, noisy_K_rbf, y_train, lb, ub)

    #################### gbq #####################
    ### uniform
    gbq_uni_rff = gbq_uni_nd_μ(rffk, X_train, noisy_K_rff, y_train, lb, ub)
    gbq_uni_m12 = gbq_uni_nd_μ(rff_m12, X_train, noisy_Km12, y_train, lb, ub)
    gbq_uni_m32 = gbq_uni_nd_μ(rff_m32, X_train, noisy_Km32, y_train, lb, ub)
    gbq_uni_m52 = gbq_uni_nd_μ(rff_m52, X_train, noisy_Km52, y_train, lb, ub)

    ### gaussian
    rff_pₓ = RandomFeatureMvGaussian(ffₖ, pₓ.μ, Matrix(pₓ.Σ), false)
    gbq_gauss = gbq_gauss_μ_nd(rffk, rff_pₓ, X_train, noisy_K_rff, y_train, lb, ub)
    gbq_m12 = gbq_gauss_μ_nd(rff_m12, rff_pₓ, X_train, noisy_Km12, y_train, lb, ub)
    gbq_m32 = gbq_gauss_μ_nd(rff_m32, rff_pₓ, X_train, noisy_Km32, y_train, lb, ub)
    gbq_m52 = gbq_gauss_μ_nd(rff_m52, rff_pₓ, X_train, noisy_Km52, y_train, lb, ub)

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
    :n_train => 200,
    :lb => [-4.0, -2.5],
    :ub => [4.0, 2.5],
    :noise_sd => 0.1,
    :rng => global_rng,
    :jitter_val => 1e-7,
    :n_fourier_feats => 300,
    :μₓ => [0.0, 0.0],
    :Σₓ => [4.0, 2.5],
    :jitter_val => 1e-7,
    :ls => [2.0, 2.0],
    :λ => 1.0,
])

# res = run_once(;exp_params...)

runs_per_n = 5
ns = [10, 25, 50, 100, 250, 500, 750]
# ns2 = [10, 25, 50, 100]
res_means, res_stds, err_means, err_sds = exp_runs_over_n(ns, runs_per_n, run_once, exp_params)

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
err_σ_df = DataFrame(err_sds, err_nms)

# plot
plot_labels = [
    "quad" "mc" "qmc" "bq" "gbq_uni" "gbq_uni_m12" "gbq_uni_m32" "gbq_uni_m52" "gbq_gauss" "gbq_gauss_m12" "gbq_gauss_m32" "gbq_gauss_m52"
]

# save
# save_results(mean_df, std_df, err_df, runs_per_n, "experiments/2D/p2d.hdf5")
CSV.write("experiments/2D/results/p2d_means.csv", mean_df)
CSV.write("experiments/2D/results/p2d_stds.csv", std_df)
CSV.write("experiments/2D/results/p2d_err_means.csv", err_df)
CSV.write("experiments/2D/results/p2d_err_stds.csv", err_σ_df)
CSV.write("experiments/2D/results/p2d_data.csv", DataFrame(data, ["x1", "x2", "y"]))
