using Distributions
using LinearAlgebra
using Random
using Quadrature, Cubature
using Plots
using IterTools
using SymPy
using DataFrames 
using CSV

########################################## SETUP ###########################################
global_rng = 2022
Random.seed!(global_rng)
n_full_data = 2500

################################## EXPERIMENT DEFINITION ###################################
p2d(x, y) = -0.005x^4 * 0.1x^3 + y^5 * (0.02x - 0.08) - 0.001y^2 + 0.2y + 0.5

function trainable_prms(k)
    return k.σ
end

exp_params = Dict([ 
    :n_train => 100,
    :n_fourier_feats => 100,
    :lb => [-4.0, -2.5],
    :ub => [4.0, 2.5],
    :jitter_val => 1e-7,
    :λ_init => [1.000001],
    :ls_init => [2.0, 2.0],  # kernel variance
    :noise_sd => [0.1],
    :sin_feats => false,
    :μₓ => [0.0, 0.0],
    :Σₓ => [4.0, 2.5],  # covariance, not sd
    :trainable_params_func => trainable_prms,
    :true_func => p2d,
    :opt_params => Dict([]),
    :opt_steps => 1000,
    :rng => global_rng,
])

###################################### GENERATE DATA #######################################
_X, _y, _y_noisy = generate_experimental_data(p2d, n_full_data, exp_params[:lb], exp_params[:ub], 0.0,  global_rng)
data = vcat(_X, _y', _y_noisy')
surface(-4:0.1:4, -2.5:0.1:2.5, p2d)

# quadrature solution
quad_sol = quadrature(p2d, exp_params[:lb], exp_params[:ub])  # 19.916666

# analytical solution
@syms x_ana, y_ana
expr = -0.005x_ana^4 * 0.1x_ana^3 + y_ana^5 * (0.02x_ana - 0.08) - 0.001y_ana^2 + 0.2y_ana + 0.5
analytical = integrate(
    expr, 
    (x_ana, exp_params[:lb][1], exp_params[:ub][1]),
    (y_ana, exp_params[:lb][2], exp_params[:ub][2])
)  # 19.91666
exp_params[:analytical_sol] = analytical

######################################## EXPERIMENT ########################################
# res = experiment(;exp_params...)
runs_per_n = 10
ns = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
res_means, res_stds, err_means, err_σ = exp_runs_over_n(ns, runs_per_n, exp_params)

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
CSV.write("experiments/2D/results/p2d_data.csv", DataFrame(data', ["x1", "x2", "y", "y_noisy"]))
