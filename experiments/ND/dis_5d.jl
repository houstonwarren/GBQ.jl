using Distributions
using LinearAlgebra
using Random
using Quadrature, Cubature
using Plots
using IterTools
using DataFrames 
using CSV

########################################## SETUP ###########################################
global_rng = 2022
Random.seed!(global_rng)
n_full_data = 2500

################################## EXPERIMENT DEFINITION ###################################
function dis5d_func(x1, x2, x3, x4, x5)
    if x1 > 0.5 && x2 > 0.5 && x3 > 0.5 && x4 > 0.5 && x5 > 0.5
        z = (10sin(π*x1*x2) + 20(x3 - 1/2) + 10x4 + 5x5) * 4
    else
        z = 10sin(π*x1*x2) + 20(x3 - 1/2) + 10x4 + 5x5
    end
    return z
end

function trainable_prms(k)
    return k.σ
end

exp_params = Dict([
    :n_train => 100,
    :n_fourier_feats => 100,
    :lb => [0.0, 0.0, 0.0, 0.0, 0.0],
    :ub => [1.0, 1.0, 1.0, 1.0, 1.0],
    :jitter_val => 1e-7,
    :λ_init => [1.000001],
    :ls_init => [1.0, 1.0, 1.0, 1.0, 1.0],  # kernel variance
    :noise_sd => [0.5],
    :sin_feats => false,
    :μₓ => [0.5, 0.5, 0.5, 0.5, 0.5],
    :Σₓ => [1.0, 1.0, 1.0, 1.0, 1.0] .* 0.25,  # covariance, not sd
    :trainable_params_func => trainable_prms,
    :true_func => dis5d_func,
    :opt_params => Dict([]),
    :opt_steps => 1000,
    :rng => global_rng,
])

###################################### GENERATE DATA #######################################
_X, _y, _y_noisy = generate_experimental_data(dis5d_func, n_full_data, exp_params[:lb], exp_params[:ub], 0.0,  global_rng)
data = vcat(_X, _y', _y_noisy')

# quadrature solution
analytical = quadrature(dis5d_func, exp_params[:lb], exp_params[:ub])  # 15.087708873902969
exp_params[:analytical_sol] = analytical

######################################## EXPERIMENT ########################################
# res = experiment(;exp_params...)
runs_per_n = 10
ns = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
res_means, res_stds, err_means, err_σ, raw_results = exp_runs_over_n(ns, runs_per_n, exp_params)

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

# save
# save_results(mean_df, std_df, err_df, runs_per_n, "experiments/ND/dis5d_exp.hdf5")
CSV.write("experiments/ND/results/dis5d_means.csv", mean_df)
CSV.write("experiments/ND/results/dis5d_stds.csv", std_df)
CSV.write("experiments/ND/results/dis5d_err_means.csv", err_df)
CSV.write("experiments/ND/results/dis5d_err_stds.csv", err_σ_df)
CSV.write("experiments/ND/results/dis5d_data.csv", 
    DataFrame(data', ["x1", "x2", "x3", "x4", "x5", "y", "y_noisy"])
)
CSV.write("experiments/ND/results/dis5d_raw.csv", 
    DataFrame(data', ["x1", "x2", "x3", "x4", "x5", "y", "y_noisy"])
)
