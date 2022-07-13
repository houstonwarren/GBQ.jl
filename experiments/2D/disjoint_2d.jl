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
function disjoint_poly_2d(x, y)  # https://www.sfu.ca/~ssurjano/disc.html
    if x > 0.5 || y > 0.5
        return 0
    else
        return exp(5x + 5y)
    end
end

function trainable_prms(k)
    return k.σ
end

exp_params = Dict([
    :n_train => 100,
    :n_fourier_feats => 100,
    :lb => [0.0, 0.0],
    :ub => [1.0, 1.0],
    :jitter_val => 1e-7,
    :λ_init => [1.000001],
    :ls_init => [0.5, 0.5],  # kernel variance
    :noise_sd => [0.1],
    :sin_feats => false,
    :μₓ => [0.5, 0.5],
    :Σₓ => [0.5, 0.5],  # covariance, not sd
    :trainable_params_func => trainable_prms,
    :true_func => p2d,
    :opt_params => Dict([]),
    :opt_steps => 1000,
    :rng => global_rng,
])


###################################### GENERATE DATA #######################################
_X, _y, _y_noisy = generate_experimental_data(disjoint_poly_2d, n_full_data, exp_params[:lb], exp_params[:ub], 0.0,  global_rng)
data = vcat(_X, _y', _y_noisy')

# plot
surface(0:0.01:1, 0:0.01:1, disjoint_poly_2d)

# quadrature solution
quad_sol = quadrature(disjoint_poly_2d, exp_params[:lb], exp_params[:ub] ./ 2)  # 5.00

# analytical solution
@syms x_ana, y_ana
expr = exp(5x_ana + 5y_ana)
analytical = integrate(
    expr, 
    (x_ana, exp_params[:lb][1], exp_params[:ub][1] / 2),
    (y_ana, exp_params[:lb][2], exp_params[:ub][2] / 2)
)  # 5.00
exp_params[:analytical_sol] = analytical

######################################## EXPERIMENT ########################################
# res = experiment(;exp_params...)
runs_per_n = 10
ns = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
res_means, res_stds, err_means, err_σ = exp_runs_over_n(ns, runs_per_n, exp_params)

######################################### RESULTS ##########################################
# results
nms = [
    "n", "true", "quad", "mc", "qmc", "bq", "sbq_uni", "sbq_uni_m12", "sbq_uni_m32", "sbq_uni_m52",
    "sbq_gauss", "sbq_gauss_m12", "sbq_gauss_m32", "sbq_gauss_m52",
]
mean_df = DataFrame(res_means, nms)
std_df = DataFrame(res_stds, nms)

# pct error
err_nms = [
    "n", "quad", "mc", "qmc", "bq", "sbq_uni", "sbq_uni_m12", "sbq_uni_m32", "sbq_uni_m52",
    "sbq_gauss", "sbq_gauss_m12", "sbq_gauss_m32", "sbq_gauss_m52"
]
err_df = DataFrame(err_means, err_nms)
err_σ_df = DataFrame(err_σ, err_nms)

# plot
plot_labels = [
    "quad" "mc" "qmc" "bq" "sbq_uni" "sbq_uni_m12" "sbq_uni_m32" "sbq_uni_m52" "sbq_gauss" "sbq_gauss_m12" "sbq_gauss_m32" "sbq_gauss_m52"
]

# save
# save_results(mean_df, std_df, err_df, runs_per_n, "experiments/2D/results/p2d.hdf5")
CSV.write("experiments/2D/results/disjoint_2d_means.csv", mean_df)
CSV.write("experiments/2D/results/disjoint_2d_stds.csv", std_df)
CSV.write("experiments/2D/results/disjoint_2d_err_means.csv", err_df)
CSV.write("experiments/2D/results/disjoint_2d_err_stds.csv", err_σ_df)
CSV.write("experiments/2D/results/disjoint_2d_data.csv", DataFrame(data', ["x1", "x2", "y", "y_noisy"]))
