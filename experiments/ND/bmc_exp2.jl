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
using CUDA
using Flux
using KernelFunctions
using AbstractGPs

########################################## SETUP ###########################################
global_rng = 2022
Random.seed!(global_rng)
n_full_data = 2500

################################## EXPERIMENT PARAMETERS ###################################
function bmc_func(x1, x2, x3, x4, x5)
    z = 10sin(π*x1*x2) + 20(x3 - 1/2) + 10x4 + 5x5
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
    :true_func => bmc_func,
    :opt_params => Dict([]),
    :opt_steps => 1000,
    :rng => global_rng,
])


###################################### GENERATE DATA #######################################
_X, _y, _y_noisy = generate_experimental_data(bmc_func, n_full_data, exp_params[:lb], exp_params[:ub], 0.0,  global_rng)
data = vcat(_X, _y', _y_noisy')

# quadrature solution
analytical = quadrature(bmc_func, exp_params[:lb], exp_params[:ub])  # 12.74
exp_params[:analytical_sol] = analytical

######################################## EXPERIMENT ########################################
# res = experiment(;exp_params...)
runs_per_n = 10
ns = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
res_means, res_stds, err_means, err_σ = exp_runs_over_n(ns, runs_per_n, exp_params)




print("hi")
# res = run_once(;exp_params...)
# runs_per_n = 5
# res_means, res_stds, err_means, err_σ = exp_runs_over_n([10, 25, 50, 100, 250, 500, 750, 1000], runs_per_n, run_once, exp_params)

######################################### RESULTS ##########################################
# # results
# nms = [
#     "n", "true", "quad", "mc", "qmc", "bq", "gbq_uni", "gbq_uni_m12", "gbq_uni_m32", "gbq_uni_m52",
#     "gbq_gauss", "gbq_gauss_m12", "gbq_gauss_m32", "gbq_gauss_m52",
# ]
# mean_df = DataFrame(res_means, nms)
# std_df = DataFrame(res_stds, nms)

# # pct error
# err_nms = [
#     "n", "quad", "mc", "qmc", "bq", "gbq_uni", "gbq_uni_m12", "gbq_uni_m32", "gbq_uni_m52",
#     "gbq_gauss", "gbq_gauss_m12", "gbq_gauss_m32", "gbq_gauss_m52"
# ]
# err_df = DataFrame(err_means, err_nms)
# err_σ_df = DataFrame(err_σ, err_nms)

# # plot
# # plot
# plot_labels = [
#     "quad" "mc" "qmc" "bq" "gbq_uni" "gbq_uni_m12" "gbq_uni_m32" "gbq_uni_m52" "gbq_gauss" "gbq_gauss_m12" "gbq_gauss_m32" "gbq_gauss_m52"
# ]

# # save
# # save_results(mean_df, std_df, err_df, runs_per_n, "experiments/ND/bmc_exp.hdf5")
# CSV.write("experiments/ND/results/bmc_exp_means.csv", mean_df)
# CSV.write("experiments/ND/results/bmc_exp_stds.csv", std_df)
# CSV.write("experiments/ND/results/bmc_exp_err_means.csv", err_df)
# CSV.write("experiments/ND/results/bmc_exp_err_stds.csv", err_σ_df)
# CSV.write("experiments/ND/results/bmc_exp_data.csv", DataFrame(data, ["x1", "x2", "x3", "x4", "x5", "y"]))

############################ rewrite QMC implementation
ff, fb, ffₖ, fbₖ, ff_m12, ff_m32, ff_m52 = generate_features(100, 5, rng)
thing1 = QuasiMonteCarlo.sample(100,[0],[2π], LatticeRuleSample())[1, :]
thing2 = QuasiMonteCarlo.sample(100,[0],[2π], LatticeRuleSample())[1, :]
thing1 == thing2


histogram(k_cpu.β)
histogram!(thing)

z1 = []
z2 = []
for i in 1:10
    zx = ϕ(Xv, k_cpu.ω, shuffle(k_cpu.β), k_cpu.σ, k_cpu.λ, false)
    est = zx' * zx ./ size(zx, 1)
    append!(z1, est)
    thing = QuasiMonteCarlo.sample(100,[0],[2π], LatticeRuleSample())[1, :]
    zx = ϕ(Xv, k_cpu.ω, thing, k_cpu.σ, k_cpu.λ, false)
    est = zx' * zx ./ size(zx, 1)
    append!(z2, est)
end


ff, fb, ffₖ, fbₖ, ff_m12, ff_m32, ff_m52 = generate_features(10000, 5, rng)
ϕ(Xv, ff, fb, ones(5), [1], false)' * ϕ(Xv, ff, fb, ones(5), [1], false) / 10000

# Current TODO
# 0. rewrite fourier feature generator to just pass in parametric function as sampling method and dist
# 1. Fix error regarding generation of evenly distributed fourier features - try out different pkgs, use entropy
# set up a test that runs 100 times under random seeds to see what works best
# 1.5 Make Problem non-random to allow for model training etc
# 2. Rewrite integration equation in parametric form for no sin feats/yes sin feats
# 3. Implement in vectorized manner

########################################### DEV ############################################
############################ generate toy data
function toydatafunc(x)
    return sin(sum(x))
end

X = rand(Uniform(), 5, 60)
X_test = rand(Uniform(), 5, 100)
Xv = X[:,1]
Xt = rand(Uniform(), 5, 20)
y = toydatafunc.(ColVecs(X))
y_test = toydatafunc.(ColVecs(X_test))
yv = y[1:1]
X_d = X |> gpu
Xv_d = Xv |> gpu
Xt_d = Xt |> gpu
y_d = y |> gpu
yv_d = yv |> gpu

# defining vars
ls = [1.0, 1.0, 1.0, 1.0, 1.0]


# draw FF
ff, fb, ffₖ, fbₖ, ff_m12, ff_m32, ff_m52 = generate_features(100, 5, global_rng)

# make rff measure distribution
rff_pₓ = RandomFeatureMvGaussian(ffₖ, fbₖ, μₓ, Σₓ, false)

# make kernels
k_cpu = RandomFeatureKernel{AbstractArray}(ff, fb, ls, [1.0], false)
k_gpu = RandomFeatureKernel{CuArray}(ff, fb, ls, [1.0], false)

# cosine and sine variants
k_cpu_cos = deepcopy(k_cpu)
k_cpu_cos.sin_feats = false
K_cos = kernelmatrix(k_cpu_cos, ColVecs(X))
K_cos = add_jitter(K_cos, 1e-7)
k_cpu_sin = deepcopy(k_cpu)
k_cpu_sin.sin_feats = true
K_sin = kernelmatrix(k_cpu_sin, ColVecs(X))
K_sin = add_jitter(K_sin, 1e-7)
# untrained
untrained_gp = build_vanilla_gp(rbf_untrained, X, log.([0.05]))

# trained
opt = ADAM()
rbf_trained = RBF{AbstractVector}(ls, [1.00001])
rbf_params = Flux.params(rbf_trained.σ)
# gp_opt_step!(opt, rbf_trained, rbf_params, y, log.([0.05]))
train_gp!(opt, rbf_trained, rbf_params, X, y, log.([0.05]), 5000)
trained_gp =  build_vanilla_gp(rbf_trained, X, log.([0.05]))


exp.(rbf_params[1])

-AbstractGPs.logpdf(untrained_gp, y)
-AbstractGPs.logpdf(trained_gp, y)
mae(y_test, gp_preds(untrained_gp, y, X_test)[1])
mae(y_test, gp_preds(trained_gp, y, X_test)[1])

k_cpu.σ = exp.(rbf_trained.σ)
k_cpu.λ = exp.(rbf_trained.λ)
gbq_uni = UniformGBQ(k_cpu, 0.05, lb, ub)
gbq_uni_sol_trained = gbq_uni(X, y)
k_cpu.σ = exp.(rbf_untrained.σ)
k_cpu.λ = exp.(rbf_untrained.λ)
gbq_uni = UniformGBQ(k_cpu, 0.05, lb, ub)
gbq_uni_sol_untrained = gbq_uni(X, y)

abs(quad_sol - gbq_uni_sol_untrained)
abs(quad_sol - gbq_uni_sol_trained)


px_uni = Product(Uniform.(lb, ub))
pₓ = MvNormal(μₓ, Σₓ)


# quad sol
function f_quad(x1, x2, x3, x4, x5)
    return sin(sum([x1, x2, x3, x4, x5]))
end
quad_sol = quadrature(f_quad, lb, ub)  # 0.4850647814093779

# mc uni sol
mc_z_uni = mc_quadrature_z(ColVecs(X), k_cpu, px_uni, lb, ub, 10000)
mc_z_uni2 = mc_quadrature_z(ColVecs(X), K_rbf, px_uni, lb, ub, 10000)
mc_uni_sol = mc_z_uni' * inv(K_sin) * y
mc_uni_sol2 = mc_z_uni2' * inv(kernelmatrix(K_rbf, ColVecs(X))) * y

# all bounds uni sol
function f_all_bounds(lb, ub, X, kernel)
    
    ω = kernel.ω ./ kernel.σ
    bounds, signs = integration_bounds_and_signs(lb, ub)
    
    # indef = indefinite_uniform_kme_func(lb, ub)
    function indefinite(x, x₀, ω)
        ω_prod = prod(ω, dims=1)[1, :]
        return sin.(ω' * (x .- x₀)) ./ ω_prod
    end

    function z_term(x₀)
        integration_terms = map(
            s -> indefinite(
                bounds[s], x₀, ω
            ) * signs[s],
            1:length(signs)
        )
        # return sum(integration_terms)
        return mean(sum(hcat(integration_terms...), dims=2))
    end

    z = map(
        i -> z_term(X[:, i]),
        1:size(X, 2)
    )
    return z
end

rff_uni_z = f_all_bounds(lb, ub, X, k_cpu)
rff_uni_z' * inv(K_sin) * y

# mc gauss
mc_z_gauss = mc_quadrature_z(ColVecs(X), k_cpu, pₓ, lb, ub, 10000)
mc_gauss_sol = mc_z_gauss' * inv(K_sin) * y
