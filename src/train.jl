using Flux
using CUDA
using Distributions
using KernelFunctions
using AbstractGPs
using Plots
using QuasiMonteCarlo
using FluxOptTools
using Random
using LinearAlgebra

################################ DATA GENERATION AND SPLITS ################################
function generate_data(f, n, lb, ub, noise_sd=0.0, rng=nothing)
    X, y = f(n, lb, ub, rng)

    if noise_sd > 0.0
        if isnothing(rng)
            y = y .+ rand(Normal(0, noise_sd), n)
        else
            y = y .+ rand(MersenneTwister(rng), Normal(0, noise_sd), n)
        end
    end

    return X, y
end

function partition_data(X, y, n_train, rng, noise_sd=0.0)
    # full data
    
    # train/test data
    train_pct = n_train / size(X, 1)
    X_train, X_test = MLJ.partition(X, train_pct, rng=rng, shuffle=true)
    y_train, y_test = MLJ.partition(y, train_pct, rng=rng, shuffle=true)

    if noise_sd > 0.0
        y_train = y_train .+ rand(Normal(0, noise_sd), n_train)
    end

    return X_train, y_train, X_test, y_test
end

################################ KERNEL TRAINING FUNCTIONS #################################
# general process
# 1. Create object (using deepcopy)
    # current problem here is that the underlying params of the model are not existing outside scope
# 2. create parameters object that has the ones you want to opt over
# 3. create inline function which takes as arguments input/output and doesn't include model vars as input
# 4. run gradient according to the params on that loss function 
# 5. update optimizer and param string_format_mean_sd
# 6. return updated object?

############ VANILLA GP FUNCTIONS #############
mutable struct RBF{T} <: KernelFunctions.Kernel
    σ::T  # vector of kernel LS - this is in squared form
    λ::T  # vector of kernel variance
    RBF{T}(σ, λ) where {T} = new(deepcopy(log.(σ)), deepcopy(log.(λ)))
end

function build_vanilla_gp(k, X, noise_sd::AbstractVector)  # note - expects positive arguments in log form
    k_actual = exp(k.λ[1]) * (SqExponentialKernel() ∘ ARDTransform(exp.(k.σ)))
    rbf_gp_prior = GP(k_actual)
    fx = AbstractGPs.FiniteGP(rbf_gp_prior, ColVecs(X), exp.(noise_sd[1])^2)
    return fx
end

function gp_loss(k::RBF, X, y, noise_sd)
    fx = build_vanilla_gp(k, X, noise_sd)
    nll = -logpdf(fx, y)
    return nll
end

############## RFF GP FUNCTIONS ###############
function build_rff_gp(k, X, noise_std::AbstractVector)  # note - expects positive arguments in log form
    gpprior = GP(k)
    fx = AbstractGPs.FiniteGP(gpprior, ColVecs(X), noise_std[1]^2)  # Prior at the observations
    return fx
end

function gp_loss(k::RandomFeatureKernel, X, y, noise_sd)
    fx = build_rff_gp(k, X, noise_sd)
    nll = -logpdf(fx, y)
    return nll
end

########## JOINT TRAINING FUNCTIONS ##########
function gp_opt_step!(opt, k, trainable_params, X, y, noise_sd)
    # with updated values, make sure loss function is recalculated
    grads = Flux.gradient(() -> gp_loss(k, X, y, noise_sd), trainable_params)
    Flux.update!(opt, trainable_params, grads)
end

function train_gp!(opt, k, trainable_params, X, y, noise_sd, steps)  # this will update k and prms IN-PLACE OUTSIDE SCOPE!
    for step in 1:steps
        gp_opt_step!(opt, k, trainable_params, X, y, noise_sd)
    end
    return
end

function gp_preds(gp, y_train, x_test)
    posterior_gp = posterior(gp, y_train)
    fs = marginals(posterior_gp(ColVecs(x_test)))
    return mean.(fs), std.(fs)
end

# function train_gp_lbfgs!(loss_func, y, trainable_params, steps)  # this will update k and prms IN-PLACE OUTSIDE SCOPE!
#     gp_loss() = -AbstractGPs.logpdf(finite_gp, y)
#     Zygote.refresh()
#     lossfun, gradfun, fg!, p0 = optfuns(gp_loss, prms)
#     res = Optim.optimize(Optim.only_fg!(fg!), p0, LBFGS(), Optim.Options(iterations=steps, store_trace=true))
# end

################################### EXPERIMENT FUNCTION ####################################
function experiment(;
    n_train, n_fourier_feats,
    lb, ub,
    jitter_val, noise_sd,
    ls_init, λ_init, sin_feats,  # kernel params init
    μₓ, Σₓ,  # measure distribution init
    analytical_sol,
    trainable_params_func,  # func that takes k and returns tuple of trainable params
    true_func, data_gen_func,
    opt_params, opt_steps,
    rng)
    
    ############### GENERATE DATA ################
    X_train, y_train = generate_data(data_gen_func, n_train, lb, ub, noise_sd[1], rng)

    ############# GENERATE FEATURES ##############
    dim = length(lb)
    ff, fb, ffₖ, fbₖ, ff_m12, ff_m32, ff_m52 = generate_features(n_fourier_feats, dim, rng)

    ######### TRAIN KERNEL HYPER-PARAMS ##########
    opt = ADAM(;opt_params...)
    rbf_training = RBF{AbstractVector}(ls_init, λ_init)
    # rbf_training_params = Flux.params(trainable_params_func(rbf_training))
    rbf_training_params = Flux.params(rbf_training.σ)
    train_gp!(opt, rbf_training, rbf_training_params, X_train, y_train, log.(noise_sd), opt_steps)
    ls_trained = deepcopy(exp.(rbf_training))  # finalized values

    ############### CREATE KERNELS ###############
    # baseline rbf
    k_rbf = λ_init[1] * (SqExponentialKernel() ∘ ARDTransform(ls_trained))
    K_rbf = kernelmatrix(k_rbf, ColVecs(X_train))
    K_rbf = add_jitter(K_rbf, jitter_val)
    noisy_K_rbf = K_rbf + I * noise_sd[1]^2
    
    # gaussian rff
    rffk = RandomFeatureKernel{AbstractVector}(ff, fb, ls_trained, λ_init, sin_feats)

    # matern
    rff_m12 = RandomFeatureKernel(ff_m12, fb, ls_trained, λ_init, sin_feats)
    rff_m32 = RandomFeatureKernel(ff_m32, fb, ls_trained, λ_init, sin_feats)
    rff_m52 = RandomFeatureKernel(ff_m52, fb, ls_trained, λ_init, sin_feats)

    # # parametric
    # rff_parametric = RandomFeatureKernel(ff, fb, ls_init, λ_init, sin_feats)
    # rff_parametric_gp = build_rff_gp(rff_parametric, X_train, noise_sd)

    ################# BASELINES ##################
    ### quadrature
    quad_est = quadrature(true_func, lb, ub, noise_sd[1], n_train)

    ### mc integration
    # mc_est, mc_sd = mc_quadrature(bmc_func, lb, ub, n_train, noise_sd, rng)

    # quasi mc integration
    # qmc_est, qmc_sd = mc_quadrature_with_data(bmc_func, lb, ub, X_train, noise_sd, rng)

    ## bq
    pₓ = MvNormal(μₓ, diagm(Σₓ))
    # bq_est = bayesian_quadrature(X_train, diagm(ls).^2, pₓ, noisy_K_rbf, y_train, lb, ub)

    #################### gbq #####################
    ### uniform
    gbq_uni_rff = UniformGBQ(rffk, noise_sd[1], lb, ub)(X_train, y_train)
    gbq_uni_m12 = UniformGBQ(rff_m12, noise_sd[1], lb, ub)(X_train, y_train)
    gbq_uni_m32 = UniformGBQ(rff_m32, noise_sd[1], lb, ub)(X_train, y_train)
    gbq_uni_m52 = UniformGBQ(rff_m52, noise_sd[1], lb, ub)(X_train, y_train)

    ### gaussian
    rff_pₓ = RandomFeatureMvGaussian(ffₖ, fbₖ, pₓ.μ, Matrix(pₓ.Σ), sin_feats)
    gbq_gauss = GaussianGBQ(rffk, rff_pₓ, noise_sd[1])(X_train, y_train, lb, ub)
    gbq_m12 = GaussianGBQ(rff_m12, rff_pₓ, noise_sd[1])(X_train, y_train, lb, ub)
    gbq_m32 = GaussianGBQ(rff_m32, rff_pₓ, noise_sd[1])(X_train, y_train, lb, ub)
    gbq_m52 = GaussianGBQ(rff_m52, rff_pₓ, noise_sd[1])(X_train, y_train, lb, ub)
    
    ################## RESULTS ###################
    # results = [
    #     analytical_sol, quad_est, mc_est, qmc_est, bq_est,
    #     gbq_uni_rff, gbq_uni_m12, gbq_uni_m32, gbq_uni_m52,
    #     gbq_gauss, gbq_m12, gbq_m32, gbq_m52
    # ]
    results = [
        analytical_sol,
        gbq_uni_rff, gbq_uni_m12, gbq_uni_m32, gbq_uni_m52,
        gbq_gauss, gbq_m12, gbq_m32, gbq_m52
    ]
    print(results, "\n")
    return results

end

################################### REPEATED EXPERIMENTS ###################################
function exp_repeated_runs(n_reps, exp_function, params, verbose=false)
    rngs = rand(MersenneTwister(params[:rng]), 1:100, n_reps)
    run_params = copy(params)
    
    full_results = []
    for run in 1:n_reps
        if verbose
            print(run, "\n")
        end
        run_params[:rng] = rngs[run]
        results = exp_function(;run_params...)
        push!(full_results, results)
    end

    full_results = convert.(Float64, Array(hcat(full_results...)'))
    return full_results
end


############################## REPEATED EXPERIMENTS ACROSS N ###############################
function exp_runs_over_n(ns, n_reps, exp_function, params, verbose=false)
    n_res_means = []
    n_res_stds = []
    n_res_errs = []
    n_res_errs_σ = []
    for n in ns
        print(n, "\n")
        
        # update params
        params_n = copy(params)
        params_n[:n_train] = n
        
        # run
        n_res = exp_repeated_runs(n_reps, exp_function, params_n, verbose)
        
        # values
        n_res_mean = mean(n_res, dims=1)[1, :]
        n_res_std = std(n_res, dims=1)[1, :]

        # errors
        true_val = n_res[1, 1]
        preds = n_res[:, 2:size(n_res, 2)]
        errs = pct_error(true_val, preds)
        err_means = mean(errs, dims=1)[1, :]
        err_stds = std(errs, dims=1)[1, :]

        # add to df
        push!(n_res_means, vcat(n, n_res_mean))
        push!(n_res_stds, vcat(n, n_res_std))
        push!(n_res_errs, vcat(n, err_means))
        push!(n_res_errs_σ, vcat(n, err_stds))

    end
    n_res_means = Array(hcat(n_res_means...)')
    n_res_stds = Array(hcat(n_res_stds...)')
    n_res_errs = Array(hcat(n_res_errs...)')
    n_res_errs_σ = Array(hcat(n_res_errs_σ...)')
    return n_res_means, n_res_stds, n_res_errs, n_res_errs_σ
end

