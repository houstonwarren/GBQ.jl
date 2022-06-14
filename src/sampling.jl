######################################## PACKAGING #########################################
using Distributions
using HaltonSequences
using Random
using QuasiMonteCarlo

######################################### HELPERS ##########################################
function piecewise_mv_cdf(dist, lb, ub)
    μ = dist.μ
    σ = sqrt.(dist.Σ)
    n_dims = length(lb)
    dists = [Normal(μ[i], σ[i, i]) for i in 1:n_dims]
    cdfs = [cdf(dists[i], ub[i]) - cdf(dists[i], lb[i]) for i in 1:n_dims]
    trunc_val = prod(cdfs)
    return trunc_val
end

######################################## FUNCTIONS #########################################
################## GAUSSIAN ##################
function sample_uniform_grid(n_samples, lb, ub)
    n_dims = length(lb)
    grid_uniform = rand(n_samples, n_dims)
    dim_delta = abs.(ub .- lb)
    deltas = grid_uniform .* dim_delta'
    grid = lb' .+ deltas
    return grid
end

function halton_sampler(n, dim, rng=nothing)
    uniform_samples = hcat(collect(HaltonPoint(dim, length=n))...)
    
    if isnothing(rng)
        row_shuffled_samples = uniform_samples[shuffle(1:dim), :]
    else
        row_shuffled_samples = uniform_samples[shuffle(MersenneTwister(rng), 1:dim), :]
    end

    return row_shuffled_samples
end
  
function inv_norm_cdf(uniform_samples::Vector)
    return quantile.(Normal(), uniform_samples)
end

function inv_norm_cdf(uniform_samples::Matrix)    
    return Array(hcat(map(i -> quantile.(Normal(), i), RowVecs(uniform_samples))...)')
end

function qmc_gauss_fourier_features(halton_samples)
    dist_size = size(halton_samples, 1) - 1
    bias_col = size(halton_samples, 1)
    ff = inv_norm_cdf(halton_samples[1:dist_size, :])
    fb = halton_samples[bias_col, :] .* 2π
    return ff, fb
end

################### MATERN ###################    
function qmc_matern_12_ff(halton_samples)
    ff = tan.(π * (halton_samples .- 0.5))
    ff = ifelse.(ff .== 0.0, 1e-2, ff)
end

function qmc_matern_32_ff(halton_samples)
    ff = (2 .* halton_samples .- 1) ./ sqrt.(2 .* halton_samples .* (1 .- halton_samples))
    ff = ifelse.(ff .== 0.0, 1e-2, ff)
end

function qmc_matern_52_ff(halton_samples)
    α = 4 .* halton_samples .* (1 .- halton_samples)
    ff = 4 .* cos.(acos.(sqrt.(α)) ./ 3) ./ sqrt.(α)
    ff = sign.(halton_samples .- 0.5) .* sqrt.(ff .- 4)
    ff = ifelse.(ff .== 0.0, 1e-2, ff)
end

########################################### ALL ############################################
function generate_features(n_feats, dim, rng)
    # generate halton samples
    n_rows = (dim + 1) * 2
    split_point = Int64(n_rows / 2)
    halton_samples = halton_sampler(n_feats, n_rows, rng)

    # generate gaussian features
    ff, fb = qmc_gauss_fourier_features(halton_samples[1:split_point, :])
    ff = ifelse.(ff .== 0.0, 1e-2, ff)
    ffₖ, fbₖ = qmc_gauss_fourier_features(halton_samples[split_point+1:n_rows, :])
    ffₖ = ifelse.(ffₖ .== 0.0, 1e-2, ffₖ)

    # matern features
    ff_m12 = qmc_matern_12_ff(halton_samples[1:dim, :])
    ff_m32 = qmc_matern_32_ff(halton_samples[1:dim, :])
    ff_m52 = qmc_matern_52_ff(halton_samples[1:dim, :])
    return ff, fb, ffₖ, fbₖ, ff_m12, ff_m32, ff_m52
end
