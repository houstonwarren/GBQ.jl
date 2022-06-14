
using Distributions

######################################### STRUCTS ##########################################

################# UNIVARIATE #################
mutable struct RandomFeatureGaussian
    ρ::AbstractVector  # fourier features
    β::AbstractVector
    μ::Float64  # distribution mean
    σ  # distribution sd
    sin_feats::Bool
end

function Distributions.pdf(d::RandomFeatureGaussian, x::Float64)
    delta = (x - d.μ) / d.σ
    p = mean(cos.(d.ρ * delta)) / sqrt(2π * d.σ^2)
    return p
end

function Distributions.cdf(d::RandomFeatureGaussian, x::Float64)
    delta = (x - d.μ) / d.σ
    rff_terms = d.σ .* sin.(d.ρ .* delta) ./ d.ρ
    P = mean(rff_terms) / sqrt(2π * d.σ^2) + 0.5
    return P
end

################ MULTIVARIATE ################
mutable struct RandomFeatureMvGaussian
    ρ::AbstractMatrix  # fourier features
    β::AbstractVector  # bias
    μ::AbstractVector  # distribution mean
    Σ::AbstractMatrix  # distribution co-variance
    sin_feats::Bool
end

function Distributions.pdf(d::RandomFeatureMvGaussian, x::AbstractVector)
    n_dim = length(d.μ)
    # scale
    ff = d.ρ ./ sqrt.(diag(d.Σ))

    # run through rff
    delta = x .- d.μ
    
    if d.sin_feats
        cos_sin =  vcat(cos.(ff' * delta), sin.(ff' * delta))
        p = mean(cos_sin) / (sqrt(2π)^n_dim * sqrt(det(d.Σ)))
    else
        p = mean(cos.(ff' * delta)) / (sqrt(2π)^n_dim * sqrt(det(d.Σ)))
    end
    return p
end

function Distributions.cdf(p::RandomFeatureMvGaussian, lb, ub)
    d = length(lb)
    ρ = p.ρ
    σ = sqrt.(diag(p.Σ))
    k = RandomFeatureKernel{AbstractArray}(ρ, p.β, σ, [1.0], p.sin_feats)  # this currently isn't ideal, need variable type and sin feats

    integral = uniform_rff_kernel_mean(p.μ, k, lb, ub, k.sin_feats)[1]
    integral = integral / (sqrt(2π)^d * sqrt(det(p.Σ)))

    return integral
end
