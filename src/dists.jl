
using Distributions

######################################### STRUCTS ##########################################

################# UNIVARIATE #################
mutable struct RandomFeatureGaussian
    ρ::Vector  # fourier features
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
    ρ::Matrix  # fourier features
    μ::Vector  # distribution mean
    Σ::Matrix  # distribution co-variance
    sin_feats::Bool
end

function Distributions.pdf(d::RandomFeatureMvGaussian, x::Vector)
    n_dim = length(x)
    # scale
    ff = d.ρ ./ sqrt.(diag(d.Σ)')

    # run through rff
    delta = x - d.μ
    p = mean(cos.(ff * delta)) / (sqrt(2π)^n_dim * sqrt(det(d.Σ)))
    return p
end

function Distributions.cdf(d::RandomFeatureMvGaussian, lb, ub)
    n_dim = length(lb)
    indefinite_integral = indefinite_uniform_kme_func(lb, ub)
    ff = d.ρ ./ sqrt.(diag(d.Σ)')
    # ff = d.ρ ./ diag(d.Σ)'
    integral = definite_integral_uni(indefinite_integral, d.μ, ff, lb, ub) /  (sqrt(2π)^n_dim * sqrt(det(d.Σ)))
    return integral
end
