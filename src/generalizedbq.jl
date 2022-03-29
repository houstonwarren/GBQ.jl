######################################## PACKAGING #########################################
using Distributions
using KernelFunctions

##################################### HELPER FUNCTIONS #####################################
########## GENERAL INTEGRATION FUNS ##########
function n_ints_by_dim(lb, ub)
    # calculate the number of integrations over each dimension implied by vectors of bounds
    delta = ub - lb
    [ifelse.(abs.(delta) .> 0, 1, 0)]
end

function n_ints_by_dim(lb::Vector, ub::Vector)
    # calculate the number of integrations over each dimension implied by vectors of bounds
    delta = ub .- lb
    ifelse.(abs.(  delta) .> 0, 1, 0)
end

function integration_bounds_and_signs(lb, ub)
    d = size(lb, 2)
    integrations_by_dim = n_ints_by_dim(lb, ub)

    # add signs to matrix
    negs = [integrations_by_dim[i] > 0 ? -1 : 1 for i in 1:d]
    pos = [integrations_by_dim[i] > 0 ? 1 : 1 for i in 1:d]

    # create possible choices and all permutations of bounds
    choices = [vcat.(lb, negs) vcat.(ub, pos)]
    combos = reshape(collect(Iterators.product([choices[i, :] for i in 1:size(choices, 1)]...)), :)
    combos = unique(combos)

    # calculate bounds of each integral as well as the sign
    bounds = collect.(map.(z -> z[1], combos))
    signs = prod.(map.(z -> z[2], combos))
    return bounds, signs
end

function definite_integral_uni(indefinite_int_fun, x₀, ff, lb, ub)
    bounds, signs = integration_bounds_and_signs(lb, ub)
    
    integration_terms = map(
        s -> indefinite_int_fun(
            bounds[s], x₀, ff
        ) * signs[s],
        1:length(signs)
    )  
    return sum(integration_terms)
end

function definite_integral_gauss(indefinite_int_fun, x₀, ff, lb, ub)
    bounds, signs = integration_bounds_and_signs(lb, ub)
    
    integration_terms = map(
        s -> indefinite_int_fun(
            bounds[s], x₀, ff
        ) * signs[s],
        1:length(signs)
    )  
    return sum(integration_terms)
end

########## TRIG FUNCTION INTEGRALS ###########
# uniform KME with RFF
function indefinite_uniform_kme_func(lb, ub)
     # take in the bounds and return a function that can calculate the integral value at a point (with gaussian sampling p(x))
    # calculate the number of integrations being performed
    n_ints_vec = n_ints_by_dim(lb, ub)
    total_integrations = sum(n_ints_vec)
    if total_integrations == 0
        return error("Not an integral")
    end

    # choose which trig function is used based on n_ints
    cos_trig_signs = [z -> cos(z), z -> sin(z), z -> -cos(z), z -> -sin(z)]
    trig_sign_index = ifelse(total_integrations < 4, total_integrations, mod(total_integrations, 4)) + 1
    cos_trig_func = cos_trig_signs[trig_sign_index]

    # if sin_feats, calculate integral of sin term as well
    # sin_trig_signs = [z -> sin(z), z -> -cos(z), z -> -sin(z), z -> cos(z)]
    # sin_trig_func = sin_trig_signs[trig_sign_index]

    # create variable functions, for single dimensional observation and multi-dim case
    # single dim
    function indefinite_integral(xₛ::Float64, x₀::Float64, ω::Vector)
        α = ω * (x₀ - xₛ)
        out = cos_trig_func.(α) ./ ω
        out = mean(out)
        return out
    end

    # multi-dim
    function indefinite_integral(xₛ::Vector, x₀::Vector, ω::Matrix)
        # if sin_feats
        #     t1 = sin_trig_func(xₛ' * (ω .+ ρ) - ω'*x₀ - ρ'*μ)
        #     t2 = sin_trig_func(xₛ' * (ω .- ρ) - ω'*x₀ + ρ'*μ)
        # else
        #     t1 = cos_trig_func(xₛ' * (ω .+ ρ) - ω'*x₀ - ρ'*μ)
        #     t2 = cos_trig_func(xₛ' * (ω .- ρ) - ω'*x₀ + ρ'*μ)
        # end
        # t1 = t1 / (2 * prod((ω .+ ρ)))
        # t2 = t2 / (2 * prod((ω .- ρ)))

        α = ω * (x₀ .- xₛ)
        out = cos_trig_func.(α) ./ prod(ω, dims=2)
        out = mean(out)
        return out
    end

    return indefinite_integral
end

# Gaussian KME with RFF
function indefinite_gaussian_kme_func_pt(lb, ub)       
    # take in the bounds and return a function that can calculate the integral value at a point (with gaussian sampling p(x))
    # calculate the number of integrations being performed
    n_ints_vec = n_ints_by_dim(lb, ub)
    total_integrations = sum(n_ints_vec)
    if total_integrations == 0
        return error("Not an integral")
    end

    # choose which trig function is used based on n_ints
    cos_trig_signs = [z -> cos(z), z -> sin(z), z -> -cos(z), z -> -sin(z)]
    trig_sign_index = ifelse(total_integrations < 4, total_integrations, mod(total_integrations, 4)) + 1
    cos_trig_func = cos_trig_signs[trig_sign_index]

    # # if sin_feats, calculate integral of sin term as well
    # sin_trig_signs = [z -> sin(z), z -> -cos(z), z -> -sin(z), z -> cos(z)]
    # sin_trig_func = sin_trig_signs[trig_sign_index]

    # create variable functions, for single dimensional observation and multi-dim case
    function indefinite_integral(xₛ::Float64, x₀::Float64, ω::Float64, ρ::Float64, μ::Float64)
        
        # if sin_feats
        #     t1 = sin_trig_func(xₛ * (ω + ρ) - ω*x₀ - ρ*μ)
        #     t2 = sin_trig_func(xₛ * (ω - ρ) - ω*x₀ + ρ*μ)
        # else
        #     t1 = cos_trig_func(xₛ * (ω + ρ) - ω*x₀ - ρ*μ)
        #     t2 = cos_trig_func(xₛ * (ω - ρ) - ω*x₀ + ρ*μ)
        # end
        α = ω * (xₛ - x₀)
        β = ρ * (xₛ - μ)
        t1 = cos_trig_func.(α + β) ./ (2 * (ω .+ ρ))
        t2 = cos_trig_func.(α - β) ./ (2 * (ω .- ρ))
        return t1 + t2
    end

    # multi-dim
    function indefinite_integral(xₛ::Vector, x₀::Vector, ω::Vector, ρ::Vector, μ::Vector)
        # if sin_feats
        #     t1 = sin_trig_func(xₛ' * (ω .+ ρ) - ω'*x₀ - ρ'*μ)
        #     t2 = sin_trig_func(xₛ' * (ω .- ρ) - ω'*x₀ + ρ'*μ)
        # else
        #     t1 = cos_trig_func(xₛ' * (ω .+ ρ) - ω'*x₀ - ρ'*μ)
        #     t2 = cos_trig_func(xₛ' * (ω .- ρ) - ω'*x₀ + ρ'*μ)
        # end
        # t1 = t1 / (2 * prod((ω .+ ρ)))
        # t2 = t2 / (2 * prod((ω .- ρ)))
        α = ω' * (xₛ - x₀)
        β = ρ' * (xₛ - μ)
        t1 = cos_trig_func(α .+ β) / (2 * prod(ω .+ ρ))
        t2 = cos_trig_func(α .- β) / (2 * prod(ω .- ρ))
        return t1 + t2
    end

    return indefinite_integral
end


######################################### UNIFORM ##########################################
##################### 1D #####################
function gbq_uni_kme_1d(rff_k, x, lb, ub)
    # apply the integral to upper
    ff = rff_k.ω ./ rff_k.σ
    upper = sin.(ff * (ub - x)) ./ ff

    # apply to lower
    lower = sin.(ff * (lb - x)) ./ ff

    return mean(upper .- lower)
end

function gbq_uni_1d_μ(rff_k, X, K, y, lb, ub)
    z = map(
        i -> gbq_uni_kme_1d(rff_k, X[i], lb, ub),
        1:size(X, 1)
    )
    return z' * inv(K) * y
end

function gbq_uni_1d_σ(rff_k, lb, ub)
    1
end

function gbq_uni_1d(rff_k, X, K, y, lb, ub)
    μ = gbq_uni_1d_μ(rff_k, X, K, y, lb, ub)
    σ = gbq_uni_1d_σ(rff_k, lb, ub)
    return (μ, σ)
end

##################### ND #####################
function gbq_uni_kme(x, ff, lb::Vector, ub::Vector)
    indefinite_integral = indefinite_uniform_kme_func(lb, ub)
    integral = definite_integral_uni(indefinite_integral, x, ff, lb, ub)
    return integral
end

function gbq_uni_nd_μ(rff_k, X, K, y, lb, ub)
    ff = rff_k.ω ./ rff_k.σ'
    z = map(
        i -> gbq_uni_kme(X[i, :], ff, lb, ub),
        1:size(X, 1)
    )
    return z' * inv(K) * y
end

function gbq_uni_nd_σ(rff_k, lb, ub)
    1
end

function gbq_uni_nd(rff_k, X, K, y, lb, ub)
    μ = gbq_uni_nd_μ(rff_k, X, K, y, lb, ub)
    σ = gbq_uni_nd_σ(rff_k, lb, ub)
    return (μ, σ)
end

######################################### GAUSSIAN #########################################
##################### 1D #####################
function gbq_gauss_kme_1d_pt(xₛ, x₀, ω, ρ, μ)
    # calculate the integral value for a single pair of feature points
    α = ω * (xₛ - x₀)
    β = ρ * (xₛ - μ)
    t1 = sin(α + β) / (2 * (ω + ρ))
    t2 = sin(α - β) / (2 * (ω - ρ))
    return t1 + t2
end

function gbq_gauss_kme_1d(rff_k, rff_px, x, lb, ub)
    # transform fourier features by lengthscale and sigma
    Ω = rff_k.ω ./ rff_k.σ
    Ρ = rff_px.ρ ./ rff_px.σ

    # truncation term to normalize bounded expectation over p(x)
    trunc_term = cdf(rff_px, ub) - cdf(rff_px, lb)

    # upper 
    # map over the kernel fourier features and measure (p(x)) fourier features
    top = hcat(map(
        j -> map(
            k -> gbq_gauss_kme_1d_pt(ub, x, Ω[j], Ρ[k], rff_px.μ), 1:size(Ρ, 1)
        ), 1:size(Ω, 1)
    )...)
    tbound = mean(top) / sqrt(2π * rff_px.σ^2)
    tbound = tbound / trunc_term

    # lower
    # do the same
    bot = hcat(map(
        j -> map(
            k -> gbq_gauss_kme_1d_pt(lb, x, Ω[j], Ρ[k], rff_px.μ), 1:size(Ρ, 1)
        ), 1:size(Ω, 1)
    )...)
    bbound = mean(bot) / sqrt(2π * rff_px.σ^2)
    bbound = bbound / trunc_term

    return (tbound - bbound) * (ub - lb)
end

function gbq_gauss_μ_1d(rff_k, rff_px, X, K, y, lb, ub)
    # map over all x to produce z term
    z = map(
        i -> gbq_gauss_kme_1d(rff_k, rff_px, X[i], lb, ub),
        1:size(X, 1)
    )
    # final BQ formula
    return z' * inv(K) * y
end

function gbq_gauss_σ_1d(rff_k, rff_px, lb, ub)
    1
end

function gbq_gauss_1d(rff_k, rff_px, X, K, y, lb, ub)
    μ = gbq_gauss_μ_1d(rff_k, rff_px, X, K, y, lb, ub)
    σ = gbq_gauss_σ_1d(rff_k, rff_px, lb, ub)
    return (μ, σ)
end

##################### ND #####################
function definite_integral_gauss(indefinite_int_fun, x₀, lb, ub)
    bounds, signs = integration_bounds_and_signs(lb, ub)
    
    integration_terms = map(
        s -> indefinite_int_fun(
            bounds[s], x₀
        ) * signs[s],
        1:length(signs)
    )  
    return sum(integration_terms)
end

function gbq_gauss_kme(rff_k, rff_px, x::Vector, lb::Vector, ub::Vector)
    n_dim = length(lb)
    
    # transform fourier features by lengthscale and sigma
    Ω = rff_k.ω ./ rff_k.σ'
    Ρ = rff_px.ρ ./ sqrt.(diag(rff_px.Σ)')

    # calculate truncation term
    trunc_term = cdf(rff_px, lb, ub)

    # calculate indefinite integral and make function to run over all combinations of ω and ρ
    indefinite_integral = indefinite_gaussian_kme_func_pt(lb, ub)

    function indefinite_gaussian_kme_func(xₛ::Vector, x₀::Vector)
        integrated = hcat(map(
            j -> map(
                k -> indefinite_integral(xₛ, x₀, Ω[j, :], Ρ[k, :], rff_px.μ), 1:size(Ρ, 1)
            ), 1:size(Ω, 1)
        )...)
        tbound = mean(integrated) / (sqrt(2π)^n_dim * sqrt(det(rff_px.Σ)))
        tbound = tbound / trunc_term * prod(ub .- lb)
    end

    # apply to definite integral
    integral = definite_integral_gauss(indefinite_gaussian_kme_func, x, lb, ub)

    return integral
end

function gbq_gauss_μ_nd(rff_k, rff_px, X, K, y, lb, ub)
    # map over all x to produce z terms
    z = map(
        i -> gbq_gauss_kme(rff_k, rff_px, X[i, :], lb, ub),
        1:size(X, 1)
    )
    # final BQ formula
    return z' * inv(K) * y
end

function gbq_gauss_σ_nd(rff_k, rff_px, lb, ub)
    1
end

function gbq_gauss_nd(rff_k, rff_px, X, K, y, lb, ub)
    μ = gbq_gauss_μ_nd(rff_k, rff_px, X, K, y, lb, ub)
    σ = gbq_gauss_σ_nd(rff_k, rff_px, lb, ub)
    return (μ, σ)
end
