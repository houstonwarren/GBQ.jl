######################################## PACKAGING #########################################
using Distributions
using KernelFunctions
using LinearAlgebra

##################################### HELPER FUNCTIONS #####################################
########## GENERAL INTEGRATION FUNS ##########
function n_ints_by_dim(lb::AbstractVector, ub::AbstractVector)
    # calculate the number of integrations over each dimension implied by vectors of bounds
    delta = ub .- lb
    ifelse.(abs.(delta) .> 0, 1, 0)
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

# function definite_integral_gauss(indefinite_int_fun, x₀, ff, lb, ub)
#     bounds, signs = integration_bounds_and_signs(lb, ub)
    
#     integration_terms = map(
#         s -> indefinite_int_fun(
#             bounds[s], x₀, ff
#         ) * signs[s],
#         1:length(signs)
#     )  
#     return sum(integration_terms)
# end

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
        α = ω * (xₛ - x₀)
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

        α = ω * (xₛ .- x₀)
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
function gbq_gauss_kme(rff_k, rff_px, x::Vector, lb::Vector, ub::Vector)
    n_dim = length(lb)
    
    # transform fourier features by lengthscale and sigma
    Ω = rff_k.ω ./ rff_k.σ'
    Ρ = rff_px.ρ ./ sqrt.(diag(rff_px.Σ)')

    # calculate truncation term
    trunc_term = cdf(rff_px, lb, ub)

    # calculate indefinite integral and make function to run over all combinations of ω and ρ
    indefinite_integral_func = indefinite_gaussian_kme_func_pt(lb, ub)

    function indefinite_gaussian_kme_func(xₛ::Vector, x₀::Vector)
        integrated = hcat(map(
            j -> map(
                k -> indefinite_integral_func(xₛ, x₀, Ω[j, :], Ρ[k, :], rff_px.μ), 1:size(Ρ, 1)
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

####################################### DEVELOPMENT ########################################
function fourier_feats_cartesian_product(ω, ρ)
    ωρ = map(ρ_ -> map(ω_ -> [ω_, ρ_], ColVecs(ω)), ColVecs(ρ))
    ωρ = hcat(hcat.(ωρ...)...)  # first row is kernel.ω, second row measure.ρ
    return ωρ
end

function init_harmonic_addition_params(t₁)
    t₂ = sign.(cos.(t₁))
    t₃ = -atan.(tan.(t₁))
    return t₂, t₃
end

function uniform_init_sin_feats_harmonic_addition_params(t₁)
    t₂ = sign.(cos.(t₁) .- sin.(t₁)) .* √2
    t₃ = atan.((sin.(t₁) .+ cos.(t₁)) ./ (sin.(t₁) .- cos.(t₁)))
    return t₂, t₃
end

function gaussian_init_sin_feats_harmonic_addition_params(t₁)
    t₂ = sign.(-sin.(t₁))
    t₃ = atan.(cot.(t₁))
    return t₂, t₃
end

function integrated_trig_functions(n_ints)
    # cosine integrated
    trig_funcs = [z -> cos(z), z -> sin(z), z -> -cos(z), z -> -sin(z)]
    is_cosines = [true, false, true, false]
    is_negatives = [false, false, true, true]

    trig_sign_index = ifelse(n_ints < 4, n_ints, mod(n_ints, 4)) + 1
    trig_func = trig_funcs[trig_sign_index]
    is_cosine = is_cosines[trig_sign_index]
    is_negative = is_negatives[trig_sign_index]

    return trig_func, is_cosine, is_negative
end

function definite_integration_param_update(xₗ, xᵤ, ff, t₁, t₂)
    t₁_upper = cos.(ff .* xᵤ .+ t₂)
    t₁_lower = cos.(ff .* xₗ .+ t₂)
    t₁_delta = t₁_upper .- t₁_lower
    # from identity (cos(b) - cos(a))² + (sin(a) - sin(b))² = 2 - 2cos(b - a)
    t₃ = t₁ .* sign.(t₁_delta) .* sqrt.(2 .- 2cos.(ff .* (xᵤ - xₗ)))
    # from identity (cos(b) - cos(a)) / (sin(a) - sin(b)) = cot((a + b) / 2)
    t₄ = ((ff .* (xᵤ + xₗ)) ./ 2) .+ t₂  #  (a + b) / 2
    t₄ = atan.(-cot.(t₄))
    return t₃, t₄
end

function uniform_definite_integral_step(ω, α, β, dim, lb, ub)
    ω_dim = ω[dim, :]
    xₗ = lb[dim]
    xᵤ = ub[dim]

    # update params
    α, β = definite_integration_param_update(xₗ, xᵤ, ω_dim, α, β)

    return α, β
end

function gaussian_definite_integral_step(ω₊ρ, ω₋ρ, α, β, γ, δ, dim, lb, ub)
    ω₊ρ_dim = ω₊ρ[dim, :]
    ω₋ρ_dim = ω₋ρ[dim, :]
    xₗ = lb[dim]
    xᵤ = ub[dim]

    # update params
    α, β = definite_integration_param_update(xₗ, xᵤ, ω₊ρ_dim, α, β)
    γ, δ = definite_integration_param_update(xₗ, xᵤ, ω₋ρ_dim, γ, δ)

    return α, β, γ, δ
end

function final_definite_integral_param_update(xₗ, xᵤ, ff, t₁, t₂)
    upper = cos.(ff .* xᵤ .+ t₂)
    lower = cos.(ff .* xₗ .+ t₂)
    delta = upper .- lower
    return t₁ .* delta
end

function final_uniform_definite_integral_step(ω, α, β, lb, ub)
    dim = size(ω, 1)
    ω_dim = ω[dim, :]
    xₗ = lb[dim]
    xᵤ = ub[dim]

    # update params
    fin = final_definite_integral_param_update(xₗ, xᵤ, ω_dim, α, β)

    return fin
end

function final_gaussian_definite_integral_step(ω₊ρ, ω₋ρ, α, β, γ, δ, lb, ub)
    dim = size(ω₊ρ, 1)
    ω₊ρ_dim = ω₊ρ[dim, :]
    ω₋ρ_dim = ω₋ρ[dim, :]
    xₗ = lb[dim]
    xᵤ = ub[dim]

    # update params
    fin_1 = final_definite_integral_param_update(xₗ, xᵤ, ω₊ρ_dim, α, β)
    fin_2 = final_definite_integral_param_update(xₗ, xᵤ, ω₋ρ_dim, γ, δ)
    
    return fin_1 .+ fin_2
end

function uniform_rff_kernel_mean(X, kernel, lb, ub, sin_feats=false)
    d = length(lb)
    ints_by_dim = n_ints_by_dim(lb, ub)
    n_ints = sum(ints_by_dim)
    ω = kernel.ω ./ kernel.σ

    # establish R and Z
    # if sin_feats
    #     R = size(kernel.ω, 2) * 2
    # else
    #     R = size(kernel.ω, 2)
    # end
    R = size(kernel.ω, 2)
    
    # normalization constant
    L = prod(abs.(ub .- lb)) * kernel.λ / R
    # split out x* and X terms and initialize params
    ωX = ω' * X
    if sin_feats
        α, β = uniform_init_sin_feats_harmonic_addition_params(ωX)
    else
        α, β = init_harmonic_addition_params(ωX)
    end

    # perform indefinite integration
    ω_prod = prod(ω, dims=1)[1, :]
    α = α ./ ω_prod  # update due to chain rule
    trig_func, is_cosine, is_negative = integrated_trig_functions(n_ints)

    # if integrated trig func is negative, modify params
    if is_negative
        α = α .* -1
    end

    # if integrated trig func is not cosine, put in cosine form
    if !is_cosine
        β = β .- π/2
    end

    ## # perform definite integration over bounds
    for dim in 1:d-1
        α, β = uniform_definite_integral_step(ω, α, β, dim, lb, ub)
    end

    # final integral step
    out = final_uniform_definite_integral_step(ω, α, β, lb, ub)

    return sum(L .* out, dims=1)[1, :]
end

function gaussian_rff_kernel_mean(X, kernel, measure, lb, ub, sin_feats=false)
    d = length(lb)
    ints_by_dim = n_ints_by_dim(lb, ub)
    n_ints = sum(ints_by_dim)

    # establish R and Z
    R = size(kernel.ω, 2)
    Z = size(measure.ρ, 2)
    # if sin_feats
    #     R = size(kernel.ω, 2) * 2
    #     Z = size(measure.ρ, 2) * 2
    # else
    #     R = size(kernel.ω, 2)
    #     Z = size(measure.ρ, 2)
    # end
    
    # normalization constant
    # trunc_term = cdf(measure, lb, ub)
    trunc_term = 0.14829144308886272
    L = kernel.λ * prod(abs.(ub .- lb)) ./ trunc_term
    L = L ./ (sqrt(2π)^d * sqrt(det(measure.Σ)))
    if sin_feats
        L = L / (R * Z)
    else
        L = L / (2 * R * Z)
    end
    
    # cartesian product of kernel and measure fourier features
    ω = kernel.ω ./ kernel.σ
    ρ = measure.ρ ./ sqrt.(diag(measure.Σ))
    ωρ = fourier_feats_cartesian_product(ω, ρ)

    # initialize the two sets of convolved feature pairs
    ω₊ρ =  ωρ[1, :] .+ ωρ[2, :]
    ω₊ρ = hcat(ω₊ρ...)  # omega + rho for all pairs
    ω₊ρ_prod = prod(ω₊ρ, dims=1)[1, :]

    ω₋ρ =  ωρ[1, :] .- ωρ[2, :]  
    ω₋ρ = hcat(ω₋ρ...)  # omega - rho for all pairs
    ω₋ρ_prod = prod(ω₋ρ, dims=1)[1, :]
    
    # constant values dependent on X and μ
    ωX = X' * hcat(ωρ[1, :]...)
    ρμ = measure.μ' * hcat(ωρ[2, :]...)
    ρμ = ρμ[1, :]
    ωX₊ρμ = ωX' .+ ρμ
    ωX₋ρμ = ωX' .- ρμ

    # combine terms using identity:
    # z₁cos(x) + z₂sin(x) = z₃cos(x + z₄)
    # z₃ = sign(z₁)√(z₁^2 + z₂^2)
    # z₄ = arctan(- z₂ / z₁)
    if sin_feats
        α, β = gaussian_init_sin_feats_harmonic_addition_params(ωX₊ρμ)
        γ, δ = init_harmonic_addition_params(ωX₋ρμ)
    else
        α, β = init_harmonic_addition_params(ωX₊ρμ)
        γ, δ = init_harmonic_addition_params(ωX₋ρμ)
    end  # these are the params we work with from hereon out

    # perform indefinite integration
    α = α ./ ω₊ρ_prod
    γ = γ ./ ω₋ρ_prod  # update due to chain rule
    trig_func, is_cosine, is_negative = integrated_trig_functions(n_ints)

    # if integrated trig func is negative, modify params
    if is_negative
        α = α .* -1
        γ = γ .* -1
    end

    # if integrated trig_func isn't cosine, modify back into cosine form
    if !is_cosine
        β = β .- π/2
        δ = δ .- π/2
    end

    # # # perform definite integration over bounds
    for dim in 1:d-1
        α, β, γ, δ = gaussian_definite_integral_step(ω₊ρ, ω₋ρ, α, β, γ, δ, dim, lb, ub) 
    end

    # final integral step
    out = final_gaussian_definite_integral_step(ω₊ρ, ω₋ρ, α, β, γ, δ, lb, ub)
    out = sum(L .* out, dims=1)[1, :]

    return out
end

mutable struct UniformGBQ
    kernel::RandomFeatureKernel
    lb::AbstractVector
    ub::AbstractVector
end

function (gbq_model::UniformGBQ)(X::AbstractVector, y)
    ## establish variables
    kernel = gbq_model.kernel
    lb, ub = gbq_model.lb, gbq_model.ub
    sin_feats = kernel.sin_feats

    # make cleaned sin feats kernel matrix
    sin_kernel = deepcopy(kernel)
    sin_kernel.sin_feats = true
    K = sin_kernel(X)

    # run gbq
    z = uniform_rff_kernel_mean(X, kernel, lb, ub, sin_feats)

    # full output
    out = z' * inv(K) * y

    return out
end

function (gbq_model::UniformGBQ)(X::AbstractMatrix, y)
    ## establish variables
    kernel = gbq_model.kernel
    lb, ub = gbq_model.lb, gbq_model.ub
    sin_feats = kernel.sin_feats

    # make cleaned sin feats kernel matrix
    sin_kernel = deepcopy(kernel)
    sin_kernel.sin_feats = true
    K = add_jitter(kernelmatrix(sin_kernel, ColVecs(X)), 1e-7)

    # run gbq
    z = uniform_rff_kernel_mean(X, kernel, lb, ub, sin_feats)

    # full output
    out = z' * inv(K) * y

    return out
end

mutable struct GaussianGBQ
    kernel::RandomFeatureKernel
    measure
end

function (gbq_model::GaussianGBQ)(X::AbstractVector, y, lb::AbstractVector, ub::AbstractVector)
    ## establish variables
    kernel = gbq_model.kernel
    measure = gbq_model.measure
    sin_feats = kernel.sin_feats & measure.sin_feats

    # make cleaned sin feats kernel matrix
    sin_kernel = deepcopy(kernel)
    sin_kernel.sin_feats = true
    K = sin_kernel(X)

    # run gbq
    z = gaussian_rff_kernel_mean(X, kernel, measure, lb, ub, sin_feats)

    # full output
    out = z' * inv(K) * y

    return out
end

function (gbq_model::GaussianGBQ)(X::AbstractMatrix, y, lb::AbstractVector, ub::AbstractVector)
    ## establish variables
    kernel = gbq_model.kernel
    measure = gbq_model.measure
    sin_feats = kernel.sin_feats & measure.sin_feats

    # make cleaned sin feats kernel matrix
    sin_kernel = deepcopy(kernel)
    sin_kernel.sin_feats = true
    K = add_jitter(kernelmatrix(sin_kernel, ColVecs(X)), 1e-7)
    
    # run gbq
    z = gaussian_rff_kernel_mean(X, kernel, measure, lb, ub, sin_feats)

    # full output
    out = z' * inv(K) * y

    return out
end
