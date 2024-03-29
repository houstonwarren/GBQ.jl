######################################## PACKAGING #########################################
using Distributions
using KernelFunctions
using LinearAlgebra

##################################### HELPER FUNCTIONS #####################################
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

########## GENERAL INTEGRATION FUNS ##########
function n_ints_by_dim(lb::AbstractVector, ub::AbstractVector)
    # calculate the number of integrations over each dimension implied by vectors of bounds
    delta = ub .- lb
    ifelse.(abs.(delta) .> 0, 1, 0)
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

function final_definite_integral_param_update(xₗ, xᵤ, ff, t₁, t₂)
    upper = cos.(ff .* xᵤ .+ t₂)
    lower = cos.(ff .* xₗ .+ t₂)
    delta = upper .- lower
    return t₁ .* delta
end

########################## UNIFORM ##########################################
function uniform_init_sin_feats_harmonic_addition_params(t₁)
    t₂ = sign.(cos.(t₁) .- sin.(t₁)) .* √2
    t₃ = atan.((sin.(t₁) .+ cos.(t₁)) ./ (sin.(t₁) .- cos.(t₁)))
    return t₂, t₃
end

function uniform_definite_integral_step(ω, α, β, dim, lb, ub)
    ω_dim = ω[dim, :]
    xₗ = lb[dim]
    xᵤ = ub[dim]

    # update params
    α, β = definite_integration_param_update(xₗ, xᵤ, ω_dim, α, β)

    return α, β
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

################### STRUCT ###################
mutable struct UniformGBQ
    kernel::RandomFeatureKernel
    noise_sd::Float64
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
    K = K + I * gbq_model.noise_sd[1]^2

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
    K = K + I * gbq_model.noise_sd^2

    # run gbq
    z = uniform_rff_kernel_mean(X, kernel, lb, ub, sin_feats)

    # full output
    out = z' * inv(K) * y

    return out
end

######################################### GAUSSIAN #########################################
function gaussian_init_sin_feats_harmonic_addition_params(t₁)
    t₂ = sign.(-sin.(t₁))
    t₃ = atan.(cot.(t₁))
    return t₂, t₃
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
    trunc_term = cdf(measure, lb, ub)
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

################### STRUCT ###################
mutable struct GaussianGBQ
    kernel::RandomFeatureKernel
    measure
    noise_sd::Float64
end

function (gbq_model::GaussianGBQ)(X::AbstractVector, y, lb::AbstractVector, ub::AbstractVector)
    ## establish variables
    kernel = gbq_model.kernel
    measure = gbq_model.measure
    sin_feats = kernel.sin_feats & measure.sin_feats

    # make cleaned sin feats kernel matrix
    sin_kernel = deepcopy(kernel)
    sin_kernel.sin_feats = true
    K = sin_kernel(X) + I * gbq_model.noise_sd^2

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
    K = K + I * gbq_model.noise_sd^2
    
    # run gbq
    z = gaussian_rff_kernel_mean(X, kernel, measure, lb, ub, sin_feats)

    # full output
    out = z' * inv(K) * y

    return out
end
