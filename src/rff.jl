using Random
using KernelFunctions
using Statistics
using AbstractGPs
using CUDA

######################################### STRUCTS ##########################################
mutable struct RandomFeatureKernel{T} <: KernelFunctions.Kernel
    ω::T  # matrix of spectral frequencies
    β::T  # vector of kernel bias
    σ::T  # vector of kernel LS - this is in squared form
    λ::T  # vector of kernel variance
    sin_feats::Bool
    RandomFeatureKernel{T}(ω, β, σ, λ, sin_feats) where {T} = new(deepcopy(ω), deepcopy(β), deepcopy(σ), deepcopy(λ), deepcopy(sin_feats))
end

######################################## PROJECTION ########################################
#################### CPU #####################
function  ϕ(x::AbstractArray, fourier_feats::AbstractMatrix, bias::AbstractVector, ls::AbstractVector, λ::AbstractVector, sin_feats=false)
    fourier_feats = fourier_feats ./ ls
    zₓ = .√λ .* √2 .* cos.(fourier_feats' * x .+ bias)

    if sin_feats
        sf = .√λ .* √2 .* sin.(fourier_feats' * x .+ bias)
        zₓ = vcat(zₓ, sf)
    end
    return zₓ
end

#################### GPU #####################
function  ϕ(x::CuArray, fourier_feats::CuArray, bias::CuArray, ls::CuArray, λ::CuArray, sin_feats=false)
    fourier_feats = fourier_feats ./ ls
    zₓ = .√λ .* √2 .* cos.(fourier_feats' * x .+ bias)

    if sin_feats
        sf = .√λ .* √2 .* sin.(fourier_feats' * x .+ bias)
        zₓ = vcat(zₓ, sf)
    end
    return zₓ
end

####################################### KERNEL FUNCS #######################################
function rff_kernel_estimate(z₁, z₂)
    m = size(z₁, 1)
    return z₁' * z₂ ./ m
end

# define call functions
# pointwise - k(x1, x2)
function rff_kernel(x1, x2, fourier_feats, bias, ls, λ, sin_feats=false)
    if x1 == x2
        z₁ = z₂ = ϕ(x1, fourier_feats, bias, ls, λ, sin_feats)
    else
        z₁ = ϕ(x1, fourier_feats, bias, ls, λ, sin_feats)
        z₂ = ϕ(x2, fourier_feats, bias, ls, λ, sin_feats)
    end
    return rff_kernel_estimate(z₁, z₂)
end

# single observations
(k::RandomFeatureKernel)(x1::AbstractVector) = rff_kernel(x1, x1, k.ω, k.β, k.σ, k.λ, k.sin_feats)
(k::RandomFeatureKernel)(x1::AbstractVector, x2::AbstractVector) = rff_kernel(x1, x2, k.ω, k.β, k.σ, k.λ, k.sin_feats)
(k::RandomFeatureKernel)(x1::CuArray) = rff_kernel(x1, x1, k.ω, k.β, k.σ, k.λ, k.sin_feats)
(k::RandomFeatureKernel)(x1::CuArray, x2::CuArray) = rff_kernel(x1, x2, k.ω, k.β, k.σ, k.λ, k.sin_feats)

# multiple observations
function vec_of_obs_to_mat(x_vec)
    return hcat(x_vec...)
end

KernelFunctions.kernelmatrix(k::RandomFeatureKernel, x1::ColVecs) = rff_kernel(vec_of_obs_to_mat(x1), vec_of_obs_to_mat(x1), k.ω, k.β, k.σ, k.λ, k.sin_feats)
KernelFunctions.kernelmatrix(k::RandomFeatureKernel, x1::ColVecs, x2::ColVecs) = rff_kernel(vec_of_obs_to_mat(x1), vec_of_obs_to_mat(x2), k.ω, k.β, k.σ, k.λ, k.sin_feats)

############## TEMPORARY GPU FIX
# function Statistics.mean(f, x::AbstractVector)
#     fdataα_d = f.data.α |> gpu
#     return mean(f.prior, x) + cov(f.prior, x, f.data.x) * fdataα_d
# end
