using Random
using KernelFunctions


######################################### STRUCTS ##########################################
mutable struct RandomFeatureKernel <: KernelFunctions.Kernel
    ω  # fourier features
    β:: Vector{Float64}  # fourier feature bias
    σ::Vector  # kernel LS
    λ:: Float64  # kernel variance
    sin_feats:: Bool
end

################################### KERNEL HELPER FUNCS ####################################
# pointwise observation
function  ϕ(x, fourier_feats::Vector, bias::Vector, ls::Vector, λ, sin_feats=false)
    fourier_feats = fourier_feats ./ ls
    zₓ = √λ .* √2 .* cos.(fourier_feats .* x .+ bias)

    if sin_feats
        sf = √λ .* √2 .* sin.(fourier_feats .* x .+ bias)
        append!(zₓ, sf)
    end
    return zₓ
end

# vector observation
function  ϕ(x, fourier_feats::Matrix, bias::Vector, ls::Vector, λ, sin_feats=false)
    fourier_feats = fourier_feats ./ ls'
    zₓ = √λ .* √2 .* cos.(fourier_feats * x .+ bias)

    if sin_feats
        sf = √λ .* √2 .* sin.(fourier_feats * x .+ bias)
        append!(zₓ, sf)
    end
    return zₓ
end

function rff_kernel_estimate(z₁, z₂)
    m = size(z₁, 1)
    return z₁' * z₂ ./ m
end

####################################### KERNEL FUNCS #######################################
# define call functions
# pointwise - k(x1, x2)
function rff_kernel(x1, x2, fourier_feats, bias::Vector, ls::Vector, λ, sin_feats=false)
    if x1 == x2
        z₁ = z₂ = ϕ(x1, fourier_feats, bias, ls, λ, sin_feats)
    else
        z₁ = ϕ(x1, fourier_feats, bias, ls, λ, sin_feats)
        z₂ = ϕ(x2, fourier_feats, bias, ls, λ, sin_feats)
    end
    return rff_kernel_estimate(z₁, z₂)
end

(k::RandomFeatureKernel)(x1) = rff_kernel(x1, x1, k.ω, k.β, k.σ, k.λ, k.sin_feats)
(k::RandomFeatureKernel)(x1, x2) = rff_kernel(x1, x2, k.ω, k.β, k.σ, k.λ, k.sin_feats)
