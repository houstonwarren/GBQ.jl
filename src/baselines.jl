   using Distributions
using Quadrature

################################### NUMERICAL QUADRATURE ###################################
function quadrature(f, st, stp, noise_sd=0.0, maxiters=nothing, rng=nothing)
    
    if noise_sd > 0
        if isnothing(rng)
            quad_f_noise(x_, p) = f(x_...) + rand(Normal(0, noise_sd))
            prob = QuadratureProblem(quad_f_noise, st, stp)
        else
            quad_f_noise_rng(x_, p) = f(x_...) + rand(MersenneTwister(rng), Normal(0, noise_sd))
            prob = QuadratureProblem(quad_f_noise_rng, st, stp)
        end
 
    else
        quad_f(x_, p) = f(x_...)
        prob = QuadratureProblem(quad_f, st, stp)
    end
    
    if isnothing(maxiters)
        quad_int = solve(prob, HCubatureJL())
    else
        quad_int = solve(prob, HCubatureJL(), maxiters=maxiters)
    end

    return quad_int
end

####################################### MONTE CARLO ########################################
function mc_quadrature_with_data(f, lb, ub, pts, noise_sd=0.0, rng=nothing)
    n_samples = size(pts, 1)
    
    # generate samples
    fx = map(i -> f(pts[i, :]...), 1:n_samples)

    if noise_sd > 0.0
        if isnothing(rng)
            fx = fx .+ rand(Normal(0, noise_sd), n_samples)
        else
            fx = fx .+ rand(MersenneTwister(rng), Normal(0, noise_sd), n_samples)
        end
    end

    # mean
    μ = sum(fx) / n_samples * (ub - lb)

    # variance
    σ = std(fx) / sqrt(n_samples)

    return μ, σ
end

function mc_quadrature_with_data(f, lb::Vector, ub::Vector, pts::Matrix, noise_sd=0.0, rng=nothing)
    n_samples = size(pts, 1)
    
    fx = map(i -> f(pts[i, :]...), 1:n_samples)
    
    if noise_sd > 0.0
        if isnothing(rng)
            fx = fx .+ rand(Normal(0, noise_sd), n_samples)
        else
            fx = fx .+ rand(MersenneTwister(rng), Normal(0, noise_sd), n_samples)
        end
    end

    # area
    area = prod(abs.(ub .- lb))

    # mean
    μ = mean(fx) * area

    # variance
    σ = std(fx) / sqrt(n_samples)

    return μ, σ
end

function mc_quadrature(f, lb, ub, n_samples, noise_sd=0.0, rng=nothing)
    # generate samples
    mc_pts = convert(Array, LinRange(lb, ub, n_samples))
    fx = f(mc_pts)

    if noise_sd > 0.0
        if isnothing(rng)
            fx = fx .+ rand(Normal(0, noise_sd), n_samples)
        else
            fx = fx .+ rand(MersenneTwister(rng), Normal(0, noise_sd), n_samples)
        end
    end

    # mean
    μ = sum(fx) / n_samples * (ub - lb)

    # variance
    σ = std(fx) / sqrt(n_samples)

    return μ, σ
end

function mc_quadrature(f, lb::Vector, ub::Vector, n_samples, noise_sd=0.0, rng=nothing)
    mc_pts = sample_uniform_grid(n_samples, lb, ub)
    fx = map(i -> f(mc_pts[i, :]...), 1:n_samples)
    
    if noise_sd > 0.0
        if isnothing(rng)
            fx = fx .+ rand(Normal(0, noise_sd), n_samples)
        else
            fx = fx .+ rand(MersenneTwister(rng), Normal(0, noise_sd), n_samples)
        end
    end

    # area
    area = prod(abs.(ub .- lb))

    # mean
    μ = mean(fx) * area

    # variance
    σ = std(fx) / sqrt(n_samples)

    return μ, σ
end

function mc_quadrature_z(X, kernel, measure, lb, ub, n_samples)
    samples = rand(measure, n_samples)
    samples = samples[:, all(lb .< samples  .< ub, dims=1)[1, :]]

    if size(samples, 2) < n_samples
        diff = n_samples - size(samples, 2)
        more_samples = rand(measure, diff * 10)
        more_samples = more_samples[:, all(lb .< more_samples  .< ub, dims=1)[1, :]]
        samples = hcat((samples, more_samples[:, 1:diff])...)
    end

    print(typeof(X))

    km = kernelmatrix(kernel, X, ColVecs(samples))
    expectations = mean(km, dims=2) .* prod(abs.(ub .- lb))

    return expectations[:, 1]
end

############################### GAUSSIAN BAYESIAN QUADRATURE ###############################
# 1D
function bq_kme(x, μ::Float64, kernel_ls::Float64, σ::Float64, lb, ub)
    # posterior term of multiple of two gaussians
    p_x = Normal(μ, σ)
    posterior_mu = (x * σ^2 + μ * kernel_ls^2) / (kernel_ls^2 + σ^2)
    posterior_sd = sqrt((kernel_ls^2 * σ^2) / (kernel_ls^2 + σ^2))
    posterior = Normal(posterior_mu, posterior_sd)

    # normalizing constant
    # normalizer = Normal(μ, sqrt(kernel_ls^2 + σ^2))
    normalizer = Normal(μ, sqrt(kernel_ls^2 + σ^2))
    
    # calculate output
    out = pdf(normalizer, x) * (cdf(posterior, ub) - cdf(posterior, lb))
    out = out * sqrt(2π * kernel_ls^2)

    # truncation corrections
    trunc_term = cdf(p_x, ub) - cdf(p_x, lb)
    out = out / trunc_term
    out = out * (ub - lb)

    return out
end

function bayesian_quadrature(x, kernel_ls, x_dist, K, y, lb, ub)
    z = map(i -> bq_kme(x[i], x_dist.μ, kernel_ls, x_dist.σ, lb, ub), 1:size(x, 1))
    μ = z' * inv(K) * y
    return μ
end

# z-term in >1D
function bq_kme(x::Vector, μ::Vector, Σ_ls::Matrix, Σₓ::Matrix, lb::Vector, ub::Vector)
    # helpers
    n_dims = length(lb)
    p_x = MvNormal(μ, Σₓ)

    # posterior term of multiple of two gaussians
    sum_Σ_inv = inv(Σ_ls + Σₓ)
    posterior_mu = Σₓ * sum_Σ_inv * x + Σ_ls * sum_Σ_inv * μ
    posterior_sd = Σ_ls * sum_Σ_inv * Σₓ
    posterior = MvNormal(posterior_mu, posterior_sd)

    # normalizing constant
    normalizer = MvNormal(μ, Σ_ls .+ Σₓ)

    # p_x trunc term
    p_x_trunc_term = piecewise_mv_cdf(p_x, lb, ub)

    # posterior cdf estimation
    posterior_trunc_term = piecewise_mv_cdf(posterior, lb, ub)

    # calculate output
    out = pdf(normalizer, x) * posterior_trunc_term
    out = out / p_x_trunc_term
    out = out * (2π)^(n_dims/2) * sqrt(det(Σ_ls))
    out = out * prod(abs.(ub .- lb))
    return out

end

function bayesian_quadrature(x, Σ_ls, x_dist::MvNormal, K, y, lb, ub)
    z = map(i -> bq_kme(x[i, :], x_dist.μ, Σ_ls, Matrix(x_dist.Σ), lb, ub), 1:size(x, 1))
    μ = z' * inv(K) * y
    return μ
end
