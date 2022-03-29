using Plots
using Printf
using HDF5
using Gadfly

######################################### GENERAL ##########################################
function formatcomment(comment)
    comment_len = length(comment) + 2
    hash_len = (92 - comment_len) / 2
    if mod(hash_len, 2) == 0
        hast_st_len = convert(Int, hash_len)
        hast_end_len = convert(Int, hash_len)
        hash_st = repeat("#", hast_st_len) * " "
        hash_end = " " * repeat("#", hast_end_len)
    else
        hast_st_len = convert(Int, floor(hash_len))
        hast_end_len = convert(Int, ceil(hash_len))
        hash_st = repeat("#", hast_st_len)  * " "
        hash_end =  " " * repeat("#", hast_end_len)
    end
    println(hash_st * comment * hash_end)
end

function formatsubcomment(comment)
    comment_len = length(comment) + 2
    hash_len = (92 / 2 - comment_len) / 2
    if mod(hash_len, 2) == 0
        hast_st_len = convert(Int, hash_len)
        hast_end_len = convert(Int, hash_len)
        hash_st = repeat("#", hast_st_len) * " "
        hash_end = " " * repeat("#", hast_end_len)
    else
        hast_st_len = convert(Int, floor(hash_len))
        hast_end_len = convert(Int, ceil(hash_len))
        hash_st = repeat("#", hast_st_len)  * " "
        hash_end =  " " * repeat("#", hast_end_len)
    end
    println(hash_st * comment * hash_end)
end

function add_jitter(K::Matrix, jitter)
    K = K + jitter*I
    return K
end

function verify_invertible(K::Matrix, tolerance)
    eig_condition = minimum(eigvals(K)) >= 0
    mean_condition = mean(abs.(I - (inv(K) * K))) <= tolerance
    if eig_condition & mean_condition
        return true
    else
        return false
    end
end

####################################### EXPERIMENTS ########################################
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

######################################## EVALUATION ########################################
function pct_error(true_val, pred)
    return abs((true_val - pred) / true_val) * 100
end

function pct_error(true_val::Float64, preds::Matrix)
    return abs.((true_val .- preds) ./ true_val) .* 100
end

function error_df(preds, nms)
    preds = Matrix(preds)
    ns = preds[:, 1]
    true_val = preds[1, 2]
    preds = preds[:, 3:size(preds, 2)]

    pct_errs = pct_error.(ones(size(preds)) * true_val, preds)
    pct_errs = hcat(ns, pct_errs)
    err_df = DataFrame(pct_errs, nms)
    return err_df
end

function confidence_bounds(err_means, err_sds)
    n_cols = size(err_means, 2)
    ns = err_means[:, 1]
    upper_vals = err_means[:, 2:n_cols] .+ err_sds[:, 2:n_cols] .* 1.95
    lower_vals = err_means[:, 2:n_cols] .- err_sds[:, 2:n_cols] .* 1.95
    lower_vals = ifelse.(lower_vals .< 0, 0, lower_vals)
    upper_vals = ifelse.(upper_vals .< 0, 0, upper_vals)
    lower_vals = hcat(ns, lower_vals)
    upper_vals = hcat(ns, upper_vals)

    rename!(lower_vals, :x1 => "n")
    rename!(upper_vals, :x1 => "n")
    return lower_vals, upper_vals
end

function full_error_df(err_means, err_sds)
    lower_errs, upper_errs = confidence_bounds(err_means, err_sds)
    
    # reshape and rename
    err_means = stack(err_means, 2:13)
    rename!(err_means, Dict(:value => "μ", :variable => "model"))
    lower_errs = stack(lower_errs, 2:13)
    rename!(lower_errs, Dict(:value => "lb", :variable => "model"))
    upper_errs = stack(upper_errs, 2:13)
    rename!(upper_errs, Dict(:value => "ub", :variable => "model"))

    # join
    errs = innerjoin(err_means, lower_errs, on =[:n, :model])
    errs = innerjoin(errs, upper_errs, on =[:n, :model])
    return errs
end

function plot_err(err_df, nms, cols=nothing)
    errs = Matrix(err_df)
    if !isnothing(cols)
        plot(errs[:, 1], errs[:, cols], label = nms)
    else
        plot(errs[:, 1], errs[:, 2:size(err_df, 2)], label = nms)
    end
end

function plot_1d_func(df)
    p = Gadfly.plot(
        df, x=:x, y=:y, Guide.xlabel(nothing), Guide.ylabel(nothing), Geom.line,
        Theme(major_label_font="CMU Serif",
            minor_label_font="CMU Serif",
            major_label_font_size=8pt,minor_label_font_size=8pt,
            key_label_font_size=8pt, default_color="deepskyblue4"
        ), style(line_width=0.5mm),
    )
    return p
end

function disjoint_1d_func(df, split)
    df1 = df[df.x .< split, :]
    df2 = df[df.x .>= split, :]
    p = Gadfly.plot(
        layer(x=df1.x, y=df1.y, Geom.line, style(line_width=0.5mm)),
        layer(x=df2.x, y=df2.y, Geom.line, style(line_width=0.5mm)),
        Guide.xlabel(nothing), Guide.ylabel(nothing),
        Theme(major_label_font="CMU Serif",
            minor_label_font="CMU Serif",
            major_label_font_size=8pt,minor_label_font_size=8pt,
            key_label_font_size=8pt, default_color="deepskyblue4"
        )
    )

    return p
end

function final_err_plot(err, title, position, labels=nothing, columns=nothing)
    if !isnothing(columns)
        err = filter(row -> row.model ∈ columns, err)
    end

    if !isnothing(labels)
        p = Gadfly.plot(
            err, x=:n, y=:μ, color=:model, ymin=:lb, ymax=:ub,
            Geom.point, Geom.line, Geom.ribbon, alpha=[0.5],
            Guide.title(nothing), style(line_width=3mm),
            Scale.color_discrete_manual("brown2", "deepskyblue1", "springgreen", "sienna2"),
            Guide.colorkey(title="", labels=labels, pos=position),
            Guide.xlabel("N Training Data"),
            Guide.ylabel("% Error vs. Analytical"),
            Theme(major_label_font="CMU Serif",
                minor_label_font="CMU Serif",
                key_label_font="CMU Serif",
                major_label_font_size=8pt,minor_label_font_size=6pt,
                key_label_font_size=7.25pt
            )
        )
    else
        p = Gadfly.plot(
            err, x=:n, y=:μ, color=:model, ymin=:lb, ymax=:ub,
            Geom.point, Geom.line, Geom.ribbon, alpha=[0.5],
            Guide.title(nothing), style(line_width=2mm),
            Guide.colorkey(title="", pos=position),
            Scale.color_discrete_manual("brown2", "deepskyblue1", "springgreen", "sienna2"),
            Guide.xlabel("N Training Data"),
            Guide.ylabel("% Error vs. Analytical"),
            Theme(major_label_font="CMU Serif",
                minor_label_font="CMU Serif",
                key_label_font="CMU Serif",
                major_label_font_size=8pt, minor_label_font_size=6pt,
                key_label_font_size=7.25pt
            )
        )
    end
    return p
end

function string_format_mean_sd(μ, σ)
    return "$(@sprintf("%.2f", μ)) ± $(@sprintf("%.2f", σ))"
end

function error_table(err_means, err_stds, cols)
    ns = convert.(Int64, err_means[:, 1])
    err_means = err_means[:, cols]
    err_stds = err_stds[:, cols]
    err_df = string_format_mean_sd.(err_means, err_stds)
    err_df = hcat(ns, err_df)
    rename!(err_df, :x1 => "n")

    return err_df
end

function save_results(means, stds, errs, runs, filename)
    fid = h5open(filename, "w")
    fid["means"] = means
    fid["stds"] = stds
    fid["errs"] = errs
    fid["runs"] = runs
    close(fid)
end
