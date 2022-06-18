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

######################################## EVALUATION ########################################
function pct_error(true_val, pred)
    return abs((true_val - pred) / true_val) * 100
end

function pct_error(true_val::Float64, preds::Matrix)
    return abs.((true_val .- preds) ./ true_val) .* 100
end

function mse(true_val::AbstractVector, preds::AbstractVector)
    return mean(((true_val .- preds)).^2)
end

function mae(true_val::AbstractVector, preds::AbstractVector)
    return mean(abs.((true_val .- preds)))
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

function confidence_bounds(err_means, err_sds, logy=false)
    n_cols = size(err_means, 2)
    ns = err_means[:, 1]

    if logy
        err_sd_frame = log.(err_sds[:, 2:n_cols])
        err_mu_frame = log.(err_means[:, 2:n_cols])
        upper_vals = err_mu_frame .+ err_sd_frame .* 1.95
        lower_vals = err_mu_frame .- err_sd_frame .* 1.95
    else
        err_sd_frame = err_sds[:, 2:n_cols]
        err_mu_frame = err_means[:, 2:n_cols]
        upper_vals = err_mu_frame .+ err_sd_frame .* 1.95
        lower_vals = err_mu_frame .- err_sd_frame .* 1.95
        lower_vals = ifelse.(lower_vals .< 0, 0, lower_vals)
        upper_vals = ifelse.(upper_vals .< 0, 0, upper_vals)
    end

    lower_vals = hcat(ns, lower_vals)
    upper_vals = hcat(ns, upper_vals)
    rename!(lower_vals, :x1 => "n")
    rename!(upper_vals, :x1 => "n")
    return lower_vals, upper_vals
end

function full_error_df(err_means, err_sds, logx=false, logy=false)
    lower_errs, upper_errs = confidence_bounds(err_means, err_sds, logy)
    
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
    if logx
        errs[!, :n] = log.(errs[!, :n])
    end
    if logy
        errs[!, :μ] = log.(errs[!, :μ])
    end
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

function final_err_plot(err, position, 
        title=nothing, labels=nothing, columns=nothing,
        logx=false, logy=false
    )
    
    if !isnothing(columns)
        err = filter(row -> row.model ∈ columns, err)
    end

    if logx
        x_label = "N Training Data (log)"
    else
        x_label = "N Training Data"
    end

    if logy
        y_label = "Mean % Error vs. Analytical (log)"
        if !isnothing(labels)
            p = Gadfly.plot(
                err, x=:n, y=:μ, color=:model,
                Geom.point, Geom.line, alpha=[0.5],
                Guide.title(title), style(line_width=3mm),
                Scale.color_discrete_manual("red1", "deepskyblue1", "springgreen", "maroon1"),
                Guide.colorkey(title="", labels=labels, pos=position),
                Guide.xlabel(x_label),
                Guide.ylabel(y_label),
                Theme(#major_label_font="Computer Modern",
                    #minor_label_font="Computer Modern",
                    #key_label_font="Computer Modern",
                    major_label_font_size=8pt,minor_label_font_size=6pt,
                    key_label_font_size=7pt
                )
            )
        else
            p = Gadfly.plot(
                err, x=:n, y=:μ, color=:model,
                Geom.point, Geom.line, alpha=[0.5],
                Guide.title(title), style(line_width=2mm),
                Guide.colorkey(title="", pos=position),
                Scale.color_discrete_manual("red1", "deepskyblue1", "springgreen", "maroon1"),
                Guide.xlabel(x_label),
                Guide.ylabel(y_label),
                Theme(#major_label_font="Computer Modern",
                    #minor_label_font="Computer Modern",
                    #key_label_font="Computer Modern",
                    major_label_font_size=8pt, minor_label_font_size=6pt,
                    key_label_font_size=7.25pt
                )
            )
        end
    
    else
        y_label = "Mean % Error vs. Analytical"
        if !isnothing(labels)
            p = Gadfly.plot(
                err, x=:n, y=:μ, color=:model, ymin=:lb, ymax=:ub,
                Geom.point, Geom.line, Geom.ribbon, alpha=[0.5],
                Guide.title(nothing), style(line_width=3mm),
                Scale.color_discrete_manual("red1", "deepskyblue1", "springgreen", "maroon1"),
                Guide.colorkey(title="", labels=labels, pos=position),
                Guide.xlabel(x_label),
                Guide.ylabel(y_label),
                Theme(#major_label_font="Computer Modern",
                    #minor_label_font="Computer Modern",
                    #key_label_font="Computer Modern",
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
                Scale.color_discrete_manual("red1", "deepskyblue1", "springgreen", "maroon1"),
                Guide.xlabel(x_label),
                Guide.ylabel(y_label),
                Theme(#major_label_font="Computer Modern",
                    #minor_label_font="Computer Modern",
                   # key_label_font="Computer Modern",
                    major_label_font_size=8pt, minor_label_font_size=6pt,
                    key_label_font_size=7.25pt
                )
            )
        end
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
