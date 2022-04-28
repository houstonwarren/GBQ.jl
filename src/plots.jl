using Plots
using Gadfly
using CSV
using DataFrames
using Latexify
using Cairo
using ColorSchemes

nms = [
    "n", "true", "quad", "mc", "qmc", "bq", "sbq_uni", "sbq_uni_m12", "sbq_uni_m32", "sbq_uni_m52",
    "sbq_gauss", "sbq_gauss_m12", "sbq_gauss_m32", "sbq_gauss_m52",
]


err_nms = [
    "n", "quad", "mc", "qmc", "bq", "sbq_uni", "sbq_uni_m12", "sbq_uni_m32", "sbq_uni_m52",
    "sbq_gauss", "sbq_gauss_m12", "sbq_gauss_m32", "sbq_gauss_m52",
]

##################################### PAPER PLOT 1: 1D #####################################

################## POLY 1D ###################
poly1d_data = CSV.read("experiments/1D/results/poly_1d_data.csv", DataFrame)
poly1d_err_sd = CSV.read("experiments/1D/results/poly_1d_err_stds.csv", DataFrame)
poly1d_err = CSV.read("experiments/1D/results/poly_1d_err_means.csv", DataFrame)

####### Data Plot
sort!(poly1d_data, :x)
poly1d_data_plot = plot_1d_func(poly1d_data)

maxes = []
for row in 1:10
    push!(maxes, findmin(poly1d_err[row, 4:13]))
end
maxes

####### Error Plot
poly_1d_sum = full_error_df(poly1d_err, poly1d_err_sd)
labels = ["QMC","BQ", "GBQ-G w/ RBF", "GBQ-G w/ Matern 3/2"]
position = [32, 11.5]
columns = ["qmc", "bq", "sbq_gauss", "sbq_gauss_m32"]
poly_1d_err_plot = final_err_plot(poly_1d_sum, title, position, labels, columns)

################## DISJOINT 1D ###################
dis1d_err_means = CSV.read("experiments/1D/results/disjoint_1d_err_means.csv", DataFrame)
dis1d_err_sd = CSV.read("experiments/1D/results/disjoint_1d_err_stds.csv", DataFrame)
dis1d_data = CSV.read("experiments/1D/results/disjoint_1d_data.csv", DataFrame)

####### Data Plot
sort!(dis1d_data, :x)
dis1d_data_plot = disjoint_1d_func(dis1d_data, 2.5)

maxes = []
for row in 1:10
    push!(maxes, findmin(dis1d_err_means[row, 5:13]))
end
maxes

####### Error Plot
dis_1d_sum = full_error_df(dis1d_err_means, dis1d_err_sd)
labels = ["QMC","BQ", "GBQ-U w/ Matern 3/2",  "GBQ-G w/ RBF"]
position = [30, 11]
columns = ["qmc", "bq", "sbq_gauss", "sbq_uni_m32"]
dis_1d_err_plot = final_err_plot(dis_1d_sum, title, position, labels, columns)

################################################# COMBINE AND SAVE
combined_1d = hstack(poly1d_data_plot, poly_1d_err_plot, dis1d_data_plot, dis_1d_err_plot)
draw(PDF("experiments/1D/results/1d_combined.pdf", 10inch, 3inch), combined_1d)

##################################### PAPER PLOT 2: 2D #####################################
################## POLY 2D ###################
p2d_err_means = CSV.read("experiments/2D/results/p2d_err_means.csv", DataFrame)
p2d_err_sd = CSV.read("experiments/2D/results/p2d_err_stds.csv", DataFrame)
p2d_data = CSV.read("experiments/2D/results/p2d_data.csv", DataFrame)

####### Data Plot
sort!(p2d_data, [:x1, :x2])
p2d_data_plot = surface(-4:0.1:4, -2.5:0.1:2.5, p2d, camera = (60, 50), c=:cool, legend=:none)

####### Error Plot
poly_2d_sum = full_error_df(p2d_err_means, p2d_err_sd)
# labels = ["QMC", "BQ", "GBQ-U w/ Matern 3/2", "GBQ-G w/ RBF"]
# pos = [3, 2.5]
# columns = ["qmc", "bq", "sbq_uni_m32", "sbq_gauss"]
pos = [750, 300]
labels = ["QMC", "GBQ-U w/ RBF", "GBQ-G w/ RBF", "GBQ-G w/ Matern 5/2"]
columns = ["qmc", "sbq_uni", "sbq_gauss", "sbq_gauss_m52"]
poly_2d_err_plot = final_err_plot(poly_2d_sum, "thing", pos,labels, columns)

#### tables
maxes = []
for row in 1:8
    push!(maxes, findmin(p2d_err_means[row, 4:13]))
end
maxes

# should go n | model | val |
p2d_cols = ["qmc", "bq", "sbq_uni", "sbq_gauss", "sbq_gauss_m52"]
p2d_error_table = error_table(p2d_err_means, p2d_err_sd, p2d_cols)
latexify(p2d_error_table, env=:table)

################## DISJOINT 2D ###################
dis2d_err_means = CSV.read("experiments/2D/results/disjoint_2d_err_means.csv", DataFrame)
dis2d_err_sd = CSV.read("experiments/2D/results/disjoint_2d_err_stds.csv", DataFrame)
dis2d_data = CSV.read("experiments/2D/results/disjoint_2d_data.csv", DataFrame)

####### Data Plot
sort!(dis2d_data, [:x1, :x2])
dis2d_data_plot = surface(0:0.01:1, 0.0:0.01:1, disjoint_poly_2d, camera = (30, 55), c=:inferno, legend=:none)

####### Error Plot
dis_2d_sum = full_error_df(dis2d_err_means, dis2d_err_sd)
# labels = ["QMC", "BQ", "GBQ-U w/ Matern 3/2", "GBQ-G w/ RBF"]
# pos = [3, 2.5]
# columns = ["qmc", "bq", "sbq_uni_m32", "sbq_gauss"]
pos = [3, 2.5]
labels = ["QMC", "GBQ-U w/ Matern 3/2", "GBQ-G w/ RBF", "GBQ-G w/ Matern 5/2"]
columns = ["qmc", "sbq_uni_m32", "sbq_gauss", "sbq_gauss_m52"]
poly_2d_err_plot = final_err_plot(poly_2d_sum, "thing", pos,labels, columns)

maxes = []
for row in 1:8
    push!(maxes, findmin(dis2d_err_means[row, 4:13]))
end
maxes

# should go n | model | val |
d2d_cols = ["qmc", "bq", "sbq_uni", "sbq_uni_m12", "sbq_gauss"]
d2d_error_table = error_table(dis2d_err_means, dis2d_err_sd, d2d_cols)
latexify(d2d_error_table, env=:table)

################################### 5D & 10D EXPERIMENTS ###################################
##################### 5D #####################
bmc_err_means = CSV.read("experiments/ND/results/bmc_exp_err_means.csv", DataFrame)
bmc_err_sd = CSV.read("experiments/ND/results/bmc_exp_err_stds.csv", DataFrame)
bmc_data = CSV.read("experiments/ND/results/bmc_exp_data.csv", DataFrame)

####### Error Plot
bmc_sum = full_error_df(bmc_err_means, bmc_err_sd)

####### Calculate best methods
maxes = []
for row in 1:8
    push!(maxes, findmin(bmc_err_means[row, 3:13]))
end
maxes

####### Table
d2d_cols = ["qmc", "bq", "sbq_uni", "sbq_uni_m12", "sbq_gauss"]
d2d_error_table = error_table(dis2d_err_means, dis2d_err_sd, d2d_cols)
latexify(d2d_error_table, env=:table)

################ 5D DISJOINT #################
dis5d_err_means = CSV.read("experiments/ND/results/dis5d_err_means.csv", DataFrame)
dis5d_err_sd = CSV.read("experiments/ND/results/dis5d_err_stds.csv", DataFrame)
dis5d_data = CSV.read("experiments/ND/results/dis5d_data.csv", DataFrame)

#################### 10D #####################
morokoff_err_means = CSV.read("experiments/ND/results/morokoff_err_means.csv", DataFrame)
morokoff_err_sd = CSV.read("experiments/ND/results/morokoff_err_stds.csv", DataFrame)
morokoff_data = CSV.read("experiments/ND/results/morokoff_data.csv", DataFrame)
