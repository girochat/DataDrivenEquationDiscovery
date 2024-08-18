# SciML tools
import ModelingToolkit, Symbolics

# Standard libraries
using Statistics, Plots, CSV, DataFrames, Printf

# External libraries
using HyperTuning, StableRNGs, Distributions, SmoothingSplines, Logging, ColorSchemes, JLD2, ProgressLogging

# Packages under development (debugging)
using DataDrivenDiffEq, DataDrivenSparse

# Set a random seed for reproducibility
rng = StableRNG(1111)

# Explicitly call Plots backend
gr()


# Import utils function
include("esindy_utils.jl")
using .ESINDyModule



### USER-DEFINED parameters (model type, frequency, GF concentration)

# Retrieve file arguments
if length(ARGS) < 4
    error("Error! You need to specify as arguments:
        - Type of model (NFB/GF model)
        - Number of bootstraps
        - Coefficient threshold
        - Confidence interval
        - List of files")
else
    model = uppercase(ARGS[1])
    n_bstraps = parse(Int, ARGS[2])
    coef_threshold = parse(Int, ARGS[3])
    CI_confidence = parse(Int, ARGS[4])
    files = ARGS[5:end]

    println("Running E-SINDy for $(model) model with $(n_bstraps) bootstraps. 
        Coefficient threshold set to $(coef_threshold).
        Plotting with $(CI_confidence) confidence interval.")

    # Name output file to save the results according to parameters used
    filename = "ngf_esindy_100bt"
end

if model == "gf"
    using .GFModule
else
    using .NFBModule
end





##### Import the data #####
df = CSV.read(files[1], DataFrame)
if length(files) > 1
    for i in 2:length(files)
        df2 = CSV.read(files[i], DataFrame)
        global df = vcat(df, df2)
    end
end

labels = make_labels(files)
data = create_data(df, labels, 300.)




##### Set up the SINDy library #####

# Declare necessary symbolic variables for the bases
@ModelingToolkit.variables x[1:7] i[1:1]
@ModelingToolkit.parameters t
x = collect(x)
i = collect(i)
basis = build_basis(x[1:size(data.X, 2)], i)


##### Run E-SINDy (b(r)agging) #####
results = e_sindy(data, basis, n_bstraps, coef_threshold, CI_confidence) 

# save results
JLD2.@save "./Data/$(filename).jld2" results


