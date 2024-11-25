# SciML tools
import ModelingToolkit, Symbolics

# Standard libraries
using Statistics, Plots, CSV, DataFrames

# External libraries
using HyperTuning, StableRNGs, Distributions, SmoothingSplines, ColorSchemes, JLD2

# Data Driven Equation Discovery packages
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
        - Type of model (NFB/ERK model)
        - Number of bootstraps
        - Coefficient threshold
        - Output filename
        - List of files")
else
    model = uppercase(ARGS[1])
    n_bstraps = parse(Int, ARGS[2])
    coef_threshold = parse(Int, ARGS[3])
    filename = ARGS[4]
    files = ARGS[5:end]

    println("Running E-SINDy for $(model) model with $(n_bstraps) bootstraps. 
Coefficient threshold set to $(coef_threshold).")
    flush(stdout)
end

if model == "ERK"
    using .ERKModule
else
    using .NFBModule
end


##### Import the data #####
if lowercase(model) == "erk"
    data = create_data(files, smoothing=300., var_idxs=1:5)  # for ERK
else
    data = create_data(files, smoothing=300.)  # for NFB
end


##### Set up the SINDy library #####

# Declare necessary symbolic variables for the bases
@ModelingToolkit.variables x[1:7] i[1:1]
@ModelingToolkit.parameters t
x = collect(x)
i = collect(i)
basis = build_basis(x[1:size(data.X, 2)], i)


##### Run E-SINDy (b(r)agging) #####
esindy_res = esindy(data, basis, n_bstraps, coef_threshold=coef_threshold, data_fraction=1) 


##### Run Library E-SINDy #####
#lib_coefficients = library_bootstrap(data, basis, n_bstraps, 10, implicit_id=22)
#lib_basis = ESINDyModule.build_basis(lib_coefficients, basis)
#esindy_res = esindy(data, lib_basis, n_bstraps, coef_threshold=coef_threshold, data_fraction=1)
#esindy_res = (esindy = esindy_res, esindy_lib = lib_coefficients)


##### Save results #####
println("Saving results to ../Data/$(filename).jld2.")
jldsave("../Data/$(filename).jld2"; results=esindy_res)


