# SciML tools
import ModelingToolkit, Symbolics

# Standard libraries
using Statistics, Plots, CSV, DataFrames

# External libraries
using HyperTuning, StableRNGs, Distributions, SmoothingSplines, ColorSchemes, JLD2

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
Coefficient threshold set to $(coef_threshold).\n")
    flush(stdout)
end

if model == "ERK"
    using .ERKModule
else
    using .NFBModule
end


##### Import the data #####
data = create_data(files, 300.)


##### Set up the SINDy library #####

# Declare necessary symbolic variables for the bases
@ModelingToolkit.variables x[1:7] i[1:1]
@ModelingToolkit.parameters t
x = collect(x)
i = collect(i)
basis = build_basis(x[1:size(data.X, 2)], i)


##### Run Library E-SINDy #####
#lib_coefficients = library_bootstrap(data, basis, n_bstraps, 10)

##### Run E-SINDy (b(r)agging) #####
results = e_sindy(data, basis, n_bstraps, coef_threshold) 

#results = (library_esindy = lib_coefficients, esindy_results = results)

# save results
println("Saving results to ./Data/$(filename).jld2.")
JLD2.@save "./Data/$(filename).jld2" results


