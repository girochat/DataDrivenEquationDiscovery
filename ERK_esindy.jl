### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ a806fdd2-5017-11ef-2351-dfc89f69b334
begin
	# Activate project environment
	import Pkg
	Pkg.activate(".")
	
	# SciML tools
	import ModelingToolkit, Symbolics
	
	# Standard libraries
	using StatsBase, Plots, CSV, DataFrames#, Printf, Statistics

	# External libraries
	using HyperTuning, StableRNGs, Distributions, SmoothingSplines, ColorSchemes, JLD2, ProgressLogging #, Combinatorics

	# Packages under development
	using DataDrivenDiffEq, DataDrivenSparse

	# Import utils function
	include("esindy_utils.jl")
	import .ESINDyModule
	using .ERKModule
	using .NFBModule
	
	# Set a random seed for reproducibility
	rng = StableRNG(1111)

	gr()
end

# ╔═╡ 6024651c-85f6-4e53-be0f-44f658cf9c77
md"""
##### Environment Set-up 
"""

# ╔═╡ 9ebfadf0-b711-46c0-b5a2-9729f9e042ee
md"""
# Equation discovery with E-SINDy
This notebook contains the code to implement a version of the E-SINDy algorithms inspired from "Discovering governing equations from data by sparse identification of nonlinear dynamical systems" by Brunton et al. (2016), [DOI](https://doi.org/10.1098/rspa.2021.0904). It was designed to estimate the algebraic formula of an unknown function. In this work, the function corresponds to unknown parts of an ODE system that were approximated by a neural network. The ERK signalling pathway described by Ryu et al. and its corresponding ODE system was used to generate the data.  

Reference: "Frequency modulation of ERK activation dynamics rewires cell fate" by H. Ryu et al. (2015) [DOI](https://doi.org/10.15252/msb.20156458)).
"""

# ╔═╡ d6b1570f-b98b-4a98-b5e5-9d747930a5ab
md"""
##### Import the data
"""

# ╔═╡ 6c7929f5-15b2-4e19-8c26-e709f0da182e
# Create E-SINDy compatible data structure out of dataframes
function create_erk_data(files, gf; smoothing=0., var_idxs=[2,4])

	# Load data into dataframe
	df = CSV.read("./Data/$(files[1])", DataFrame)
	if length(files) > 1
	    for i in 2:length(files)
	        df2 = CSV.read("./Data/$(files[i])", DataFrame)
	        df = vcat(df, df2)
	    end
	end

	# Retrieve the number of samples
	n_samples = sum(df.time .== 0)
	size_sample = Int(nrow(df) / n_samples)

	# Define relevant data for E-SINDy
	time = df.time
	X = [df.R_fit df.Raf_fit df.ERK_fit df.PFB_fit df.NFB_fit]
	if lowercase(gf) == "ngf"
		GT = (0.75 .* df.PFB_fit .* 
			(1 .- df.Raf_fit) ./ (0.01 .+ (1 .- df.Raf_fit)))
	else
		GT = repeat([0], length(df.time))
	end
	Y = df.NN_approx

	# Smooth the NN approximation if necessary
	if smoothing != 0.
		smoothed_Y = ERKModule.smooth_nn_output(time, Y, n_samples, size_sample, smoothing)
		Y = smoothed_Y
	end

	@assert size(X, 1) == size(Y, 1)

	# Create labels for plotting
	labels, case = ERKModule.make_labels(files)
	
	return (time=time, X=X[:,var_idxs], Y=Y, GT=GT, labels=labels, case=case)
end

# ╔═╡ d9b1a318-2c07-4d44-88ec-501b2cfc940b
begin
	# Load the NGF model estimations for various pulse regimes
	ngf_files = ["ngf_lowcc_3m_3v.csv" "ngf_lowcc_3m_20v.csv" "ngf_lowcc_10m.csv" "ngf_lowcc_10m_10v.csv" "ngf_highcc_3m_3v.csv" "ngf_highcc_3m_20v.csv" "ngf_highcc_10m.csv" "ngf_highcc_10m_10v.csv"]
	ngf_data = create_erk_data(ngf_files, "NGF", smoothing=300., var_idxs=1:5)
end

# ╔═╡ c8841434-a7af-4ade-a444-e6bc47575811
begin
	# Load the EGF model estimations for various pulse regimes
	egf_files = ["egf_lowcc_3m_3v.csv" "egf_lowcc_3m_20v.csv" "egf_lowcc_10m.csv" "egf_lowcc_10m_10v.csv" "egf_highcc_3m_3v.csv" "egf_highcc_3m_20v.csv" "egf_highcc_10m.csv" "egf_highcc_10m_10v.csv"]
	egf_data = create_erk_data(egf_files, "EGF", smoothing=300., var_idxs=1:5)
end

# ╔═╡ 74ad0ae0-4406-4326-822a-8f3e027077b3
md"""
##### Set up the SINDy library of functions
"""

# ╔═╡ 9dc0a251-a637-4144-b32a-7ebf5b86b6f3
begin
	# Declare necessary symbolic variables for the bases
	@ModelingToolkit.variables x[1:7] i[1:1]
	x = collect(x)
	i = collect(i)
end

# ╔═╡ 0f5e0785-2778-4fde-b010-7fcf813ceed2
begin
	erk_h = DataDrivenDiffEq.polynomial_basis(x[1:5], 2)
	erk_basis = DataDrivenDiffEq.Basis([erk_h; erk_h .* i], x[1:5], implicits=i[1:1])
end

# ╔═╡ 219b794d-9f51-441b-9197-b8ef0c4495b4
md"""
##### Set up the hyperparameter optimisation
"""

# ╔═╡ 4646bfce-f8cc-4d24-b461-753daff50f71
begin
	# Define a sampling method and the options for the data driven problem
	sampler = DataDrivenDiffEq.DataProcessing(split = 0.8, shuffle = true, batchsize = 400)
	options = DataDrivenDiffEq.DataDrivenCommonOptions(data_processing = sampler, digits=1, abstol=1e-10, reltol=1e-10, denoise=true)
end

# ╔═╡ 034f5422-7259-42b8-b3b3-e34cfe47b7b7
# Define an objective function for hyperparameter optimisation
function objective(trial, dd_prob, basis, with_implicits)
	@unpack λ, ν = trial

	if with_implicits
		res_dd = DataDrivenDiffEq.solve(
			dd_prob, 
			basis, 
			ImplicitOptimizer(DataDrivenSparse.SR3(λ, ν)), 
			options = options)
	else
		res_dd = DataDrivenDiffEq.solve(
			dd_prob, 
			basis, 
			DataDrivenSparse.SR3(λ, ν), 
			options = options)
	end
	
	return bic(res_dd)
end

# ╔═╡ 0f7076ef-848b-4b3c-b918-efb6419787be
# Hyperparameter optimisation function
function get_best_hyperparameters(dd_prob, basis, with_implicits)

	# Define the range of hyperparameters to consider
	scenario = Scenario(λ = 1e-3:1e-3:9e-1, ν = exp10.(-3:1:3), max_trials = 500, 
		sampler=HyperTuning.RandomSampler())

	# Find optimal hyperparameters
	hp_res = HyperTuning.optimize(trial -> objective(trial, dd_prob, basis, with_implicits), scenario)

	return hp_res.best_trial.values[:λ], hp_res.best_trial.values[:ν]
end

# ╔═╡ 5ef43425-ca26-430c-a62d-e194a8b1aebb
md"""
##### E-SINDy Utility Functions
"""

# ╔═╡ c7e825a3-8ce6-48fc-86ac-21810d32bbfb
# Bootstrapping function that estimate optimal library coefficients given data
function sindy_bootstrap(data, basis, n_bstraps, data_fraction)

	# Initialise the coefficient array
	n_eqs = size(data.Y, 2)
	l_basis = length(basis)
	bootstrap_coef = zeros(n_bstraps, n_eqs, l_basis)

	# Track best hyperparameters
	hyperparam = (λ = [], ν = [])

	@info "E-SINDy Bootstrapping:"
	@progress name="Bootstrapping" threshold=0.01 for i in 1:n_bstraps

		# Sample data from the sets of measurements
		rand_ind = rand(1:(length(data.time)), floor(Int, length(data.time) * data_fraction))
		X = data.X[rand_ind,:]
		Y = data.Y[rand_ind,:]

		# Check if the problem involves implicits
		implicits = implicit_variables(basis)
		with_implicits = false
		if !isempty(implicits)
			with_implicits = true
		end
		
		if with_implicits
			for eq in 1:n_eqs
				dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(X', Y[:,eq:eq]')
				
				# Solve problem with optimal hyperparameters
				best_λ, best_ν = get_best_hyperparameters(dd_prob, basis, with_implicits)
				
				push!(hyperparam.λ, best_λ)
				push!(hyperparam.ν, best_ν)
				
				best_res = DataDrivenDiffEq.solve(dd_prob, basis, ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), options=options)

				# Simplify the symbolic expression to get the final coefficients
				simpl_coefs = ESINDyModule.simplify_expression(best_res.out[1].coefficients[1,:], basis)
				
				# Store library coefficient for current bootstrap
				bootstrap_coef[i,eq,:] = simpl_coefs
			end
		else
			dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(X', Y')
			
			# Solve problem with optimal hyperparameters
			best_λ, best_ν = get_best_hyperparameters(dd_prob, basis, with_implicits)
			push!(hyperparam.λ, best_λ)
			push!(hyperparam.ν, best_ν)
			best_res = DataDrivenDiffEq.solve(dd_prob, basis, DataDrivenSparse.SR3(best_λ, best_ν), options = options)
			
			# Store library coefficient for current bootstrap
			bootstrap_coef[i,:,:] = best_res.out[1].coefficients
		end	
	end
	return bootstrap_coef, hyperparam
end

# ╔═╡ 79b03041-01e6-4d8e-b900-446511c81058
# Bootstrapping function that estimate optimal library terms given data
function library_bootstrap(data, basis, n_bstraps, n_libterms; implicit_id=nothing, hyperparameters=nothing)

	# Initialise the coefficient array
	n_eqs = size(data.Y, 2)
	l_basis = length(basis)
	bootstrap_coef = zeros(n_bstraps, n_eqs, l_basis)
	indices = [1]
	best_bic = 1000

	# Check if the problem involves implicits
	implicits = getfield(basis, :implicit)
	with_implicits = false

	@info "Library E-SINDy Bootstrapping:"
	@progress name="Bootstrapping" threshold=0.01 for j in 1:n_bstraps
		println(j)
		for eq in 1:n_eqs
			println(eq)

			# Create bootstrap library basis
			if !isempty(implicits)
				with_implicits = true
				idxs = [1:(implicit_id-1); (implicit_id+1):l_basis]
				rand_ind = [sample(idxs, n_libterms, replace=false); implicit_id]
				bt_basis = DataDrivenDiffEq.Basis(basis[rand_ind], x[1:size(data.X, 2)], implicits=i)
			else
				rand_ind = sample(1:l_basis, n_libterms, replace=false)
				bt_basis = DataDrivenDiffEq.Basis(basis[rand_ind], x[1:size(data.X, 2)])
			end
	
			# Solve data-driven problem with optimal hyperparameters
			dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(data.X', data.Y[:,eq]')
			if !isnothing(hyperparameters)
				best_λ, best_ν = hyperparameters
			else
				best_λ, best_ν = get_best_hyperparameters(dd_prob, bt_basis, with_implicits)
			end
			if with_implicits
				best_res = DataDrivenDiffEq.solve(dd_prob, bt_basis, ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), options=options)
				
			else
				best_res = DataDrivenDiffEq.solve(dd_prob, bt_basis, DataDrivenSparse.SR3(best_λ, best_ν), options = options)
			end

			# Check if bootstrap basis is optimal
			bt_bic = bic(best_res)
			if bt_bic < best_bic
				best_bic = bt_bic
				bootstrap_coef[indices,eq:eq,:] = zeros((length(indices),1,l_basis))
				bootstrap_coef[j,eq:eq,rand_ind] = best_res.out[1].coefficients
				empty!(indices)
				push!(indices, j)
			elseif bt_bic == best_bic
				bootstrap_coef[j,eq:eq,rand_ind] = best_res.out[1].coefficients
				push!(indices, j)
			end
		end
	end
	return bootstrap_coef 
end

# ╔═╡ 2104057c-9687-4fca-ac73-da2d711c9741
# ╠═╡ disabled = true
#=╠═╡
# Function to simplify the expression of the final equation returned by SINDy (limit redundancy of expressions)
function coef_simplify_expression(coefs, basis)
	
	# Get only right hand side of the library terms
	h = [equation.rhs for equation in equations(basis)]

	# Get the expression out of the coefficients and the basis
	expression = sum(coefs .* h)

	# Solve wrt the implicit variable and simplify
	implicit = getfield(basis, :implicit)
	implicit_id = findall(x -> (x == 0.1), [substitute(term.rhs, Dict([i[1] => 0.1, x[1] => 2, x[2] => 3, x[3] => 4, x[4] => 5, x[5]=> 6])) for term in basis])
	simpl_expr = simplify(ModelingToolkit.solve_for(expression .~ 0, implicit)[1])

	# Update the coefficients wrt to the simplified expression
	simpl_coefs = zeros(size(coefs))
	for (j, term) in enumerate(basis)
		try
			simpl_coef = Symbolics.coeff(simpl_expr, term.rhs)
			if isempty(Symbolics.get_variables(simpl_coef))
				if j == implicit_id & simpl_coef == 0
					simpl_coefs[j] = -1
				else
					simpl_coefs[j] = simpl_coef
				end
			end
		catch
			simpl_coefs[j] = coefs[j]
		end
	end
	return simpl_coefs
end
  ╠═╡ =#

# ╔═╡ 85c913fa-23be-4d01-9a78-47f786d9a4cd
# Function to identify the candidate equations returned during E-SINDy
function get_coef_mask(bootstraps)
	
	# Repeat for each equation
	n_eqs = size(bootstraps, 2)
	eq_masks = []
	for eq in 1:n_eqs

		# Initialise set of masks
		masks = []
		mask_set = Set()

		# Define mask corresponding to non-zero coefficients
		n_bootstraps = size(bootstraps, 1)
		for k in 1:n_bootstraps
			mask = (!iszero).(bootstraps[k,eq:eq,:])
			if mask in mask_set
				nothing
			else
				# Compute the frequency of current mask
				push!(mask_set, mask)
				freq = 0
				coefs = []
				for j in k:n_bootstraps
					temp_mask = (!iszero).(bootstraps[j,eq:eq,:])
					if isequal(mask, temp_mask)
						push!(coefs, bootstraps[j,eq:eq,:])
						freq = freq + 1
					end
				end

				# Compute mean and std of the coefficients of current mask
				coef_std = std(coefs)
				if any(isnan, coef_std)
					nothing
				else
					push!(masks, (mask=mask, freq=freq/n_bootstraps, coef_mean=mean(coefs), coef_std=coef_std/sqrt(n_bootstraps)))
				end
			end
		end
		push!(eq_masks, masks)
	end
	return eq_masks
end

# ╔═╡ 74b2ade4-884b-479d-9fee-828d37d7ab47
# Function to estimate coefficient statistics
function compute_ecoef(bootstrap_res, coef_threshold)

	# Retrieve dimensions of the problem
	sample_size = size(bootstrap_res, 1)
	n_eqs = size(bootstrap_res, 2)
	n_terms = size(bootstrap_res, 3)

	# Compute inclusion probabilities
	inclusion_prob = (mean((bootstrap_res .!= 0), dims=1) * 100)

	# Keep only elements of basis with probabilities above threshold
	mask = inclusion_prob .> coef_threshold
	masked_res = bootstrap_res .* mask

	# Compute the mean 
	m = zeros(Float64, n_eqs, n_terms)
	for i in 1:n_eqs
		for j in 1:n_terms
			current_coef = filter(!iszero, masked_res[:,i,j])
			if !isempty(current_coef)
				m[i,j] = median(current_coef) 
			end
		end
	end
	return m
end

# ╔═╡ 97ae69f0-764a-4aff-88cd-91accf6bb3fd
# ╠═╡ disabled = true
#=╠═╡
# Function to build callable function out of symbolic equations
function build_equations(coef, basis; verbose=true)

	# Build equation
	h = [equation.rhs for equation in DataDrivenDiffEq.equations(basis)]
	eqs = [sum(row .* h) for row in eachrow(coef)]
	final_eqs = []
	
	# Solve equation wrt the implicit variable if any
	implicits = getfield(basis, :implicit)
	if !isempty(implicits)
		for eq in 1:size(eqs, 1)
			try
				solution = ModelingToolkit.solve_for(eqs[eq] .~ 0, implicits)[1]
				push!(final_eqs, solution)
			catch exception
				push!(final_eqs, NaN)
			end
		end
	else
		final_eqs = eqs
	end

	# Build callable function and print equation
	y = []
	for (j, eq) in enumerate(final_eqs)
		if verbose
			println("y$(j) = $(eq)")
		end
		push!(y, Symbolics.build_function(eq, [x[1:5]; i[1]], expression = Val{false}))
	end
	
	return y
end
  ╠═╡ =#

# ╔═╡ aefb3b75-af6a-4bc9-98f8-713144b24c5a
# Function to compute the interquartile range of the estimated equation
function compute_CI(data, basis, masks)

	freqs = Weights([mask.freq for mask in masks])

	# Run MC simulations to estimate the distribution of estimated equations 
	n_simulations = 1000
	results = zeros(n_simulations, size(data.Y, 1), size(data.Y, 2))
	
	for i in 1:n_simulations
		current_coef = zeros(size(masks[1].coef_mean))
		mask = sample(masks, freqs)
		
		# Build Normal distribution for non-zero coefficients
		indices = findall(!iszero, mask.mask)
		mask.coef_std[iszero.(mask.coef_std)] .= 1e-12
		coef_distrib = [Normal(mask.coef_mean[k], mask.coef_std[k]) for k in indices]

		# Samples coefficient from the distribution
		coef_sample = [rand(distrib) for distrib in coef_distrib]
		current_coef[indices] = coef_sample

		# Calculate function value given sample of coef.
		current_eqs = ESINDyModule.build_equations(current_coef, basis, verbose=false)
		yvals = ESINDyModule.get_yvals(data, current_eqs)
		n_eqs = length(current_eqs)
		for eq in 1:n_eqs
			results[i,:,eq] = yvals[eq]
		end
	end
	iqr_low = mapslices(row -> percentile(filter(!isnan, row), 25), results, dims=1)
	iqr_up = mapslices(row -> percentile(filter(!isnan, row), 75), results, dims=1)

    return (iqr_low=iqr_low[1,:,:], iqr_up=iqr_up[1,:,:])
end

# ╔═╡ cc2395cc-00df-4e5e-8573-e85ce813fd41
# Plotting function for E-SINDy results
function plot_esindy(results; sample_idxs=nothing, iqr=true)

	# Retrieve results
	data, basis = results.data, results.basis
	coef_median = results.coef_median
	
	eqs = ESINDyModule.build_equations(coef_median, basis, verbose=false)
	y_vals = ESINDyModule.get_yvals(data, eqs)

	# Retrieve the indices of samples to plot
	n_samples = sum(data.time .== 0)
	size_sample = Int(length(data.time) / n_samples)
	if isnothing(sample_idxs)
		sample_idxs = 1:n_samples
	end
	
	# Plot results
	n_eqs = size(eqs, 1)
	subplots = []
	palette = colorschemes[:seaborn_colorblind]
	for eq in 1:n_eqs
		
		# Compute interquartile range if necessary
		ci_low, ci_up = nothing, nothing
		if iqr
			ci_low, ci_up = compute_CI(data, basis, results.masks[eq])
		end

		i_color = 1
		if n_eqs > 1
			p = plot(title="Equation $(eq)", xlabel="Time t", ylabel="Equation y(t)")
		else
			p = plot(title="", xlabel="Time", ylabel="Equation y(t)")
		end

		# Plot each sample separately
		for sample in 0:(n_samples-1)
			if (sample+1) in sample_idxs
				i_start = 1 + sample * size_sample
				i_end = i_start + size_sample - 1
				
				y = y_vals[eq][i_start:i_end] 
				if iqr
					plot!(p, data.time[i_start:i_end], y, label=data.labels[sample+1], color=palette[i_color],
					ribbon=(y - ci_low[i_start:i_end, eq], ci_up[i_start:i_end, eq] - y), fillalpha=0.15)
				else
					plot!(p, data.time[i_start:i_end], y, label=data.labels[sample+1], color=palette[i_color])
				end
				
				plot!(p, data.time[i_start:i_end], data.GT[i_start:i_end, eq], label="", linestyle=:dash, color=palette[i_color])
				i_color = i_color + 1
			end
		end
		plot!(p, [], [],  label="GT", color=:black, linestyle=:dash)
		push!(subplots, p)
	end
	return subplots
end

# ╔═╡ 012a5186-03aa-482d-bb62-ecba49587877
# Complete E-SINDy function 
function esindy(data, basis, n_bstrap=100; coef_threshold=65, data_fraction=1)

	# Run sindy bootstraps
	bootstrap_res, hyperparameters = sindy_bootstrap(data, basis, n_bstrap, data_fraction)

	# Compute the masks
	masks = get_coef_mask(bootstrap_res)
	
	# Compute the mean and std of ensemble coefficients
	e_coef = compute_ecoef(bootstrap_res, coef_threshold)

	# Build the final equation as callable functions
	println("E-SINDy estimated equations:")
	y = ESINDyModule.build_equations(e_coef, basis)
	
	return (data=data, basis=basis, bootstraps=bootstrap_res, coef_median=e_coef,  hyperparameters=hyperparameters, masks=masks)
end

# ╔═╡ 569c7600-246b-4a64-bba5-1e74a5888d8c
md"""
### ERK model
"""

# ╔═╡ e49b5f76-d04b-4b19-a4c8-a3a3a1b40f1b
md"""
#### NGF case
"""

# ╔═╡ 1f3f1461-09f1-4825-bb99-4064e075e23e
md"""
###### Run E-SINDy
"""

# ╔═╡ b73f3450-acfd-4b9e-9b7f-0f7289a62976
# ╠═╡ disabled = true
#=╠═╡
begin
	ngf_res = esindy(ngf_data, erk_basis, 100, coef_threshold=80, data_fraction=0.5)
end
  ╠═╡ =#

# ╔═╡ 8e71a2ec-c4cd-45bb-8537-76096ef0d0cb
md"""
###### Run Library E-SINDy
This section provides an alternative to b(r)agging E-SINDy in case of a large library. It first attempts at finding the most relevant terms of the library. Eventually, it speeds up the computations if the best set of hyperparameters have already been explored (smaller run of E-SINDy to identify them) and are specified.
"""

# ╔═╡ df7f0f7e-1bfa-4d7e-a817-8dc20a3ec0c4
begin
	# Run first library E-SINDy to reduce the library size
	#ngf_lib_res = library_bootstrap(ngf_data, erk_basis, 5000, 10, implicit_id=22, hyperparameters=(0.3, 10))
	#ngf_lib_basis = ESINDyModule.build_basis(ngf_lib_res, erk_basis)
	
	# Run E-SINDy with resulting library
	#ngf_res_full_lib = esindy(ngf_data, ngf_lib_basis, 100, coef_threshold=80)
end

# ╔═╡ 02c625dc-9d21-488e-983b-c3e2c40e0aad
md"""
###### Save or load the results (JLD2 file)
"""

# ╔═╡ 04538fbf-63b7-4394-b281-f047d0c3ea51
begin
	#jldsave("./Data/ngf_esindy_100bt.jld2"; results=ngf_res)
	#ngf_res = load("./Data/ngf_esindy_100bt.jld2")["results"]
end

# ╔═╡ 62d79e8d-6d82-4fea-a596-8c32dd16d78e
begin
	#jldsave("./Data/ngf_esindy_100bt_lib.jld2"; results=ngf_res_lib_up2)
	ngf_res_lib = load("./Data/ngf_esindy_100bt_lib.jld2")["results"]
end

# ╔═╡ 3bd630ad-f546-4b0b-8f3e-98367de739b1
md"""
###### Plot the results
"""

# ╔═╡ 5c400147-353a-43f2-9de6-c234e13f06c9
#=╠═╡
begin
	ngf_plot = plot_esindy(ngf_res, sample_idxs=1:8, iqr=false)[1]
	plot(ngf_plot, title="E-SINDy results for ERK model\nafter NGF stimulation\n", size=(800, 600), legend_position=:topright)
	#savefig("./Plots/ngf_esindy_100bt.svg")
end
  ╠═╡ =#

# ╔═╡ de4c57fe-1a29-4354-a5fe-f4e184de4dd3
begin
	ngf_plot_lib = plot_esindy(ngf_res_lib.esindy, sample_idxs=1:8, iqr=false)[1]
	plot(ngf_plot_lib, title="E-SINDy results for ERK model\nafter NGF stimulation\n", size=(800, 600), legend_position=:topright)
	#savefig("./Plots/ngf_esindy_100bt_lib.svg")
end

# ╔═╡ 03993d5c-7900-454a-9fe3-a70bd69456d1
md"""
#### EGF case
"""

# ╔═╡ e675346b-99dc-441f-913b-19387aaef3ea
md"""
###### Run E-SINDy
"""

# ╔═╡ 50d20cc1-ce44-43b6-9d72-1ff36137f6f8
#egf_res = esindy(egf_data, erk_basis, 100, coef_threshold=80, data_fraction=0.5)

# ╔═╡ 1cc0a861-8c92-4a05-85b6-6b1dfe406f2e
md"""
###### Run Library E-SINDy
This section provides an alternative to b(r)agging E-SINDy in case of a large library. It first attempts at finding the most relevant terms of the library. Eventually, it speeds up the computations if the best set of hyperparameters have already been explored (smaller run of E-SINDy to identify them) and are specified.
"""

# ╔═╡ 2e472b7f-0863-470e-a91a-13aafd12ce0b
begin 
	# Run first library E-SINDy to reduce the library size
	#egf_lib_res = library_bootstrap(egf_data, erk_basis, 5000, 10, implicit_id=22, )
	#egf_lib_basis = ESINDyModule.build_basis(egf_lib_res, erk_basis)

	# Run E-SINDy with resulting library
	#egf_res_lib = esindy(egf_data, egf_lib_basis, 100, coef_threshold=80)
end

# ╔═╡ 8f7e784d-1f35-49bf-ba73-d2306453e258
md"""
###### Save or load the results (JLD2 file)
"""

# ╔═╡ 65732e07-69dd-43d5-b04c-72444c78c470
begin
	#jldsave("./Data/egf_esindy_100bt.jld2"; results=egf_res_up)
	egf_res = load("./Data/egf_esindy_100bt.jld2")["results"]
end

# ╔═╡ 541c0d3f-d78e-49fd-b064-3b7fabc35d43
begin
	#jldsave("./Data/egf_esindy_100bt_lib.jld2"; results=egf_res_lib)
	egf_res_lib = load("./Data/egf_esindy_100bt_lib.jld2")["results"]
end

# ╔═╡ 664b3e67-09d3-4e98-a946-ffbcbed0fe90
begin
	egf_plot = plot_esindy(egf_res, iqr=false)[1]
	plot(egf_plot, title="E-SINDy results for ERK model\nafter EGF stimulation\n", size=(800, 600), legend_position=:topright, ylim=(-0.05, .1))
	#savefig("./Plots/egf_esindy_100bt.svg")
end

# ╔═╡ 2672fac7-df22-4445-887a-5404b74fc4b1
begin
	egf_lib_plot = plot_esindy(egf_res_lib.esindy, iqr=false)[1]
	plot(egf_lib_plot, title="E-SINDy results for ERK model\nafter EGF stimulation\n", size=(800, 600), legend_position=:topright, ylim=(-0.05, .1))
	#savefig("./Plots/egf_esindy_100bt_lib.svg")
end

# ╔═╡ Cell order:
# ╟─6024651c-85f6-4e53-be0f-44f658cf9c77
# ╠═a806fdd2-5017-11ef-2351-dfc89f69b334
# ╟─9ebfadf0-b711-46c0-b5a2-9729f9e042ee
# ╟─d6b1570f-b98b-4a98-b5e5-9d747930a5ab
# ╟─6c7929f5-15b2-4e19-8c26-e709f0da182e
# ╟─d9b1a318-2c07-4d44-88ec-501b2cfc940b
# ╟─c8841434-a7af-4ade-a444-e6bc47575811
# ╟─74ad0ae0-4406-4326-822a-8f3e027077b3
# ╟─9dc0a251-a637-4144-b32a-7ebf5b86b6f3
# ╟─0f5e0785-2778-4fde-b010-7fcf813ceed2
# ╟─219b794d-9f51-441b-9197-b8ef0c4495b4
# ╟─4646bfce-f8cc-4d24-b461-753daff50f71
# ╠═034f5422-7259-42b8-b3b3-e34cfe47b7b7
# ╠═0f7076ef-848b-4b3c-b918-efb6419787be
# ╟─5ef43425-ca26-430c-a62d-e194a8b1aebb
# ╠═c7e825a3-8ce6-48fc-86ac-21810d32bbfb
# ╟─79b03041-01e6-4d8e-b900-446511c81058
# ╠═2104057c-9687-4fca-ac73-da2d711c9741
# ╟─85c913fa-23be-4d01-9a78-47f786d9a4cd
# ╟─74b2ade4-884b-479d-9fee-828d37d7ab47
# ╟─97ae69f0-764a-4aff-88cd-91accf6bb3fd
# ╟─aefb3b75-af6a-4bc9-98f8-713144b24c5a
# ╟─cc2395cc-00df-4e5e-8573-e85ce813fd41
# ╟─012a5186-03aa-482d-bb62-ecba49587877
# ╟─569c7600-246b-4a64-bba5-1e74a5888d8c
# ╟─e49b5f76-d04b-4b19-a4c8-a3a3a1b40f1b
# ╟─1f3f1461-09f1-4825-bb99-4064e075e23e
# ╠═b73f3450-acfd-4b9e-9b7f-0f7289a62976
# ╟─8e71a2ec-c4cd-45bb-8537-76096ef0d0cb
# ╟─df7f0f7e-1bfa-4d7e-a817-8dc20a3ec0c4
# ╟─02c625dc-9d21-488e-983b-c3e2c40e0aad
# ╠═04538fbf-63b7-4394-b281-f047d0c3ea51
# ╠═62d79e8d-6d82-4fea-a596-8c32dd16d78e
# ╟─3bd630ad-f546-4b0b-8f3e-98367de739b1
# ╟─5c400147-353a-43f2-9de6-c234e13f06c9
# ╟─de4c57fe-1a29-4354-a5fe-f4e184de4dd3
# ╟─03993d5c-7900-454a-9fe3-a70bd69456d1
# ╟─e675346b-99dc-441f-913b-19387aaef3ea
# ╟─50d20cc1-ce44-43b6-9d72-1ff36137f6f8
# ╟─1cc0a861-8c92-4a05-85b6-6b1dfe406f2e
# ╟─2e472b7f-0863-470e-a91a-13aafd12ce0b
# ╟─8f7e784d-1f35-49bf-ba73-d2306453e258
# ╟─65732e07-69dd-43d5-b04c-72444c78c470
# ╟─541c0d3f-d78e-49fd-b064-3b7fabc35d43
# ╟─664b3e67-09d3-4e98-a946-ffbcbed0fe90
# ╟─2672fac7-df22-4445-887a-5404b74fc4b1
