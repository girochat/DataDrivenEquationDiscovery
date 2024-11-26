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
	using StatsBase, Plots, CSV, DataFrames

	# External libraries
	using HyperTuning, StableRNGs, Distributions, SmoothingSplines, ColorSchemes, JLD2, ProgressLogging

	# Data Driven Equation Discovery packages
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
This notebook contains the code to implement a version of the E-SINDy algorithms inspired from "Discovering governing equations from data by sparse identification of nonlinear dynamical systems" by Brunton et al. (2016), [DOI](https://doi.org/10.1098/rspa.2021.0904). It was designed to estimate the algebraic formula of an unknown function. In this work, the function corresponds to unknown parts of an ODE system that were approximated by a neural network. A theoretical negative feedback model of two kinases g1p and g2p presented in "Quantitative Biology: Theory, Computational Methods and Examples of Models" was used to generate the data.  

Reference: Chapter 13: Parameter Estimation, Sloppiness, and Model Identifiability by D. Daniels, M. Dobrzyński, D. Fey (2018) in "Quantitative Biology: Theory, Computational Methods and Examples of Models".
"""

# ╔═╡ d6b1570f-b98b-4a98-b5e5-9d747930a5ab
md"""
##### Import the data
"""

# ╔═╡ 6c7929f5-15b2-4e19-8c26-e709f0da182e
# Create E-SINDy compatible data structure out of dataframes
function create_nfb_data(files, smoothing=0.)

	# Load data into dataframe
	df = CSV.read("./Data/$(files[1])", DataFrame)
	if length(files) > 1
	    for i in 2:length(files)
	        df2 = CSV.read("./Data/$(files[i])", DataFrame)
	        df = vcat(df, df2)
	    end
	end

	# Create labels for plotting and retrieve NFB case
	labels, case = NFBModule.make_labels(files)

	# Retrieve the number of samples
	n_samples = sum(df.time .== 0)
	size_sample = Int(nrow(df) / n_samples)

	# Define relevant data for E-SINDy
	time = df.time
	X = [df.g1p_fit df.g2p_fit df.Xact_fit]
	Y = [df.NN1 df.NN2]

	# Smooth the NN approximation if necessary
	if smoothing != 0 
		smoothed_Y1 = smooth_nn_output(time, df.NN1, n_samples, size_sample, smoothing)
		smoothed_Y2 = smooth_nn_output(time, df.NN2, n_samples, size_sample, smoothing)
		Y = [smoothed_Y1 smoothed_Y2]
	end

	GT = 0
	if case == "a"
		GT = [df.Xact_fit repeat([1], size(df.Xact_fit, 1))]
	elseif case == "b"
		GT = [repeat([1], size(df.Xact_fit, 1)) df.Xact_fit]
	elseif case == "nofb"
		GT = [repeat([1], size(time, 1)) repeat([1], size(time, 1))]
	elseif case == "ab"
		GT = [df.Xact_fit df.Xact_fit]
	end

	@assert size(X, 1) == size(Y, 1)
		
	return (time=time, X=X, Y=Y, GT=GT, labels=labels, case=case)
end

# ╔═╡ d9b1a318-2c07-4d44-88ec-501b2cfc940b
begin
	# Load the NFB model estimations for various concentrations
	nofb_files = ["nfb_nofb_001.csv" "nfb_nofb_002.csv" "nfb_nofb_003.csv" "nfb_nofb_004.csv" "nfb_nofb_005.csv" "nfb_nofb_006.csv" "nfb_nofb_007.csv" "nfb_nofb_008.csv" "nfb_nofb_009.csv"]
	nofb_data = create_nfb_data(nofb_files, )
end

# ╔═╡ c8841434-a7af-4ade-a444-e6bc47575811
begin
	a_files = ["nfb_a_001.csv" "nfb_a_002.csv" "nfb_a_003.csv" "nfb_a_004.csv" "nfb_a_005.csv" "nfb_a_006.csv" "nfb_a_007.csv" "nfb_a_008.csv" "nfb_a_009.csv"]
	a_data = create_nfb_data(a_files)
end

# ╔═╡ e24504e0-37a3-4d08-9118-985f2293f0ee
begin
	b_files = ["nfb_b_001.csv" "nfb_b_002.csv" "nfb_b_003.csv" "nfb_b_004.csv" "nfb_b_005.csv" "nfb_b_006.csv" "nfb_b_007.csv" "nfb_b_008.csv" "nfb_b_009.csv"]
	b_data = create_nfb_data(b_files)
end

# ╔═╡ ba67edf9-df81-4123-a69d-246be652d419
begin
	ab_files = ["nfb_ab_001.csv" "nfb_ab_002.csv" "nfb_ab_003.csv" "nfb_ab_004.csv" "nfb_ab_005.csv" "nfb_ab_006.csv" "nfb_ab_007.csv" "nfb_ab_008.csv" "nfb_ab_009.csv"]
	ab_data = create_nfb_data(ab_files)
end

# ╔═╡ 74ad0ae0-4406-4326-822a-8f3e027077b3
md"""
##### Set up the SINDy library of functions
"""

# ╔═╡ 9dc0a251-a637-4144-b32a-7ebf5b86b6f3
begin
	# Declare necessary symbolic variables for the bases
	@ModelingToolkit.variables x[1:3] i[1:1]
	x = collect(x)
	i = collect(i)
end

# ╔═╡ 0f5e0785-2778-4fde-b010-7fcf813ceed2
begin
	# Define a basis of functions to estimate the unknown equation of NFB model
	nfb_h = DataDrivenDiffEq.monomial_basis(x[1:3], 2)
	nfb_basis = DataDrivenDiffEq.Basis(nfb_h, x[1:3])
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

# ╔═╡ aefb3b75-af6a-4bc9-98f8-713144b24c5a
# Function to compute the interquartile range of the estimated equation
function compute_CI(data, basis, masks, ecoef_mask)
	
	# Run MC simulations to estimate the distribution of estimated equations 
	n_eqs = size(data.Y, 2)
	n_simulations = 1000
	results = zeros(n_simulations, size(data.Y, 1), n_eqs) 
	for eq in 1:n_eqs

		# Keep only mask that share coefficient with the median coefficient
		up_masks = [mask for mask in masks[eq] if any(mask.mask .== ecoef_mask[eq:eq,:])]
		freqs = Weights([mask.freq for mask in up_masks])
		
		for i in 1:n_simulations			
			current_coef = zeros(size(up_masks[1].coef_mean))
			mask = sample(up_masks, freqs)
			current_mask = mask.mask .& ecoef_mask[eq:eq,:]
			
			# Build Normal distribution for non-zero coefficients
			indices = findall(!iszero, current_mask)
			mask.coef_std[iszero.(mask.coef_std)] .= 1e-12
			coef_distrib = [Normal(mask.coef_mean[k], mask.coef_std[k]) for k in indices]
	
			# Samples coefficient from the distribution
			coef_sample = [rand(distrib) for distrib in coef_distrib]
			current_coef[indices] = coef_sample
	
			# Calculate function value given sample of coef.
			current_eqs = ESINDyModule.build_equations(current_coef, basis, verbose=false)
			yvals = ESINDyModule.get_yvals(data, current_eqs)
			results[i,:,eq] = yvals[1]
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

	# Compute interquartile range if necessary
	ci_low, ci_up = nothing, nothing
	if iqr
		ecoef_mask = (!iszero).(coef_median)
		ci_low, ci_up = compute_CI(data, basis, results.masks, ecoef_mask)
	end
	
	# Plot results
	n_eqs = size(eqs, 1)
	subplots = []
	palette = colorschemes[:seaborn_colorblind]
	for eq in 1:n_eqs
		
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
	
	return (data=data, 
	basis=basis, 
	bootstraps=bootstrap_res, 
	coef_median=e_coef, 
	hyperparameters=hyperparameters, 
	masks=masks)
end

# ╔═╡ 569c7600-246b-4a64-bba5-1e74a5888d8c
md"""
### NFB results
"""

# ╔═╡ e49b5f76-d04b-4b19-a4c8-a3a3a1b40f1b
md"""
#### Case a
"""

# ╔═╡ 1f3f1461-09f1-4825-bb99-4064e075e23e
md"""
###### Run E-SINDy
"""

# ╔═╡ b73f3450-acfd-4b9e-9b7f-0f7289a62976
#a_res = esindy(a_data, nfb_basis, 100, coef_threshold=85, data_fraction=0.5)

# ╔═╡ 02c625dc-9d21-488e-983b-c3e2c40e0aad
md"""
###### Save or load the results (JLD2 file)
"""

# ╔═╡ 04538fbf-63b7-4394-b281-f047d0c3ea51
begin
	#jldsave("./Data/nfb_esindy_100bt_a.jld2"; results=a_res)
	a_res = load("./Data/nfb_esindy_100bt_a.jld2")["results"]
end

# ╔═╡ 3bd630ad-f546-4b0b-8f3e-98367de739b1
md"""
###### Plot the results
"""

# ╔═╡ 5c400147-353a-43f2-9de6-c234e13f06c9
begin
	# Plot the results
	a_plots = plot_esindy(a_res, sample_idxs=1:2:9, iqr=false)
	a_plot2 = plot(a_plots[2], ylim=(0, 2))
	plot(a_plots[1], a_plot2, layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB on g1p")
	#savefig("./Plots/NFB_esindy_a.svg")
end

# ╔═╡ 03993d5c-7900-454a-9fe3-a70bd69456d1
md"""
#### Case b
"""

# ╔═╡ e675346b-99dc-441f-913b-19387aaef3ea
md"""
###### Run E-SINDy
"""

# ╔═╡ 50d20cc1-ce44-43b6-9d72-1ff36137f6f8
#b_res = esindy(b_data, nfb_basis, 100, coef_threshold=85, data_fraction=0.5)

# ╔═╡ 8f7e784d-1f35-49bf-ba73-d2306453e258
md"""
###### Save or load the results (JLD2 file)
"""

# ╔═╡ 65732e07-69dd-43d5-b04c-72444c78c470
begin
	#jldsave("./Data/nfb_esindy_100bt_b.jld2"; results=b_res)
	b_res = load("./Data/nfb_esindy_100bt_b.jld2")["results"]
end

# ╔═╡ 40709d6d-5970-4500-8dde-214b14191641
md"""
###### Plot the results
"""

# ╔═╡ 664b3e67-09d3-4e98-a946-ffbcbed0fe90
begin
	# Plot the results
	b_plots = plot_esindy(b_res, sample_idxs=1:2:9, iqr=false)
	b_plot1 = plot(b_plots[1], ylim=(0, 2))
	plot(b_plot1, b_plots[2], layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB on g2p")
	#savefig("./Plots/nfb_esindy_b.svg")
end

# ╔═╡ 7d05f90a-64a3-4621-8be3-f6ad815dab7d
md"""
#### Case ab
"""

# ╔═╡ 5b003a9f-e756-4993-a16b-bd1423420989
md"""
###### Run E-SINDy
"""

# ╔═╡ 3ec98cad-5fdb-440f-bca4-dd611c5966fa
#ab_res = esindy(ab_data, nfb_basis, 100, coef_threshold=85, data_fraction=0.5)

# ╔═╡ f28931af-d7ce-47b9-aa43-18341466949a
md"""
###### Save or load the results (JLD2 file)
"""

# ╔═╡ 38d67f7a-9b23-470c-b410-8c6dead1a9e2
begin
	#jldsave("./Data/nfb_esindy_100bt_ab.jld2"; results=ab_res)
	ab_res = load("./Data/nfb_esindy_100bt_ab.jld2")["results"]
end

# ╔═╡ 4329a4c5-c315-4052-a3ee-5e0d4d62c265
md"""
###### Plot the results
"""

# ╔═╡ 07a91981-34fd-4931-b614-278e484fcd84
begin
	# Plot the results
	ab_plots = plot_esindy(ab_res, sample_idxs=1:2:9, iqr=false)
	plot(ab_plots[1], ab_plots[2], layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB on g1p and g2p")
	#savefig("./Plots/nfb_esindy_ab.svg")
end

# ╔═╡ 55d05ba7-e066-4983-8690-ea69e2080453
md"""
#### Case no FB
"""

# ╔═╡ 31cb4fcb-7e25-4e28-aa15-6d5c900abfb0
md"""
###### Run E-SINDy
"""

# ╔═╡ 62acf3a1-86ec-4d78-985b-5ba061d7ee88
#nofb_res = esindy(nofb_data, nfb_basis, 100, coef_threshold=85, data_fraction=0.5)

# ╔═╡ 194a8ca4-28a2-4fdc-bb87-75c55049f0d4
md"""
###### Save or load the results (JLD2 file)
"""

# ╔═╡ 67d4e01a-b817-4689-8bbe-364d2cd5631c
begin
	#jldsave("./Data/nfb_esindy_100bt_nofb.jld2"; results=nofb_res)
	nofb_res = load("./Data/nfb_esindy_100bt_nofb.jld2")["results"]
end

# ╔═╡ 66ff8331-6d51-493d-8b2b-1519cb4d196f
md"""
###### Plot the results
"""

# ╔═╡ 0bca2a61-892e-47c7-b9ed-eedb33743b8c
begin
	# Plot the results
	nofb_plots = plot_esindy(nofb_res, sample_idxs=1:2:9, iqr=false)
	plot(nofb_plots[1], nofb_plots[2], ylim=(0, 2), layout=(2,1), size=(800, 800), plot_title="E-SINDy results with no FB")
	#savefig("./Plots/nfb_esindy_nofb.svg")
end

# ╔═╡ Cell order:
# ╟─6024651c-85f6-4e53-be0f-44f658cf9c77
# ╠═a806fdd2-5017-11ef-2351-dfc89f69b334
# ╟─9ebfadf0-b711-46c0-b5a2-9729f9e042ee
# ╟─d6b1570f-b98b-4a98-b5e5-9d747930a5ab
# ╟─6c7929f5-15b2-4e19-8c26-e709f0da182e
# ╟─d9b1a318-2c07-4d44-88ec-501b2cfc940b
# ╟─c8841434-a7af-4ade-a444-e6bc47575811
# ╟─e24504e0-37a3-4d08-9118-985f2293f0ee
# ╟─ba67edf9-df81-4123-a69d-246be652d419
# ╟─74ad0ae0-4406-4326-822a-8f3e027077b3
# ╟─9dc0a251-a637-4144-b32a-7ebf5b86b6f3
# ╟─0f5e0785-2778-4fde-b010-7fcf813ceed2
# ╟─219b794d-9f51-441b-9197-b8ef0c4495b4
# ╟─4646bfce-f8cc-4d24-b461-753daff50f71
# ╟─034f5422-7259-42b8-b3b3-e34cfe47b7b7
# ╟─0f7076ef-848b-4b3c-b918-efb6419787be
# ╟─5ef43425-ca26-430c-a62d-e194a8b1aebb
# ╠═c7e825a3-8ce6-48fc-86ac-21810d32bbfb
# ╟─85c913fa-23be-4d01-9a78-47f786d9a4cd
# ╟─74b2ade4-884b-479d-9fee-828d37d7ab47
# ╟─aefb3b75-af6a-4bc9-98f8-713144b24c5a
# ╟─cc2395cc-00df-4e5e-8573-e85ce813fd41
# ╟─012a5186-03aa-482d-bb62-ecba49587877
# ╟─569c7600-246b-4a64-bba5-1e74a5888d8c
# ╟─e49b5f76-d04b-4b19-a4c8-a3a3a1b40f1b
# ╟─1f3f1461-09f1-4825-bb99-4064e075e23e
# ╟─b73f3450-acfd-4b9e-9b7f-0f7289a62976
# ╟─02c625dc-9d21-488e-983b-c3e2c40e0aad
# ╟─04538fbf-63b7-4394-b281-f047d0c3ea51
# ╟─3bd630ad-f546-4b0b-8f3e-98367de739b1
# ╟─5c400147-353a-43f2-9de6-c234e13f06c9
# ╟─03993d5c-7900-454a-9fe3-a70bd69456d1
# ╟─e675346b-99dc-441f-913b-19387aaef3ea
# ╟─50d20cc1-ce44-43b6-9d72-1ff36137f6f8
# ╟─8f7e784d-1f35-49bf-ba73-d2306453e258
# ╟─65732e07-69dd-43d5-b04c-72444c78c470
# ╟─40709d6d-5970-4500-8dde-214b14191641
# ╟─664b3e67-09d3-4e98-a946-ffbcbed0fe90
# ╟─7d05f90a-64a3-4621-8be3-f6ad815dab7d
# ╟─5b003a9f-e756-4993-a16b-bd1423420989
# ╟─3ec98cad-5fdb-440f-bca4-dd611c5966fa
# ╟─f28931af-d7ce-47b9-aa43-18341466949a
# ╟─38d67f7a-9b23-470c-b410-8c6dead1a9e2
# ╟─4329a4c5-c315-4052-a3ee-5e0d4d62c265
# ╟─07a91981-34fd-4931-b614-278e484fcd84
# ╟─55d05ba7-e066-4983-8690-ea69e2080453
# ╟─31cb4fcb-7e25-4e28-aa15-6d5c900abfb0
# ╟─62acf3a1-86ec-4d78-985b-5ba061d7ee88
# ╟─194a8ca4-28a2-4fdc-bb87-75c55049f0d4
# ╟─67d4e01a-b817-4689-8bbe-364d2cd5631c
# ╟─66ff8331-6d51-493d-8b2b-1519cb4d196f
# ╟─0bca2a61-892e-47c7-b9ed-eedb33743b8c
