### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ b9ff1932-20ef-48da-9d2d-d61f3ee13a4f
begin 
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ a806fdd2-5017-11ef-2351-dfc89f69b334
begin
	# SciML tools
	import ModelingToolkit, Symbolics
	
	# Standard libraries
	using StatsBase, Plots, CSV, DataFrames, Printf, Statistics

	# External libraries
	using HyperTuning, StableRNGs, Distributions, SmoothingSplines, ProgressLogging, ColorSchemes, JLD2, Combinatorics

	# Add Revise.jl before the Dev packages to track
	#using Revise

	# Packages under development
	using DataDrivenDiffEq, DataDrivenSparse
	
	# Set a random seed for reproducibility
	rng = StableRNG(1111)

	gr()
end

# ╔═╡ 9ebfadf0-b711-46c0-b5a2-9729f9e042ee
md"""
# Equation discovery with E-SINDy
This notebook contains the code to implement a version of the E-SINDy algorithms inspired from "Discovering governing equations from data by sparse identification of nonlinear dynamical systems" by Brunton et al. (2016), [DOI](https://doi.org/10.1098/rspa.2021.0904). It was designed to estimate the algebraic formula of an unknown function. This function corresponds to unknown parts of an ODE system that were approximated by a neural network. Two different signalling pathways, and corresponding ODE systems, were considered that generated the data used in this notebook:
- Simple Negative Feedback Loop model (NFB) of two phosphatases g1p and g2p (Chapter 12: Parameter Estimation, Sloppiness, and Model Identifiability in "Quantitative Biology: Theory, Computational Methods and Examples of Models" by D. Daniels, M. Dobrzyński, D. Fey (2018)).
- ERK activation dynamics model (ERK) as described in "Frequency modulation of ERK activation dynamics rewires cell fate" by H. Ryu et al. (2015).
"""

# ╔═╡ 6024651c-85f6-4e53-be0f-44f658cf9c77
md"""
##### Environment Set-up 
"""

# ╔═╡ 6b5db2ee-bb76-4617-b6b1-e7b6064b0fb9
begin
	# Create a dev copy of the bugging packages (only once)
	#Pkg.develop(Pkg.PackageSpec(name="DataDrivenDiffEq", uuid="5C42544C-9741-4B26-99DF-F196E0D3E510"))
	#Pkg.develop(Pkg.PackageSpec(name="DataDrivenSparse", uuid="5b588203-7d8b-4fab-a537-c31a7f73f46a"))
end

# ╔═╡ d6b1570f-b98b-4a98-b5e5-9d747930a5ab
md"""
##### Import the data
"""

# ╔═╡ 8f821358-68cf-4ffd-88d3-37b64238f5ef
function smooth_nn_output(x, NN_y, n_samples, size_sample, smoothing)
	smoothed_Y = zeros(size(NN_y))
	for i in 0:(n_samples-1)
		x_t = x[1 + i*size_sample: (i+1)*size_sample]
		y = NN_y[1 + i*size_sample: (i+1)*size_sample]
		λ = smoothing
		spl = fit(SmoothingSpline, x_t, y, λ)
		smoothed_y = predict(spl)
		smoothed_Y[1 + i*size_sample: (i+1)*size_sample] = smoothed_y
	end
	return smoothed_Y
end

# ╔═╡ 666a20eb-8395-4859-ae70-aa8ea22c5c77
md"""
###### - NFB model
"""

# ╔═╡ fd914104-90da-469c-a80f-f068cac51a1c
# Make sample labels for plotting
function make_nfb_labels(files)
	labels = []
	case = ""
	for file in files
		words = split(file, ".")
		words = split(words[1], "_")
		label = "Input CC 0."
		for word in words
			if occursin("a", word) || occursin("b", word) || occursin("no", word)
				case = word
			elseif isdigit(word[1]) 
				label = label * word[2:end]
			end
		end
		push!(labels, label)
	end
	return (labels=labels, case=case)
end

# ╔═╡ f21876f0-3124-40ad-ad53-3a53efe77040
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
	labels, case = make_nfb_labels(files)

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

# ╔═╡ af06c850-2e8b-4b4b-9d0f-02e645a79743
begin
	# Load the NFB model estimations for various concentrations
	nofb_files = ["NFB_nofb_001.csv" "NFB_nofb_002.csv" "NFB_nofb_003.csv" "NFB_nofb_004.csv" "NFB_nofb_005.csv" "NFB_nofb_006.csv" "NFB_nofb_007.csv" "NFB_nofb_008.csv" "NFB_nofb_009.csv" "NFB_nofb_01.csv"]
	nofb_data = create_nfb_data(nofb_files)
end

# ╔═╡ 19225f7f-257d-4857-a709-6eb623985ba4
begin
	a_files = ["NFB_a_001.csv" "NFB_a_002.csv" "NFB_a_003.csv" "NFB_a_004.csv" "NFB_a_005.csv" "NFB_a_006.csv" "NFB_a_007.csv" "NFB_a_008.csv" "NFB_a_009.csv" "NFB_a_01.csv"]
	a_data = create_nfb_data(a_files)
end

# ╔═╡ 129f37f3-2399-44ec-9d01-fca76015f755
begin
	b_files = ["NFB_b_001.csv" "NFB_b_002.csv" "NFB_b_003.csv" "NFB_b_004.csv" "NFB_b_005.csv" "NFB_b_006.csv" "NFB_b_007.csv" "NFB_b_008.csv" "NFB_b_009.csv"]
	b_data = create_nfb_data(b_files)
end

# ╔═╡ 70cf3834-18d3-4732-837c-5af620c5f005
begin
	ab_files = ["NFB_ab_001.csv" "NFB_ab_002.csv" "NFB_ab_003.csv" "NFB_ab_004.csv" "NFB_ab_005.csv" "NFB_ab_006.csv" "NFB_ab_007.csv" "NFB_ab_008.csv" "NFB_ab_009.csv"] #["NFB_ab_0001.csv" "NFB_ab_0006.csv" "NFB_ab_0007.csv" "NFB_ab_0008.csv" "NFB_ab_0009.csv" "NFB_ab_001.csv" "NFB_ab_002.csv" "NFB_ab_003.csv" "NFB_ab_01.csv"]
	ab_data = create_nfb_data(ab_files)
end

# ╔═╡ b549bff5-d8e9-4f41-96d6-2d562584ccd9
md"""
###### -ERK model
"""

# ╔═╡ 447a8dec-ae5c-4ffa-b672-4f829c23eb1f
# Make sample labels for plotting
function make_erk_labels(files)
	labels = []
	for file in files
		words = split(file, ".")
		words = split(words[1], "_")
		label = ""
		for word in words
			if occursin("CC", word)
				label = label * uppercasefirst(word) * " "
			elseif occursin("m", word)
				label = label * filter(isdigit, word) * "'"
			elseif occursin("v", word)
				label = label * "/" * filter(isdigit, word) * "'"
			elseif occursin("pulse", word)
				label = label * " (" * filter(isdigit, word) * "x)"
			end
		end
		push!(labels, label)
	end
	return labels
end

# ╔═╡ 6c7929f5-15b2-4e19-8c26-e709f0da182e
# Create E-SINDy compatible data structure out of dataframes
function create_erk_data(files, gf, smoothing=0.)

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
	X = [df.Raf_fit df.PFB_fit] # # [df.R_fit df.Ras_fit df.Raf_fit df.MEK_fit df.PFB_fit]
	if lowercase(gf) == "ngf"
		GT = (0.75 .* df.PFB_fit .* 
		(1 .- df.Raf_fit) ./ (0.01 .+ (1 .- df.Raf_fit)))
	else
		GT = repeat([0], length(df.time))
	end
	Y = df.NN_approx

	# Smooth the NN approximation if necessary
	if smoothing != 0.
		smoothed_Y = smooth_nn_output(time, Y, n_samples, size_sample, smoothing)
		Y = smoothed_Y
	end

	@assert size(X, 1) == size(Y, 1)

	# Create labels for plotting
	labels = make_erk_labels(files)
	
	return (time=time, X=X, Y=Y, GT=GT, labels=labels)
end

# ╔═╡ f598564e-990b-436e-aa97-b2239b44f6d8
begin
	# Load the NGF model estimations for various pulse regimes
	ngf_files = ["ngf_lowCC_3m_3v.csv" "ngf_lowCC_3m_20v.csv" "ngf_lowCC_10m.csv" "ngf_lowCC_10m_10v.csv" "ngf_highCC_3m_3v.csv" "ngf_highCC_3m_20v.csv" "ngf_highCC_10m.csv" "ngf_highCC_10m_10v.csv"]
	ngf_data = create_erk_data(ngf_files, "NGF", 300.)
end

# ╔═╡ c8841434-a7af-4ade-a444-e6bc47575811
begin
	# Load the EGF model estimations for various pulse regimes
	egf_files = ["egf_lowCC_3m_3v.csv" "egf_lowCC_3m_20v.csv" "egf_lowCC_10m.csv" "egf_lowCC_10m_10v.csv" "egf_highCC_3m_3v.csv" "egf_highCC_3m_20v.csv" "egf_highCC_10m.csv" "egf_highCC_10m_10v.csv"]
	egf_data = create_erk_data(egf_files, "EGF", 300.)
end

# ╔═╡ 74ad0ae0-4406-4326-822a-8f3e027077b3
md"""
##### Set up the SINDy library of functions
"""

# ╔═╡ 9dc0a251-a637-4144-b32a-7ebf5b86b6f3
begin
	# Declare necessary symbolic variables for the bases
	@ModelingToolkit.variables x[1:7] i[1:1]
	#@ModelingToolkit.parameters t
	x = collect(x)
	i = collect(i)
end

# ╔═╡ ede9d54f-aaa4-4ff3-9519-14e8d32bc17f
begin
	# Define a basis of functions to estimate the unknown equation of NFB model
	nfb_h = DataDrivenDiffEq.monomial_basis(x[1:3], 3)
	nfb_basis = DataDrivenDiffEq.Basis(nfb_h, x[1:3])
end

# ╔═╡ e81310b5-63e1-45b8-ba4f-81751d0fcc06
begin
	# Define a basis of functions to estimate the unknown equation of GFs model
	erk_h = DataDrivenDiffEq.polynomial_basis(x[1:2], 2)
	erk_basis = DataDrivenDiffEq.Basis([erk_h; erk_h .* i], x[1:2], implicits=i[1:1])
end

# ╔═╡ 219b794d-9f51-441b-9197-b8ef0c4495b4
md"""
##### Set up the hyperparameter optimisation
"""

# ╔═╡ 4646bfce-f8cc-4d24-b461-753daff50f71
begin
	# Define a sampling method and the options for the data driven problem
	sampler = DataDrivenDiffEq.DataProcessing(split = 0.8, shuffle = true, batchsize = 100)
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
##### E-SINDy Bootstrapping
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
		
		n_samples = sum(data.time .== 0)
		size_sample = Int(length(data.time) / n_samples)
		rand_samples = sample(1:n_samples, floor(Int, n_samples*0.75), replace=false) #rand(1:n_samples, floor(Int, data_fraction * n_samples))[1] - 1
		indices = Vector{Int64}()
		for rand_sample in (rand_samples .- 1)
			i_start = (rand_sample * size_sample) + 1
			i_end = (rand_sample+1) * size_sample
			indices = vcat(indices, i_start:i_end)
		end

		# Define data driven problem with bootstrapped data
		rand_ind = rand(indices, floor(Int, length(indices) * data_fraction)) #rand(1:size(data.X, 1), floor(Int, size(data.X, 1) * data_fraction))
		X = data.X[rand_ind,:]
		Y = data.Y[rand_ind,:]
		#X = data.X[indices,:]
		#Y = data.Y[indices,:]

		dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(X', Y')

		# Check if the problem involves implicits
		implicits = implicit_variables(basis)
		with_implicits = false
		if !isempty(implicits)
			with_implicits = true
		end

		# Solve problem with optimal hyperparameters
		best_λ, best_ν = get_best_hyperparameters(dd_prob, basis, with_implicits)
		push!(hyperparam.λ, best_λ)
		push!(hyperparam.ν, best_ν)
		
		if with_implicits
			best_res = DataDrivenDiffEq.solve(dd_prob, basis, ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), options=options)
		else
			best_res = DataDrivenDiffEq.solve(dd_prob, basis, DataDrivenSparse.SR3(best_λ, best_ν), options = options)
		end
		
		# Store library coefficient for current bootstrap
		bootstrap_coef[i,:,:] = best_res.out[1].coefficients
	end
	return bootstrap_coef, hyperparam
end

# ╔═╡ 79b03041-01e6-4d8e-b900-446511c81058
# ╠═╡ disabled = true
#=╠═╡
# Bootstrapping function that estimate optimal library terms given data
function library_bootstrap(data, basis, n_bstraps, n_libterms)

	# Initialise the coefficient array
	n_eqs = size(data.Y, 2)
	l_basis = length(basis)
	bootstrap_coef = zeros(n_bstraps, n_eqs, l_basis)

	@info "Library E-SINDy Bootstrapping:"
	@progress name="Bootstrapping" threshold=0.01 for j in 1:n_bstraps
					
		# Define data driven problem with bootstrapped data
		rand_ind = sample(1:l_basis, n_libterms, replace=false)
		
		# Check if the problem involves implicits
		implicits = implicit_variables(basis)
		with_implicits = false
		if !isempty(implicits)
			with_implicits = true
			bt_basis = DataDrivenDiffEq.Basis(basis[rand_ind], x[1:size(data.X, 2)], implicits=i[1:1])
		else
			bt_basis = DataDrivenDiffEq.Basis(basis[rand_ind], x[1:size(data.X, 2)])
		end

		# Solve problem with optimal hyperparameters
		dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(data.X', data.Y')
		best_λ, best_ν = get_best_hyperparameters(dd_prob, bt_basis, with_implicits)
		if with_implicits
			best_res = DataDrivenDiffEq.solve(dd_prob, bt_basis, ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), options=options)
		else
			best_res = DataDrivenDiffEq.solve(dd_prob, bt_basis, DataDrivenSparse.SR3(best_λ, best_ν), options = options) 			

		end
		bootstrap_coef[j,:,rand_ind] = best_res.out[1].coefficients
		
	end
	return bootstrap_coef 
end
  ╠═╡ =#

# ╔═╡ 74b2ade4-884b-479d-9fee-828d37d7ab47
# Function to estimate coefficient statistics (mean, std)
function compute_coef_stat(bootstrap_res, coef_threshold)

	# Retrieve dimensions of the problem
	sample_size = size(bootstrap_res, 1)
	n_eqs = size(bootstrap_res, 2)
	n_terms = size(bootstrap_res, 3)

	# Compute inclusion probabilities
	inclusion_prob = (mean((bootstrap_res .!= 0), dims=1) * 100)

	# Keep only elements of basis with probabilities above threshold
	mask = inclusion_prob .> coef_threshold
	masked_res = bootstrap_res .* mask

	# Compute the mean and std of ensemble coefficients
	m = zeros(Float64, n_eqs, n_terms)
	low_interql = zeros(Float64, n_eqs, n_terms)
	up_interql = zeros(Float64, n_eqs, n_terms)

	#mu = zeros(Float64, n_eqs, n_terms)
	#sem = zeros(Float64, n_eqs, n_terms)
	
	#sem = zeros(Float64, n_eqs, n_terms)
	for i in 1:n_eqs
		for j in 1:n_terms
			current_coef = filter(!iszero, masked_res[:,i,j])
			if !isempty(current_coef)
				m[i,j] = median(current_coef) 
				low_interql[i,j] = percentile(current_coef, 5)
				up_interql[i,j] = percentile(current_coef, 95)
			end
			#mu[i,j] = mean(filter(!iszero, masked_res[:,i,j]))
			
			#sem[i,j] = std(filter(!iszero, masked_res[:,i,j]))
			#if !isnan(mu[i,j]) && sem[i,j] == 0
				#sem[i,j] = 1e-6
			#else
				#sem[i,j] = sem_i #/ sqrt(sample_size)
			#end
		end
	end
	#mu[isnan.(mu)] .= 0
	#sem[isnan.(sem)] .= 0

	return (median=m, p_5=low_interql, p_95=up_interql)#, mean=mu, SEM=sem)
end

# ╔═╡ 97ae69f0-764a-4aff-88cd-91accf6bb3fd
# Function to build callable function out of symbolic equations
function build_equations(coef, basis, verbose=true)

	# Build equation
	h = [equation.rhs for equation in DataDrivenDiffEq.equations(basis)]
	final_eqs = [sum(row .* h) for row in eachrow(coef)]

	# Solve equation wrt the implicit variable if any
	implicits = implicit_variables(basis)
	if !isempty(implicits)
		final_eqs = ModelingToolkit.solve_for(final_eqs .~ 0, implicits)
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

# ╔═╡ 74066f54-6e50-4db0-a98f-dc1eb9999899
function get_yvals(data, equations)
	n_eqs = length(equations)

	yvals = []
	for eq in equations
		push!(yvals, [eq(x) for x in eachrow([data.X ngf_data.Y])])
	end
	return yvals
end

# ╔═╡ d7e68526-5ba9-41ed-9669-e7a289db1e93
# Fucntion to compute the confidence interval of the estimated equation
function compute_CI(data, coef_low, coef_up, basis, confidence)

	# Build normal distribution for non-zero coefficients
	indices = findall(!iszero, coef_up)
	#coef_distrib = [Normal(mean_coef[k], sem_coef[k]) for k in indices]
	coef_distrib = [Uniform(coef_low[k], coef_up[k]) for k in indices]

	# Run MC simulations to estimate the distribution of estimated equations 
	n_simulations = 1000
	results = zeros(n_simulations, size(data.Y, 1), size(data.Y, 2))
	#current_coef = zeros(size(mean_coef))
	current_coef = zeros(size(coef_up))
    for i in 1:n_simulations
		
		# Samples coefficient from the distribution
		sample = [rand(distrib) for distrib in coef_distrib]
		current_coef[indices] = sample

		# Calculate function value given sample of coef.
		current_y = build_equations(current_coef, basis, false)
		n_eqs = length(current_y)
		for eq in 1:n_eqs
			results[i,:,eq] = [current_y[eq](x) for x in eachrow([data.X data.Y])]
		end
	end
	
	if confidence > 1
		confidence = confidence / 100
	end
	
    # Calculate confidence interval
    lower_percentile = (1 - confidence) / 2
    upper_percentile = 1 - lower_percentile
    ci = mapslices(row -> quantile(row, [lower_percentile, upper_percentile]),
		results, dims=1)

    return (ci_low=ci[1,:,:], ci_up=ci[2,:,:])
end

# ╔═╡ cc2395cc-00df-4e5e-8573-e85ce813fd41
# Plotting function for E-SINDy results
function plot_esindy(results; sample_ids=nothing, confidence=0)

	# Retrieve results
	data, basis, coef_median, coef_low, coef_up = results.data, results.basis, results.coef_median, results.coef_low, results.coef_up
	y = build_equations(coef_median, basis)

	# Retrieve the number of samples
	n_samples = sum(data.time .== 0)
	size_sample = Int(length(data.time) / n_samples)
	if isnothing(sample_ids)
		sample_ids = 1:n_samples
	end
	
	# Compute confidence interval if necessary
	ci_low, ci_up = nothing, nothing
	if confidence > 0
		ci_low, ci_up = compute_CI(data, coef_low, coef_up, basis, confidence)
	end
	
	# Plot results
	n_eqs = size(y, 1)
	subplots = []
	palette = colorschemes[:seaborn_colorblind] # [:vik10]
	for eq in 1:n_eqs
		if n_eqs > 1
			p = plot(title="Equation $(eq)", xlabel="Time", ylabel="Model species y(t)")
		else
			p = plot(title="", xlabel="Time", ylabel="Model species y(t)")
		end

		# Plot each sample separately
		for sample in 0:(n_samples-1)
			if (sample+1) in sample_ids
				i_start = 1 + sample * size_sample
				i_end = i_start + size_sample - 1
				i_color = ceil(Int, 1 + sample * (length(palette) / n_samples))

				y_vals =  [y[eq](x) for x in eachrow([data.X[i_start:i_end,:] data.Y[i_start:i_end]])] 
				if length(y_vals) == 1
					y_vals = repeat([y_vals], size_sample)
				end

				if confidence > 0
					plot!(p, data.time[i_start:i_end], y_vals, label=data.labels[sample+1], color=palette[i_color],
					ribbon=(y_vals-ci_low[i_start:i_end, eq], ci_up[i_start:i_end, eq]-y_vals), fillalpha=0.15)
				else
					plot!(p, data.time[i_start:i_end], y_vals, label=data.labels[sample+1], color=palette[i_color])
				end
				
				plot!(p, data.time[i_start:i_end], data.GT[i_start:i_end, eq], label="", linestyle=:dash, color=palette[i_color])
			end
		end
		plot!(p, [], [],  label="GT", color=:black, linestyle=:dash)
		push!(subplots, p)
	end
	return subplots
end

# ╔═╡ 012a5186-03aa-482d-bb62-ecba49587877
# Complete E-SINDy function 
function esindy(data, basis, n_bstrap=100; coef_threshold=15, data_fraction=1)

	# Run sindy bootstraps
	bootstrap_res, hyperparameters = sindy_bootstrap(data, basis, n_bstrap, data_fraction)

	# Compute the mean and std of ensemble coefficients
	e_coef, low_b, up_b = compute_coef_stat(bootstrap_res, coef_threshold)

	# Build the final equation as callable functions
	println("E-SINDy estimated equations:")
	y = build_equations(e_coef, basis)
	
	return (data=data, basis=basis, bootstraps=bootstrap_res, coef_median=e_coef, coef_low=low_b, coef_up=up_b, hyperparameters=hyperparameters)
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
#ngf_res = esindy(ngf_data, erk_basis, 100, coef_threshold=20, data_fraction=0.3)

# ╔═╡ 4a30e321-807c-43ca-a5f3-198ae8d42435
#ngf_res_up = merge(ngf_res, (coef_median=m, coef_low=low_b, coef_up=up_b))

# ╔═╡ 02c625dc-9d21-488e-983b-c3e2c40e0aad
md"""
###### Save or load the results (JLD2 file)
"""

# ╔═╡ 04538fbf-63b7-4394-b281-f047d0c3ea51
begin
	#jldsave("./Data/ngf_esindy_100bt.jld2"; results=ngf_res)
	#print_jld_key("./Data/ngf_esindy_1000bt_hp.jld2")
	ngf_res = load("./Data/ngf_esindy_100bt.jld2")["results"]
end

# ╔═╡ 60228ff7-59ba-458c-9cf5-1c105c0d2ba9
begin
	m, low_b, up_b = compute_coef_stat(ngf_res.bootstraps, 20)
	
end

# ╔═╡ 3bd630ad-f546-4b0b-8f3e-98367de739b1
md"""
###### Plot the results
"""

# ╔═╡ de4c57fe-1a29-4354-a5fe-f4e184de4dd3
begin
	ngf_plot = plot_esindy(ngf_res_up, confidence=95)[1]
	plot(ngf_plot, title="E-SINDy results for ERK model\nafter NGF stimulation\n", size=(800, 600), legend_position=:topright)
	#savefig("./Plots/ngf_esindy_100bt.svg")
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
#egf_res = esindy(egf_data, erk_basis, 100, coef_threshold=30, data_fraction=0.3)

# ╔═╡ 8f7e784d-1f35-49bf-ba73-d2306453e258
md"""
###### Save or load the results (JLD2 file)
"""

# ╔═╡ 65732e07-69dd-43d5-b04c-72444c78c470
begin
	#jldsave("./Data/egf_esindy_100bt.jld2"; results=egf_res)
	#print_jld_key("./Data/egf_esindy_1000bt_hp.jld2")
	egf_res = load("./Data/egf_esindy_100bt.jld2")["results"]
end

# ╔═╡ 664b3e67-09d3-4e98-a946-ffbcbed0fe90
begin
	#egf_plot = plot_esindy(egf_res, confidence=0)[1]
	#plot(egf_plot, title="E-SINDy results for ERK model\nafter EGF stimulation\n", size=(800, 600), legend_position=:topright, ylim=(-0.05, .1))
	#savefig("./Plots/egf_esindy_100bt.svg")
end

# ╔═╡ d0e65b25-0ea7-46c1-ac15-99eea43b6ade
md"""
### NFB model
"""

# ╔═╡ 77c1d355-8a7e-414e-9e0e-8eda1fbbbf1d
md"""
#### Case a
"""

# ╔═╡ 5e692d14-c785-4f8c-9e9a-f581c56e1bc8
md"""
###### Run E-SINDy
"""

# ╔═╡ 5a8428a3-eba1-4960-8033-bb537eda034f
#a_res = esindy(a_data, nfb_basis, 100, coef_threshold=90, data_fraction=0.3)

# ╔═╡ 07910dfe-9a66-4fc2-bbb3-66a18ed2b7a6
md"""
###### Save or load results
"""

# ╔═╡ 4edf87b9-ee15-47d2-91e1-d7c683d9cc1f
#jldsave("./Data/nfb_esindy_100bt_a.jld2"; results=a_res)

# ╔═╡ fa126d0c-868c-4083-b401-02504c033605
a_res = load("./Data/nfb_esindy_100bt_a.jld2")["results"]

# ╔═╡ dd11964a-1693-47d5-830f-3c4c632efe83
md"""
###### Plot results
"""

# ╔═╡ 20869d4b-ccd5-479a-b0d9-3271bef9921d
begin
	# Plot the results
	a_plots = plot_esindy(a_res, sample_ids=1:9, confidence=95)
	a_plot2 = plot(a_plots[2], ylim=(0, 2))
	plot(a_plots[1], a_plot2, layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB on g1p")
	
end

# ╔═╡ f9f4822e-e5cc-4c15-9131-30c8ec2e0b83
#savefig("./Plots/nfb_esindy_a.svg")

# ╔═╡ 6d7969d7-5753-4f4e-bace-7de3a542103e
md"""
#### Case b
"""

# ╔═╡ a692aaa9-3b35-44cf-9293-ae0e2faf1eeb
md"""
###### Run E-SINDy
"""

# ╔═╡ 973f831a-4050-4c34-a868-091f335bcbb4
b_res = esindy(b_data, nfb_basis, 100, coef_threshold=99, data_fraction=0.3)

# ╔═╡ 6d20c14d-7fed-4812-a677-d7d85fdbe725
md"""
###### Save or load results
"""

# ╔═╡ d34550ae-57fb-44b9-bd0f-e03fdf47b058
#jldsave("./Data/nfb_esindy_100bt_b.jld2"; results=b_res)

# ╔═╡ 99c86351-4a05-45cb-b842-7b7c2d37a7e1
#b_res = load("./Data/nfb_esindy_100bt_b.jld2")["results"]

# ╔═╡ 18d05848-5789-414d-ab88-439c60959899
md"""
###### Plot results
"""

# ╔═╡ 34c1f8fd-27de-4d89-a4f4-c8fbf62c54f6
begin
	# Plot the results
	b_plots = plot_esindy(b_res, sample_ids=1:9, confidence=95)
	b_plot1 = plot(b_plots[1], ylim=(0, 2))
	plot(b_plot1, b_plots[2], layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB on g2p")
end

# ╔═╡ bcd1baf8-548c-4798-806e-67a92238be07
#savefig("./Plots/nfb_esindy_b.svg")

# ╔═╡ a8d77be8-17f3-42e5-9ba2-b22df0ff2b04
md"""
#### Case ab
"""

# ╔═╡ 215e6f0b-faa1-4e80-b946-3fbc6d048058
md"""
###### Run E-SINDy
"""

# ╔═╡ 947b426f-bf18-4c43-bf18-85bc8b7df201
#ab_res = esindy(ab_data, nfb_basis, 100, coef_threshold=90, data_fraction=0.3)

# ╔═╡ a42646a0-a861-4687-9580-eb37e14dc05f
md"""
###### Save or load results
"""

# ╔═╡ b8cfb8e1-1d4d-410e-8629-5c93f4d15d78
#jldsave("./Data/nfb_esindy_100bt_ab.jld2"; results=ab_res)

# ╔═╡ 52485563-9ac1-46df-8c18-ce94507782c0
ab_res = load("./Data/nfb_esindy_100bt_ab.jld2")["results"]

# ╔═╡ 2bf54606-0ad6-4e2f-96aa-5a80d98772a4
md"""
###### Plot results
"""

# ╔═╡ e8c5545d-9988-42f9-bf2c-bbd0199f7655
begin
	# Plot the results
	ab_plots = plot_esindy(ab_res, confidence=95)
	plot(ab_plots[1], ab_plots[2], layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB on g1p and g2p")
end

# ╔═╡ 8f6071b2-b9d7-46a4-8ec0-aadb3f01692e
#savefig("./Plots/nfb_esindy_ab.svg")

# ╔═╡ 4ae9802d-fdbc-449c-a560-391913369306
md"""
#### Case no FB
"""

# ╔═╡ 9910d0af-d0af-4543-ad4f-29c043b70a63
md"""
###### Run E-SINDy
"""

# ╔═╡ 0529134b-963e-4c10-bb1b-848db02a2f99
#nofb_res = esindy(nofb_data, nfb_basis, 100, coef_threshold=95, data_fraction=0.3)

# ╔═╡ 837f137d-be72-4cdd-ad3e-6153754bad08
md"""
###### Save or load results
"""

# ╔═╡ ab383b83-7ff1-4e11-b7b8-256b6c78e703
#jldsave("./Data/nfb_esindy_100bt_nofb.jld2"; results=nofb_res)

# ╔═╡ 7fdae884-4a02-4f61-8b53-616e14bef473
nofb_res = load("./Data/nfb_esindy_100bt_nofb.jld2")["results"]

# ╔═╡ a46a7ac1-4b0c-45cf-a191-240bc4796b20
md"""
###### Plot results
"""

# ╔═╡ 249d11db-7af2-48f8-bc04-961b52816d4a
begin
	# Plot the results
	nofb_plots = plot_esindy(nofb_res, confidence=95)
	plot(nofb_plots[1], nofb_plots[2], ylim=(0, 2), layout=(2,1), size=(800, 800), plot_title="E-SINDy results with no FB")
end

# ╔═╡ f6354d11-84ae-459a-807a-7d759d7c07f7
#savefig("./Plots/nfb_esindy_nofb.svg")

# ╔═╡ c7808cd2-5b58-4eb6-88fe-0a7fbb66717d
md"""
## Accessory functions
"""

# ╔═╡ 9662aac5-d773-4508-a643-813130f53e5b
function print_jld_key(filename)
	# To see which key(s) were used to store the object in the jld2 file
	file = jldopen(filename, "r")
	jdl_keys = keys(file)
	for key in jdl_keys
		println(key)
	end
	close(file)
end

# ╔═╡ 7507f422-d879-474a-a1ef-5f376f936d7f
function safe_access(arr, i)
    try
        return arr[i]
    catch
        return nothing
    end
end

# ╔═╡ Cell order:
# ╟─9ebfadf0-b711-46c0-b5a2-9729f9e042ee
# ╟─6024651c-85f6-4e53-be0f-44f658cf9c77
# ╟─b9ff1932-20ef-48da-9d2d-d61f3ee13a4f
# ╟─6b5db2ee-bb76-4617-b6b1-e7b6064b0fb9
# ╟─a806fdd2-5017-11ef-2351-dfc89f69b334
# ╟─d6b1570f-b98b-4a98-b5e5-9d747930a5ab
# ╟─8f821358-68cf-4ffd-88d3-37b64238f5ef
# ╟─666a20eb-8395-4859-ae70-aa8ea22c5c77
# ╟─f21876f0-3124-40ad-ad53-3a53efe77040
# ╟─fd914104-90da-469c-a80f-f068cac51a1c
# ╠═af06c850-2e8b-4b4b-9d0f-02e645a79743
# ╟─19225f7f-257d-4857-a709-6eb623985ba4
# ╠═129f37f3-2399-44ec-9d01-fca76015f755
# ╠═70cf3834-18d3-4732-837c-5af620c5f005
# ╟─b549bff5-d8e9-4f41-96d6-2d562584ccd9
# ╟─6c7929f5-15b2-4e19-8c26-e709f0da182e
# ╟─447a8dec-ae5c-4ffa-b672-4f829c23eb1f
# ╟─f598564e-990b-436e-aa97-b2239b44f6d8
# ╟─c8841434-a7af-4ade-a444-e6bc47575811
# ╟─74ad0ae0-4406-4326-822a-8f3e027077b3
# ╟─9dc0a251-a637-4144-b32a-7ebf5b86b6f3
# ╟─ede9d54f-aaa4-4ff3-9519-14e8d32bc17f
# ╟─e81310b5-63e1-45b8-ba4f-81751d0fcc06
# ╟─219b794d-9f51-441b-9197-b8ef0c4495b4
# ╟─4646bfce-f8cc-4d24-b461-753daff50f71
# ╟─034f5422-7259-42b8-b3b3-e34cfe47b7b7
# ╟─0f7076ef-848b-4b3c-b918-efb6419787be
# ╟─5ef43425-ca26-430c-a62d-e194a8b1aebb
# ╟─c7e825a3-8ce6-48fc-86ac-21810d32bbfb
# ╟─79b03041-01e6-4d8e-b900-446511c81058
# ╠═74b2ade4-884b-479d-9fee-828d37d7ab47
# ╟─97ae69f0-764a-4aff-88cd-91accf6bb3fd
# ╟─74066f54-6e50-4db0-a98f-dc1eb9999899
# ╟─d7e68526-5ba9-41ed-9669-e7a289db1e93
# ╠═cc2395cc-00df-4e5e-8573-e85ce813fd41
# ╟─012a5186-03aa-482d-bb62-ecba49587877
# ╟─569c7600-246b-4a64-bba5-1e74a5888d8c
# ╟─e49b5f76-d04b-4b19-a4c8-a3a3a1b40f1b
# ╟─1f3f1461-09f1-4825-bb99-4064e075e23e
# ╠═b73f3450-acfd-4b9e-9b7f-0f7289a62976
# ╠═60228ff7-59ba-458c-9cf5-1c105c0d2ba9
# ╠═4a30e321-807c-43ca-a5f3-198ae8d42435
# ╟─02c625dc-9d21-488e-983b-c3e2c40e0aad
# ╠═04538fbf-63b7-4394-b281-f047d0c3ea51
# ╟─3bd630ad-f546-4b0b-8f3e-98367de739b1
# ╠═de4c57fe-1a29-4354-a5fe-f4e184de4dd3
# ╟─03993d5c-7900-454a-9fe3-a70bd69456d1
# ╟─e675346b-99dc-441f-913b-19387aaef3ea
# ╟─50d20cc1-ce44-43b6-9d72-1ff36137f6f8
# ╟─8f7e784d-1f35-49bf-ba73-d2306453e258
# ╟─65732e07-69dd-43d5-b04c-72444c78c470
# ╟─664b3e67-09d3-4e98-a946-ffbcbed0fe90
# ╟─d0e65b25-0ea7-46c1-ac15-99eea43b6ade
# ╟─77c1d355-8a7e-414e-9e0e-8eda1fbbbf1d
# ╟─5e692d14-c785-4f8c-9e9a-f581c56e1bc8
# ╟─5a8428a3-eba1-4960-8033-bb537eda034f
# ╟─07910dfe-9a66-4fc2-bbb3-66a18ed2b7a6
# ╟─4edf87b9-ee15-47d2-91e1-d7c683d9cc1f
# ╟─fa126d0c-868c-4083-b401-02504c033605
# ╟─dd11964a-1693-47d5-830f-3c4c632efe83
# ╟─20869d4b-ccd5-479a-b0d9-3271bef9921d
# ╠═f9f4822e-e5cc-4c15-9131-30c8ec2e0b83
# ╟─6d7969d7-5753-4f4e-bace-7de3a542103e
# ╟─a692aaa9-3b35-44cf-9293-ae0e2faf1eeb
# ╟─973f831a-4050-4c34-a868-091f335bcbb4
# ╟─6d20c14d-7fed-4812-a677-d7d85fdbe725
# ╟─d34550ae-57fb-44b9-bd0f-e03fdf47b058
# ╠═99c86351-4a05-45cb-b842-7b7c2d37a7e1
# ╟─18d05848-5789-414d-ab88-439c60959899
# ╟─34c1f8fd-27de-4d89-a4f4-c8fbf62c54f6
# ╟─bcd1baf8-548c-4798-806e-67a92238be07
# ╟─a8d77be8-17f3-42e5-9ba2-b22df0ff2b04
# ╟─215e6f0b-faa1-4e80-b946-3fbc6d048058
# ╟─947b426f-bf18-4c43-bf18-85bc8b7df201
# ╟─a42646a0-a861-4687-9580-eb37e14dc05f
# ╟─b8cfb8e1-1d4d-410e-8629-5c93f4d15d78
# ╟─52485563-9ac1-46df-8c18-ce94507782c0
# ╟─2bf54606-0ad6-4e2f-96aa-5a80d98772a4
# ╟─e8c5545d-9988-42f9-bf2c-bbd0199f7655
# ╟─8f6071b2-b9d7-46a4-8ec0-aadb3f01692e
# ╟─4ae9802d-fdbc-449c-a560-391913369306
# ╟─9910d0af-d0af-4543-ad4f-29c043b70a63
# ╟─0529134b-963e-4c10-bb1b-848db02a2f99
# ╟─837f137d-be72-4cdd-ad3e-6153754bad08
# ╟─ab383b83-7ff1-4e11-b7b8-256b6c78e703
# ╟─7fdae884-4a02-4f61-8b53-616e14bef473
# ╟─a46a7ac1-4b0c-45cf-a191-240bc4796b20
# ╟─249d11db-7af2-48f8-bc04-961b52816d4a
# ╟─f6354d11-84ae-459a-807a-7d759d7c07f7
# ╟─c7808cd2-5b58-4eb6-88fe-0a7fbb66717d
# ╟─9662aac5-d773-4508-a643-813130f53e5b
# ╟─7507f422-d879-474a-a1ef-5f376f936d7f
