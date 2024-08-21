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
	using StatsBase, Plots, CSV, DataFrames, Printf

	# External libraries
	using HyperTuning, StableRNGs, Distributions, SmoothingSplines, ProgressLogging, ColorSchemes, JLD2

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
This notebook contains the code to implement the E-SINDy algorithms as presented in "Discovering governing equations from data by sparse identification of nonlinear dynamical systems" by Brunton et al. (2016), [DOI](https://doi.org/10.1098/rspa.2021.0904). It was designed to estimate the algebraic formula of an unknown function approximated by a neural network in the context of an ODE system. Two different signalling pathways, and corresponding ODE systems, were considered that generated the data used in this notebook:
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

	df = CSV.read("./Data/$(files[1])", DataFrame)
	if length(files) > 1
	    for i in 2:length(files)
	        df2 = CSV.read("./Data/$(files[i])", DataFrame)
	        df = vcat(df, df2)
	    end
	end

	labels, case = make_nfb_labels(files)

	# Retrieve the number of samples
	n_samples = sum(df.time .== 0)
	size_sample = Int(nrow(df) / n_samples)

	time = df.time
	X = [df.g1p_fit df.g2p_fit df.Xact_fit]
		Y = [df.NN1 df.NN2]

	GT = 0
	if case == "a"
		GT = [df.Xact_fit repeat([0], size(df.Xact_fit, 1))]
	elseif case == "b"
		GT = [repeat([0], size(df.Xact_fit, 1)) df.Xact_fit]
	elseif case == "nofb"
		GT = [repeat([0], size(df.time, 1)) repeat([0], size(df.time, 1))]
	elseif case == "ab"
		GT = [df.Xact_fit df.Xact_fit]
	end

	@assert size(X, 1) == size(Y, 1)
		
	return (time=time, X=X, Y=Y, GT=GT, labels=labels, case=case)
end

# ╔═╡ af06c850-2e8b-4b4b-9d0f-02e645a79743
begin
	# Load the NFB model estimations for various concentrations
	nofb_files = ["NFB_nofb_01.csv" "NFB_nofb_005.csv" "NFB_nofb_001.csv" "NFB_nofb_0001.csv"]
	nofb_data = create_nfb_data(nofb_files)

	a_files = ["NFB_a_01.csv" "NFB_a_005.csv" "NFB_a_001.csv" "NFB_a_0001.csv"]
	a_data = create_nfb_data(a_files)

	b_files = ["NFB_b_01.csv" "NFB_b_005.csv" "NFB_b_001.csv" "NFB_b_0001.csv"]
	b_data = create_nfb_data(b_files)

	ab_files = ["NFB_ab_01.csv" "NFB_ab_005.csv" "NFB_ab_001.csv" "NFB_ab_0001.csv"]
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
function create_erk_data(files, smoothing=0.)

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
	X = [df.R_fit df.Ras_fit df.Raf_fit df.MEK_fit df.PFB_fit] # [df.Raf_fit df.PFB_fit] #
	GT = (0.75 .* df.PFB_fit .* 
		(1 .- df.Raf_fit) ./ (0.01 .+ (1 .- df.Raf_fit)))
	Y = df.NN_approx

	# Smooth the NN approximation if necessary
	if smoothing != 0
		smoothed_Y = zeros(size(Y))
		for i in 0:(n_samples-1)
			x_t = df.time[1 + i*size_sample: (i+1)*size_sample]
			y = df.NN_approx[1 + i*size_sample: (i+1)*size_sample]
			λ = smoothing
			spl = fit(SmoothingSpline, x_t, y, λ)
			smoothed_y = predict(spl)
			smoothed_Y[1 + i*size_sample: (i+1)*size_sample] = smoothed_y
		end
	end

	@assert size(X, 1) == size(Y, 1)

	# Create labels for plotting out of the filenames
	labels = make_erk_labels(files)
	
	return (time=time, X=X, Y=smoothed_Y, GT=GT, labels=labels)
end

# ╔═╡ f598564e-990b-436e-aa97-b2239b44f6d8
begin
	# Load the NGF model estimations for various pulse regimes
	ngf_files = ["ngf_highCC_10m_10v.csv" "ngf_highCC_10m_3pulse.csv" "ngf_highCC_3m_20v.csv" "ngf_highCC_10m.csv" "ngf_highCC_3m_3v.csv" "ngf_lowCC_10m_10v.csv"]
	ngf_data = create_erk_data(ngf_files, 300.)
end

# ╔═╡ 74ad0ae0-4406-4326-822a-8f3e027077b3
md"""
##### Set up the SINDy library of functions
"""

# ╔═╡ 9dc0a251-a637-4144-b32a-7ebf5b86b6f3
begin
	# Declare necessary symbolic variables for the bases
	@ModelingToolkit.variables x[1:7] i[1:1]
	@ModelingToolkit.parameters t
	x = collect(x)
	i = collect(i)
end

# ╔═╡ ede9d54f-aaa4-4ff3-9519-14e8d32bc17f
begin
	# Define a basis of functions to estimate the unknown equation of NFB model
	nfb_h = DataDrivenDiffEq.monomial_basis(x[1:3], 3)
	nfb_basis = DataDrivenDiffEq.Basis(nfb_h, x[1:3], iv=t)
end

# ╔═╡ e81310b5-63e1-45b8-ba4f-81751d0fcc06
begin
	# Define a basis of functions to estimate the unknown equation of GFs model
	gf_h = DataDrivenDiffEq.polynomial_basis(x[1:5], 2)
	gf_basis = DataDrivenDiffEq.Basis([gf_h; gf_h .* i], x[1:5], implicits=i[1:1])
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
	scenario = Scenario(λ = 1e-2:1e-3:9e-1, ν = exp10.(-2:1:3),
		max_trials = 500, 
		sampler=HyperTuning.RandomSampler(), batch_size=1)

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
function sindy_bootstrap(data, basis, n_bstraps, n_libterms)

	# Initialise the coefficient array
	n_eqs = size(data.Y, 2)
	l_basis = length(basis)
	bootstrap_coef = zeros(n_bstraps, n_eqs, l_basis)

	@info "E-SINDy Bootstrapping:"
	@progress name="Bootstrapping" threshold=0.01 for i in 1:n_bstraps

		# Define how much of the data is sampled
		data_percentage = floor(Int, size(data.X, 1) * 0.3)
		
		# Define data driven problem with bootstrapped data
		rand_ind = rand(1:size(data.X, 1), data_percentage)
		X = data.X[rand_ind,:]
		Y = data.Y[rand_ind,:]
		dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(X', Y')

		# Check if the problem involves implicits
		implicits = implicit_variables(basis)
		with_implicits = false
		if !isempty(implicits)
			with_implicits = true
		end

		# Solve problem with optimal hyperparameters
		best_λ, best_ν = get_best_hyperparameters(dd_prob, basis, with_implicits)
		if with_implicits
			best_res = DataDrivenDiffEq.solve(dd_prob, basis, ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), options=options)

		else
			best_res = DataDrivenDiffEq.solve(dd_prob, basis, DataDrivenSparse.SR3(best_λ, best_ν), options = options) 			

		end
		bootstrap_coef[i,:,:] = best_res.out[1].coefficients
		
	end
	return bootstrap_coef 
end

# ╔═╡ 79b03041-01e6-4d8e-b900-446511c81058
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
	sem = ones(Float64, n_eqs, n_terms)
	for i in 1:n_eqs
		for j in 1:n_terms
			m[i,j] = mean(filter(!iszero, masked_res[:,i,j]))
			
			sem_i = std(masked_res[:,i,j])
			if !isnan(m[i,j]) && sem_i == 0
				sem[i,j] = 1e-6
			else
				sem[i,j] = sem_i / sqrt(sample_size)
			end
		end
	end
	m[isnan.(m)] .= 0
	sem[isnan.(sem)] .= 0

	return (mean = m, SEM = sem)
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

# ╔═╡ 30ae9b9d-1f10-4cc8-b605-096badd27f4a
#=╠═╡
function compute_CI(data, mean_coef, sem_coef)
	lower_b = repeat([Inf], size(data.time, 1))
	upper_b = repeat([-Inf], size(data.time, 1))
	
	non_zero_idx = findall(!iszero, sem_coef)
	n_coef = length(non_zero_idx)
	
	operations = []
	combs = collect(with_replacement_combinations([+, -], n_coef))
	for comb in combs
	    perms = permutations(comb)
	    append!(operations, collect(perms))
	end
	operations = unique!(operations)
	
	ci_coef = zeros(size(e_coef))
	for operation in operations
		for i in 1:n_coef
			i_coef = non_zero_idx[i]
			ci_coef[i_coef] = operation[i](e_coef[i_coef], sem_coef[i_coef])
		end
		ci_y = build_equations(ci_coef, gf_basis, false)
		ci_y_vals = [ci_y[1](x) for x in eachrow(ngf_data.X[1:801,:])]
		println(size(min.(lower_b, ci_y_vals)))
		lower_b = min.(lower_b, ci_y_vals)
		upper_b = max.(upper_b, ci_y_vals)
	end
	return (lower_b=lower_b, upper_b=upper_b)
end
  ╠═╡ =#

# ╔═╡ cc2395cc-00df-4e5e-8573-e85ce813fd41
# Plotting function for E-SINDy results
function plot_esindy(data, y, ci_coef, basis, confidence)
	
	# Get the quantile corresponding to the confidence interval
	Z = Normal()
	if 0 <= confidence <= 1
		q = quantile(Z, confidence)
	else
		q = quantile(Z, confidence/100)
	end

	# Obtain the equations for the SEM (standard error of the mean)
	if sum(ci_coef) != 0
		ci_y = build_equations(ci_coef, basis, false)
	end
		
	# Retrieve the number of samples
	n_samples = sum(data.time .== 0)
	size_sample = Int(length(ngf_data.time) / n_samples)

	# Plot results with CI
	n_eqs = size(y, 1)
	subplots = []
	palette = colorschemes[:seaborn_colorblind] # [:vik10]
	for i in 1:n_eqs
		
		if n_eqs > 1
			p = plot(title="Equation $(i)", xlabel="Time", ylabel="Model species y(t)")
		else
			p = plot(title="", xlabel="Time", ylabel="Model species y(t)")
		end

		# Plot each sample separately
		for sample in 0:(n_samples-1)
			i_start = 1 + sample * size_sample
			i_end = i_start + size_sample - 1
			i_color = ceil(Int, 1 + sample * (length(palette) / n_samples))
			
			y_vals = [y[i](x) for x in eachrow([ngf_data.X[i_start:i_end,:] ngf_data.Y[i_start:i_end]])] 
			if length(y_vals) == 1
				y_vals = repeat([y_vals], size_sample)
			end
			
			ci_vals = [ci_y[1](x) for x in eachrow([ngf_data.X[i_start:i_end,:] ngf_data.Y[i_start:i_end]])] 

			if length(ci_vals) == 1
				ci_vals = repeat([ci_vals], size_sample)
			end

			plot!(p, ngf_data.time[i_start:i_end], y_vals, label=data.labels[sample+1], color=palette[i_color], ribbon=q .* (ci_vals-y_vals), fillalpha=0.15)

			plot!(p, ngf_data.time[i_start:i_end], ngf_data.GT[i_start:i_end], label="", linestyle=:dash, color=palette[i_color])
		end
		plot!(p, [], [],  label="GT", color=:black, linestyle=:dash)
		push!(subplots, p)
	end
	return subplots
end

# ╔═╡ 012a5186-03aa-482d-bb62-ecba49587877
# Complete E-SINDy function 
function e_sindy(data, basis, n_bstrap, coef_threshold=15, confidence=95)

	# Run sindy bootstraps
	bootstrap_res = sindy_bootstrap(data, basis, n_bstrap)

	# Compute the mean and std of ensemble coefficients
	e_coef, coef_sem = compute_coef_stat(bootstrap_res, coef_threshold)

	# Build the final equation as callable functions
	println("E-SINDy estimated equations:")
	y = build_equations(e_coef, basis)

	# Plot result
	plots = plot_esindy(data, y, coef_sem, basis, confidence)
	
	return (equations=y, bootstraps=bootstrap_res, coef_mean=e_coef, coef_sem=coef_sem, plots=plots)
end

# ╔═╡ 569c7600-246b-4a64-bba5-1e74a5888d8c
md"""
### GFs model
"""

# ╔═╡ b790d677-690c-487e-b40f-24a295096a38
# ╠═╡ disabled = true
#=╠═╡
lib_res = library_bootstrap(ngf_data, gf_basis, 100, 10)
  ╠═╡ =#

# ╔═╡ b73f3450-acfd-4b9e-9b7f-0f7289a62976
# ╠═╡ disabled = true
#=╠═╡
ngf_results_full = sindy_bootstrap(ngf_data, gf_basis, 50) #e_sindy(ngf_data, gf_basis, 10, 15, 95) 
  ╠═╡ =#

# ╔═╡ 02c625dc-9d21-488e-983b-c3e2c40e0aad
md"""
##### Save the results (JLD2 file)
"""

# ╔═╡ 53bc1c92-cf25-4de0-99f2-9bdaa754ea18
# ╠═╡ disabled = true
#=╠═╡
#JLD2.@save "./Data/ngf_esindy_1000bt.jld2" ngf_results
JLD2.@load "./Data/ngf_esindy_1000bt.jld2" ngf_results
  ╠═╡ =#

# ╔═╡ 68ef2a1c-69ae-42fd-ae7a-c99fdf05888a
e_coef, sem_coef = compute_coef_stat(lib_res, 20)

# ╔═╡ 546a2d07-3793-4428-8379-7e60eec2b9dd
#=╠═╡
y = build_equations(e_coef, gf_basis)
  ╠═╡ =#

# ╔═╡ 92253578-43c5-499e-95b1-1bd770592403
#=╠═╡
p = plot_esindy(ngf_data, y, e_coef .+ sem_coef, gf_basis, 95)
  ╠═╡ =#

# ╔═╡ a3e48b28-2826-4254-bd26-a10a67688b68
#=╠═╡
plot(p[1])
  ╠═╡ =#

# ╔═╡ d0e65b25-0ea7-46c1-ac15-99eea43b6ade
md"""
### NFB models
"""

# ╔═╡ 77c1d355-8a7e-414e-9e0e-8eda1fbbbf1d
md"""
##### Case a
"""

# ╔═╡ f3077d61-cb49-4efb-aac0-66e8de6e15ae
# ╠═╡ disabled = true
#=╠═╡
esindy_res = e_sindy(df_a, nfb_basis, nfb_dd_prob, 20, 65)
  ╠═╡ =#

# ╔═╡ d10c9991-ada5-4239-8b1a-5d9baaf27a1a
#=╠═╡
begin
	# Plot E-SINDy results
	subplots = deepcopy(esindy_res.plots)
	
	# Add ground truths
	plot!(subplots[1], df_a.time[1:801], [df_a.Xact_fit[1:801], df_a.Xact_fit[802:1602], df_a.Xact_fit[1603:2403], df_a.Xact_fit[2404:end]], label=["GT (sample 1)" "GT (sample 2)" "GT (sample 3)" "GT (sample 4)"], colour=[:blue :orange :green :pink], linewidth=1, linestyle=:dash)

	plot!(subplots[2], df_a.time[1:801], repeat([1], 801), label="GT", colour=:black, linewidth=1, linestyle=:dash, ylim=(0.95, 1.2))

	plot(subplots[1], subplots[2], layout=(2,1), size=(600, 800))
end
  ╠═╡ =#

# ╔═╡ c20a0b57-28a2-49b0-abf5-22e4b8f61b12
#=╠═╡
esindy_res.coef_mean
  ╠═╡ =#

# ╔═╡ 6d7969d7-5753-4f4e-bace-7de3a542103e
md"""
##### Case b
"""

# ╔═╡ 973f831a-4050-4c34-a868-091f335bcbb4
# ╠═╡ disabled = true
#=╠═╡
esindy_res_b = e_sindy(df_b, nfb_basis, nfb_dd_prob, 100, 25)
  ╠═╡ =#

# ╔═╡ df552696-7d87-487b-b1ea-be208bd9cfaa
#=╠═╡
begin
	# Plot E-SINDy results
	subplots_b = deepcopy(esindy_res_b.plots)
	
	# Add ground truths
	plot!(subplots_b[2], df_b.time[1:801], [df_b.Xact_fit[1:801], df_b.Xact_fit[802:1602], df_b.Xact_fit[1603:2403], df_b.Xact_fit[2404:end]], label=["GT (sample 1)" "GT (sample 2)" "GT (sample 3)" "GT (sample 4)"], colour=[:blue :orange :green :pink], linewidth=1, linestyle=:dash)

	plot!(subplots_b[1], df_b.time[1:801], repeat([1], 801), label="GT", colour=:black, linewidth=1, linestyle=:dash, ylim=(0.95, 1.2))

	plot(subplots_b[1], subplots_b[2], layout=(2,1), size=(600, 800))
end
  ╠═╡ =#

# ╔═╡ c0ab5631-01c6-44f6-b7be-77639b7986b5
#=╠═╡
esindy_res_b.coef_mean
  ╠═╡ =#

# ╔═╡ a8d77be8-17f3-42e5-9ba2-b22df0ff2b04
md"""
##### Case ab
"""

# ╔═╡ 7a512836-d5dc-4716-a7a6-e0d18e0d21af
# ╠═╡ disabled = true
#=╠═╡
esindy_res_ab = e_sindy(df_ab, nfb_basis, nfb_dd_prob, 10, 65)
  ╠═╡ =#

# ╔═╡ 6ee3cdfc-9c3e-4275-bda9-de57c85df627
#=╠═╡
begin
	# Plot E-SINDy results
	subplots_ab = deepcopy(esindy_res_ab.plots)
	
	# Add ground truths
	plot!(subplots_ab[1], df_ab.time[1:801], [df_ab.Xact_fit[1:801], df_ab.Xact_fit[802:1602], df_ab.Xact_fit[1603:2403], df_ab.Xact_fit[2404:end]], label=["GT (sample 1)" "GT (sample 2)" "GT (sample 3)" "GT (sample 4)"], colour=[:blue :orange :green :pink], linewidth=1, linestyle=:dash)

	plot!(subplots_ab[2], df_ab.time[1:801], [df_ab.Xact_fit[1:801], df_ab.Xact_fit[802:1602], df_ab.Xact_fit[1603:2403], df_ab.Xact_fit[2404:end]], label=["GT (sample 1)" "GT (sample 2)" "GT (sample 3)" "GT (sample 4)"], colour=[:blue :orange :green :pink], linewidth=1, linestyle=:dash)#, ylim=(0.95, 1.2))

	plot(subplots_ab[1], subplots_ab[2], layout=(2,1), size=(600, 800))
end
  ╠═╡ =#

# ╔═╡ bce846f1-341c-4f53-88b0-ab7b827ba24c
#=╠═╡
esindy_res_ab.coef_mean
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─9ebfadf0-b711-46c0-b5a2-9729f9e042ee
# ╟─6024651c-85f6-4e53-be0f-44f658cf9c77
# ╟─b9ff1932-20ef-48da-9d2d-d61f3ee13a4f
# ╟─6b5db2ee-bb76-4617-b6b1-e7b6064b0fb9
# ╟─a806fdd2-5017-11ef-2351-dfc89f69b334
# ╟─d6b1570f-b98b-4a98-b5e5-9d747930a5ab
# ╟─666a20eb-8395-4859-ae70-aa8ea22c5c77
# ╟─f21876f0-3124-40ad-ad53-3a53efe77040
# ╟─fd914104-90da-469c-a80f-f068cac51a1c
# ╟─af06c850-2e8b-4b4b-9d0f-02e645a79743
# ╟─b549bff5-d8e9-4f41-96d6-2d562584ccd9
# ╟─6c7929f5-15b2-4e19-8c26-e709f0da182e
# ╟─447a8dec-ae5c-4ffa-b672-4f829c23eb1f
# ╟─f598564e-990b-436e-aa97-b2239b44f6d8
# ╟─74ad0ae0-4406-4326-822a-8f3e027077b3
# ╟─9dc0a251-a637-4144-b32a-7ebf5b86b6f3
# ╟─ede9d54f-aaa4-4ff3-9519-14e8d32bc17f
# ╟─e81310b5-63e1-45b8-ba4f-81751d0fcc06
# ╟─219b794d-9f51-441b-9197-b8ef0c4495b4
# ╟─4646bfce-f8cc-4d24-b461-753daff50f71
# ╟─034f5422-7259-42b8-b3b3-e34cfe47b7b7
# ╟─0f7076ef-848b-4b3c-b918-efb6419787be
# ╟─5ef43425-ca26-430c-a62d-e194a8b1aebb
# ╠═c7e825a3-8ce6-48fc-86ac-21810d32bbfb
# ╠═79b03041-01e6-4d8e-b900-446511c81058
# ╟─74b2ade4-884b-479d-9fee-828d37d7ab47
# ╟─97ae69f0-764a-4aff-88cd-91accf6bb3fd
# ╟─30ae9b9d-1f10-4cc8-b605-096badd27f4a
# ╟─cc2395cc-00df-4e5e-8573-e85ce813fd41
# ╟─012a5186-03aa-482d-bb62-ecba49587877
# ╟─569c7600-246b-4a64-bba5-1e74a5888d8c
# ╠═b790d677-690c-487e-b40f-24a295096a38
# ╠═b73f3450-acfd-4b9e-9b7f-0f7289a62976
# ╟─02c625dc-9d21-488e-983b-c3e2c40e0aad
# ╠═53bc1c92-cf25-4de0-99f2-9bdaa754ea18
# ╠═68ef2a1c-69ae-42fd-ae7a-c99fdf05888a
# ╠═546a2d07-3793-4428-8379-7e60eec2b9dd
# ╠═92253578-43c5-499e-95b1-1bd770592403
# ╠═a3e48b28-2826-4254-bd26-a10a67688b68
# ╟─d0e65b25-0ea7-46c1-ac15-99eea43b6ade
# ╟─77c1d355-8a7e-414e-9e0e-8eda1fbbbf1d
# ╠═f3077d61-cb49-4efb-aac0-66e8de6e15ae
# ╟─d10c9991-ada5-4239-8b1a-5d9baaf27a1a
# ╟─c20a0b57-28a2-49b0-abf5-22e4b8f61b12
# ╟─6d7969d7-5753-4f4e-bace-7de3a542103e
# ╟─973f831a-4050-4c34-a868-091f335bcbb4
# ╟─df552696-7d87-487b-b1ea-be208bd9cfaa
# ╟─c0ab5631-01c6-44f6-b7be-77639b7986b5
# ╟─a8d77be8-17f3-42e5-9ba2-b22df0ff2b04
# ╟─7a512836-d5dc-4716-a7a6-e0d18e0d21af
# ╟─6ee3cdfc-9c3e-4275-bda9-de57c85df627
# ╟─bce846f1-341c-4f53-88b0-ab7b827ba24c