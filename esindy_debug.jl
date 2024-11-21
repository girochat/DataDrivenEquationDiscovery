### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ b9ff1932-20ef-48da-9d2d-d61f3ee13a4f
begin 
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ a806fdd2-5017-11ef-2351-dfc89f69b334
begin
	# SciML tools
	import ModelingToolkit, Symbolics
	
	# Standard libraries
	using StatsBase, Plots, CSV, DataFrames#, Printf, Statistics

	# External libraries
	using HyperTuning, StableRNGs, Distributions, SmoothingSplines, ColorSchemes, JLD2, ProgressLogging #, Combinatorics #

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
This notebook contains the code to implement a version of the E-SINDy algorithms inspired from "Discovering governing equations from data by sparse identification of nonlinear dynamical systems" by Brunton et al. (2016), [DOI](https://doi.org/10.1098/rspa.2021.0904). It was designed to estimate the algebraic formula of an unknown function. In this work, the function corresponds to unknown parts of an ODE system that were approximated by a neural network. Two different signalling pathways, and corresponding ODE systems, were considered that generated the data used in this notebook:
- Simple Negative Feedback Loop model (NFB) of two phosphatases g1p and g2p (Chapter 12: Parameter Estimation, Sloppiness, and Model Identifiability in "Quantitative Biology: Theory, Computational Methods and Examples of Models" by D. Daniels, M. Dobrzyński, D. Fey (2018)).
- ERK activation dynamics model (ERK) as described in "Frequency modulation of ERK activation dynamics rewires cell fate" by H. Ryu et al. (2015) [DOI](https://doi.org/10.15252/msb.20156458) .
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
# Smooth the neural network output
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
	nofb_files = ["nfb_nofb_001.csv" "nfb_nofb_002.csv" "nfb_nofb_003.csv" "nfb_nofb_004.csv" "nfb_nofb_005.csv" "nfb_nofb_006.csv" "nfb_nofb_007.csv" "nfb_nofb_008.csv" "nfb_nofb_009.csv"]
	nofb_data = create_nfb_data(nofb_files)
end

# ╔═╡ 19225f7f-257d-4857-a709-6eb623985ba4
begin
	a_files = ["nfb_a_001.csv" "nfb_a_002.csv" "nfb_a_003.csv" "nfb_a_004.csv" "nfb_a_005.csv" "nfb_a_006.csv" "nfb_a_007.csv" "nfb_a_008.csv" "nfb_a_009.csv"]
	a_data = create_nfb_data(a_files)
end

# ╔═╡ 129f37f3-2399-44ec-9d01-fca76015f755
begin
	b_files = ["nfb_b_001.csv" "nfb_b_002.csv" "nfb_b_003.csv" "nfb_b_004.csv" "nfb_b_005.csv" "nfb_b_006.csv" "nfb_b_007.csv" "nfb_b_008.csv" "nfb_b_009.csv"]
	b_data = create_nfb_data(b_files)
end

# ╔═╡ 70cf3834-18d3-4732-837c-5af620c5f005
begin
	ab_files = ["nfb_ab_001.csv" "nfb_ab_002.csv" "nfb_ab_003.csv" "nfb_ab_004.csv" "nfb_ab_005.csv" "nfb_ab_006.csv" "nfb_ab_007.csv" "nfb_ab_008.csv" "nfb_ab_009.csv"]
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

# ╔═╡ dd110815-cb48-4abd-8604-a275e3ae07b8
# Create E-SINDy compatible data structure out of dataframes
function create_erk_data_v2(files, gf; smoothing=0., var_idxs=[2,4])
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
	Y = [df.NN1 df.NN2]
	GT = [df.NN1_GT df.NN2_GT]

	# Smooth the NN approximation if necessary
	if smoothing != 0.
		smoothed_Y1 = smooth_nn_output(time, df.NN1, n_samples, size_sample, smoothing)
		smoothed_Y2 = smooth_nn_output(time, df.NN2, n_samples, size_sample, smoothing)
		Y = [smoothed_Y1 smoothed_Y2]
	end

	@assert size(X, 1) == size(Y, 1)

	# Create labels for plotting
	labels = make_erk_labels(files)
	
	return (time=time, X=X[:,var_idxs], Y=Y, GT=GT, labels=labels)
end

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
		smoothed_Y = smooth_nn_output(time, Y, n_samples, size_sample, smoothing)
		Y = smoothed_Y
	end

	@assert size(X, 1) == size(Y, 1)

	# Create labels for plotting
	labels = make_erk_labels(files)
	
	return (time=time, X=X[:,var_idxs], Y=Y, GT=GT, labels=labels)
end

# ╔═╡ d9b1a318-2c07-4d44-88ec-501b2cfc940b
begin
	# Load the NGF model estimations for various pulse regimes
	ngf_files = ["ngf_lowcc_3m_3v.csv" "ngf_lowcc_3m_20v.csv" "ngf_lowcc_10m.csv" "ngf_lowcc_10m_10v.csv" "ngf_highcc_3m_3v.csv" "ngf_highcc_3m_20v.csv" "ngf_highcc_10m.csv" "ngf_highcc_10m_10v.csv" "ngf_highcc_100m_1v.csv"]
	ngf_data = create_erk_data(ngf_files, "NGF", smoothing=300.)
	ngf_data_full = create_erk_data(ngf_files, "NGF", smoothing=300., var_idxs=1:5)
end

# ╔═╡ d8eb7ce2-7efb-47e0-9404-e58eea193225
begin
	# Load the NGF model estimations for various pulse regimes
	ngf_files_v2 = ["ngf_highcc_10m_10v_full_data_rbf_3d_25w.csv" "ngf_highcc_10m_3v_full_data_rbf_3d_25w.csv" "ngf_highcc_3m_20v_full_data_rbf_3d_25w.csv" "ngf_highcc_10m_full_data_rbf_3d_25w.csv" "ngf_lowcc_10m_10v_full_data_rbf_3d_25w.csv" "ngf_lowcc_10m_3v_full_data_rbf_3d_25w.csv" "ngf_highcc_3m_20v_full_data_rbf_3d_25w.csv" "ngf_highcc_10m_full_data_rbf_3d_25w.csv"]
	ngf_data_v2 = create_erk_data_v2(ngf_files_v2, "NGF", smoothing=300., var_idxs=[2,4,5])
end

# ╔═╡ c8841434-a7af-4ade-a444-e6bc47575811
begin
	# Load the EGF model estimations for various pulse regimes
	egf_files = ["egf_lowcc_3m_3v.csv" "egf_lowcc_3m_20v.csv" "egf_lowcc_10m.csv" "egf_lowcc_10m_10v.csv" "egf_highcc_3m_3v.csv" "egf_highcc_3m_20v.csv" "egf_highcc_10m.csv" "egf_highcc_10m_10v.csv"]
	egf_data = create_erk_data(egf_files, "EGF", smoothing=0.)
	egf_data_full = create_erk_data(egf_files, "EGF", smoothing=300., var_idxs=1:5)
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

# ╔═╡ dafd3b7d-f1ce-4c1f-8ba4-6ddf09e4f7d4
begin
	# Define a basis of functions to estimate the unknown equation of GFs model
	erk_h = DataDrivenDiffEq.polynomial_basis([x[1], x[2]], 2)
	erk_basis = Basis([erk_h; erk_h .* i], x[1:2], implicits=i)
end

# ╔═╡ d7cc32e8-9bb0-4d40-98b0-5af57b16e8fd
begin
	# Define a basis of functions to estimate the unknown equation of GFs model
	erk_h_raf_inact = DataDrivenDiffEq.polynomial_basis([1-x[1], x[1], x[2]], 2)
	erk_basis_raf_inact = DataDrivenDiffEq.Basis([erk_h_raf_inact; erk_h_raf_inact .* i], x[1:2], implicits=i[1:1])
end

# ╔═╡ 0f5e0785-2778-4fde-b010-7fcf813ceed2
begin
	erk_h_full = DataDrivenDiffEq.polynomial_basis(x[1:5], 2)
	erk_basis_full = DataDrivenDiffEq.Basis([erk_h_full; erk_h_full .* i], x[1:5], implicits=i[1:1])
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

		# Randomly select 75% of the samples
		n_samples = sum(data.time .== 0)
		size_sample = Int(length(data.time) / n_samples)
		rand_samples = sample(1:n_samples, floor(Int, n_samples * 1), replace=false)
		
		# Sample data in the selected samples
		indices = Vector{Int64}()
		for rand_sample in (rand_samples .- 1)
			i_start = (rand_sample * size_sample) + 1
			i_end = (rand_sample+1) * size_sample
			indices = vcat(indices, i_start:i_end)
		end

		# Define data driven problem with sampled data
		rand_ind = rand(indices, floor(Int, length(indices) * data_fraction))
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
				dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(X', Y[:,eq]')

				# Solve problem with optimal hyperparameters
				best_λ, best_ν = get_best_hyperparameters(dd_prob, basis, with_implicits)
				push!(hyperparam.λ, best_λ)
				push!(hyperparam.ν, best_ν)
				
				best_res = DataDrivenDiffEq.solve(dd_prob, basis, ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), options=options)
				
				# Store library coefficient for current bootstrap
				bootstrap_coef[i,eq,:] = best_res.out[1].coefficients
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
function library_bootstrap(data, basis, n_bstraps, n_libterms; implicit_id=none)

	# Initialise the coefficient array
	n_eqs = size(data.Y, 2)
	l_basis = length(basis)
	bootstrap_coef = zeros(n_bstraps, n_eqs, l_basis)
	indices = [1]
	best_bic = 1000

	@info "Library E-SINDy Bootstrapping:"
	@progress name="Bootstrapping" threshold=0.01 for j in 1:n_bstraps
		for eq in 1:n_eqs		
		
			# Check if the problem involves implicits
			implicits = implicit_variables(basis)
			with_implicits = false

			# Create bootstrap library basis
			if !isempty(implicits)
				with_implicits = true
				idxs = [1:(implicit_id-1); (implicit_id+1):l_basis]
				rand_ind = [sample(idxs, n_libterms, replace=false); implicit_id]
				bt_basis = DataDrivenDiffEq.Basis(basis[rand_ind], x[1:size(data.X, 2)], implicits=i[1:1])
			else
				rand_ind = sample(1:l_basis, n_libterms, replace=false)
				bt_basis = DataDrivenDiffEq.Basis(basis[rand_ind], x[1:size(data.X, 2)])
			end
	
			# Solve data-driven problem with optimal hyperparameters
			dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(data.X', data.Y[:,eq]')
			best_λ, best_ν = 0.6, 1000 #get_best_hyperparameters(dd_prob, bt_basis, with_implicits)
			if with_implicits
				best_res = DataDrivenDiffEq.solve(dd_prob, bt_basis, ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), options=options)
			else
				best_res = DataDrivenDiffEq.solve(dd_prob, bt_basis, DataDrivenSparse.SR3(best_λ, best_ν), options = options)
			end

			# Check if bootstrap basis is optimal
			bt_bic = bic(best_res)
			if bt_bic < best_bic
				best_bic = bt_bic
				bootstrap_coef[indices,eq,:] = zeros(length(indices),1,l_basis)
				bootstrap_coef[j,eq,rand_ind] = best_res.out[1].coefficients
				empty!(indices)
				push!(indices, j)
			elseif bt_bic == best_bic
				bootstrap_coef[j,eq,rand_ind] = best_res.out[1].coefficients
			end
		end
	end
	return bootstrap_coef 
end

# ╔═╡ 85c913fa-23be-4d01-9a78-47f786d9a4cd
function get_coef_mask(bootstraps)
	n_eqs = size(bootstraps, 2)
	println(n_eqs)
	eq_masks = []
	for eq in 1:n_eqs
		masks = []
		mask_set = Set()
		n_bootstraps = size(bootstraps, 1)
	
		for k in 1:n_bootstraps
			mask = (!iszero).(bootstraps[k,eq,:])
			if mask in mask_set
				nothing
			else
				push!(mask_set, mask)
				freq = 0
				coefs = []
				for j in k:n_bootstraps
					temp_mask = (!iszero).(bootstraps[j,eq,:])
					if isequal(mask, temp_mask)
						push!(coefs, bootstraps[j,eq,:])
						freq = freq + 1
					end
				end
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

# ╔═╡ acf3c904-9784-4b43-983e-8dd35763c100
# Function to estimate coefficient statistics
function get_final_coef(bootstrap_res, masks, coef_threshold)

	frequent_masks = [mask.mask for mask in masks if mask.freq > coef_threshold]

	final_mask = (!iszero).(sum(frequent_masks))

	masked_res = bootstrap_res .* reshape(final_mask, 1, 1, 12)

	m = median(masked_res, dims=1)[1,:,:]

	return (med=m,)
end

# ╔═╡ 74b2ade4-884b-479d-9fee-828d37d7ab47
# Function to estimate coefficient statistics
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
	for i in 1:n_eqs
		for j in 1:n_terms
			current_coef = filter(!iszero, masked_res[:,i,j])
			if !isempty(current_coef)
				m[i,j] = median(current_coef) 
			end
		end
	end
	return (med=m,)
end

# ╔═╡ 64246dd6-8aa6-4ebd-8385-624f5bffd7c7
# Function to build the reduced basis based on Library E-SINDy results
function build_basis(lib_bootstraps, basis)
	lib_stats = compute_coef_stat(lib_bootstraps, 0)
	lib_ind = findall(!iszero, lib_stats.median[1,:])
	lib_basis = DataDrivenDiffEq.Basis(basis[lib_ind], x, implicits=i[1:1])
	return lib_basis
end

# ╔═╡ 97ae69f0-764a-4aff-88cd-91accf6bb3fd
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
				nothing
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

# ╔═╡ 74066f54-6e50-4db0-a98f-dc1eb9999899
# Function to compute the equations output (y) given the input (x) 
function get_yvals(data, equations)
	n_eqs = length(equations)

	yvals = []
	for eq in equations
		push!(yvals, [eq(x) for x in eachrow([data.X data.Y])])
	end
	return yvals
end

# ╔═╡ aefb3b75-af6a-4bc9-98f8-713144b24c5a
# Function to compute the interquartile range of the estimated equation
function compute_CI(data, basis, masks)

	freqs = Weights([mask.freq for mask in masks if mask.freq >= 0.01])

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
		current_eqs = build_equations(current_coef, basis, verbose=false)
		yvals = get_yvals(data, current_eqs)
		n_eqs = length(current_eqs)
		for eq in 1:n_eqs
			results[i,:,eq] = yvals[eq]
		end
	end

	iqr_low = mapslices(row -> percentile(filter(!isnan, row), 25), results, dims=1)
	iqr_up = mapslices(row -> percentile(filter(!isnan, row), 75), results, dims=1)

    return (iqr_low=iqr_low[1,:,:], iqr_up=iqr_up[1,:,:])
end

# ╔═╡ d7e68526-5ba9-41ed-9669-e7a289db1e93
# Function to compute the interquartile range of the estimated equation
function compute_CI_backup(data, coef_low, coef_up, basis)

	# Build uniform distribution for non-zero coefficients
	indices = findall(!iszero, coef_up .- coef_low)
	coef_distrib = [Uniform(coef_low[k], coef_up[k]) for k in indices]

	# Run MC simulations to estimate the distribution of estimated equations 
	n_simulations = 1000
	results = zeros(n_simulations, size(data.Y, 1), size(data.Y, 2))
	current_coef = zeros(size(coef_up))
    for i in 1:n_simulations
		
		# Samples coefficient from the distribution
		sample = [rand(distrib) for distrib in coef_distrib]
		current_coef[indices] = sample

		# Calculate function value given sample of coef.
		current_eqs = build_equations(current_coef, basis, verbose=false)
		yvals = get_yvals(data, current_eqs)
		n_eqs = length(current_eqs)
		for eq in 1:n_eqs
			results[i,:,eq] = yvals[eq]
		end
	end

	iqr_low = mapslices(row -> minimum(row), results, dims=1)
	iqr_up = mapslices(row -> maximum(row), results, dims=1)

    return (iqr_low=iqr_low[1,:,:], iqr_up=iqr_up[1,:,:])
end

# ╔═╡ cc2395cc-00df-4e5e-8573-e85ce813fd41
# Plotting function for E-SINDy results
function plot_esindy(results; sample_idxs=nothing, iqr=true)

	# Retrieve results
	data, basis = results.data, results.basis
	coef_median = results.coef_median #, results.coef_low, results.coef_up #, coef_low, coef_up
	
	eqs = build_equations(coef_median, basis, verbose=false)
	y_vals = get_yvals(data, eqs)

	# Retrieve the number of samples
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
function esindy(data, basis, n_bstrap=100; coef_threshold=15, data_fraction=1)

	# Run sindy bootstraps
	bootstrap_res, hyperparameters = sindy_bootstrap(data, basis, n_bstrap, data_fraction)

	# Compute the masks
	masks = get_coef_mask(bootstrap_res)
	
	# Compute the mean and std of ensemble coefficients
	e_coef = compute_coef_stat(bootstrap_res, coef_threshold)[1] #, low_coef, up_coef

	# Build the final equation as callable functions
	println("E-SINDy estimated equations:")
	y = build_equations(e_coef, basis)
	
	return (data=data, basis=basis, bootstraps=bootstrap_res, coef_median=e_coef,  hyperparameters=hyperparameters, masks=masks) #coef_low=low_coef, coef_up=up_coef,
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
begin
	#ngf_res = esindy(ngf_data, erk_basis, 100, coef_threshold=20, data_fraction=1)
	#ngf_res_full = esindy(ngf_data_full, erk_basis_full, 100, coef_threshold=20, data_fraction=1)
	#ngf_res_v2 = esindy(ngf_data_v2, erk_basis_full, 100, coef_threshold=10, data_fraction=0.3)
	#ngf_res_raf_inact = esindy(ngf_data, erk_basis_raf_inact, 100, coef_threshold=20, data_fraction=1)
end

# ╔═╡ 3b541b2d-6644-4376-beb8-6e8587be286c
md"""
###### Run Library E-SINDy
"""

# ╔═╡ df7f0f7e-1bfa-4d7e-a817-8dc20a3ec0c4
begin
	# Run first library E-SINDy to reduce the library size
	#ngf_lib_res = esindy(ngf_data_full, erk_basis_full, 5000, 10, implicit_id=22)
	#ngf_lib_basis = build_basis(ngf_lib_res, erk_basis_full)
end

# ╔═╡ 92f3a5ea-9838-400a-9120-52534d9cad52
# Run E-SINDy with resulting library
#egf_res_full_lib = esindy(egf_data_full, egf_lib_basis, 100, coef_threshold=20)

# ╔═╡ 02c625dc-9d21-488e-983b-c3e2c40e0aad
md"""
###### Save or load the results (JLD2 file)
"""

# ╔═╡ 04538fbf-63b7-4394-b281-f047d0c3ea51
#jldsave("./Data/ngf_esindy_100bt.jld2"; results=ngf_res)
#jldsave("./Data/ngf_esindy_100bt_full.jld2"; results=ngf_res_full)
#jldsave("./Data/ngf_esindy_100bt_raf_inact.jld2"; results=ngf_res_raf_inact)

# ╔═╡ 62d79e8d-6d82-4fea-a596-8c32dd16d78e
begin
	ngf_res = load("./Data/ngf_esindy_100bt2.jld2")["results"]
	ngf_res_v2 = load("./Data/ngf_esindy_100bt_v2.jld2")["results"]
	ngf_res_full = load("./Data/ngf_esindy_100bt_full2.jld2")["results"]
	ngf_res_full_lib = load("./Data/ngf_esindy_100bt_full_lib.jld2")["results"]
end

# ╔═╡ 1f055fd8-c26e-46e0-bffe-51cadbfb6410
findall(!iszero, ngf_res_full.masks[1][4].mask)

# ╔═╡ 4feff5b4-66dd-4e69-ab1e-95d4363c8cb0
ngf_res.masks

# ╔═╡ 3bd630ad-f546-4b0b-8f3e-98367de739b1
md"""
###### Plot the results
"""

# ╔═╡ 5c400147-353a-43f2-9de6-c234e13f06c9
begin
	ngf_plot = plot_esindy(ngf_res, sample_idxs=1:8, iqr=true)[1]
	plot(ngf_plot, title="E-SINDy results for ERK model\nafter NGF stimulation\n", size=(800, 600), legend_position=:topright)
	#savefig("./Plots/ngf_esindy_100bt.svg")
end

# ╔═╡ a7741d65-e0b5-4547-be64-59cb736132b5
begin
	ngf_plot_full = plot_esindy(ngf_res_full, sample_idxs=1:8, iqr=false)[1]
	plot(ngf_plot_full, title="E-SINDy results for ERK model\nafter NGF stimulation\n", size=(800, 600), legend_position=:topright)
	#savefig("./Plots/ngf_esindy_100bt_full.svg")
end

# ╔═╡ de4c57fe-1a29-4354-a5fe-f4e184de4dd3
begin
	ngf_plot_full_lib = plot_esindy(ngf_res_full_lib.esindy, sample_idxs=1:8, iqr=false)[1]
	plot(ngf_plot_full_lib, title="E-SINDy results for ERK model\nafter NGF stimulation\n", size=(800, 600), legend_position=:topright)
	#savefig("./Plots/ngf_esindy_100bt_full_lib.svg")
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

# ╔═╡ 1cc0a861-8c92-4a05-85b6-6b1dfe406f2e
md"""
###### Run Library E-SINDy
"""

# ╔═╡ 2e472b7f-0863-470e-a91a-13aafd12ce0b
begin 
	# Run first library E-SINDy to reduce the library size
	#egf_lib_res = library_bootstrap(egf_data_full, erk_basis_full, 5000, 10, implicit_id=22)
	#egf_lib_basis = build_basis(egf_lib_res, erk_basis_full)
end

# ╔═╡ 900aabf8-cce0-4aa6-a257-8739b977e9b3
# Run E-SINDy with resulting library
#egf_res_full_lib = esindy(egf_data_full, egf_lib_basis, 100, coef_threshold=20)

# ╔═╡ 8f7e784d-1f35-49bf-ba73-d2306453e258
md"""
###### Save or load the results (JLD2 file)
"""

# ╔═╡ 65732e07-69dd-43d5-b04c-72444c78c470
begin
	#jldsave("./Data/egf_esindy_100bt.jld2"; results=egf_res)
	egf_res = load("./Data/egf_esindy_100bt.jld2")["results"]
end

# ╔═╡ 34be558a-80bb-41b5-80ae-4d982e612c06
begin
	#jldsave("./Data/EGF_esindy_100bt_full_lib.jld2"; results=(esindy=egf_res_full_lib, lib_esindy=egf_lib_res))
	egf_res_full_lib = load("./Data/EGF_esindy_100bt_full_lib.jld2")["results"]
end

# ╔═╡ 4a1ab1e5-90cb-4b87-abdc-7278cbca407f
begin
	#jldsave("./Data/EGF_esindy_100bt_full.jld2"; results=egf_res_full)
	egf_res_full = load("./Data/EGF_esindy_100bt_full.jld2")["results"]
end

# ╔═╡ 664b3e67-09d3-4e98-a946-ffbcbed0fe90
begin
	egf_plot = plot_esindy(egf_res_full_lib.esindy, iqr=false)[1]
	plot(egf_plot, title="E-SINDy results for ERK model\nafter EGF stimulation\n", size=(800, 600), legend_position=:topright, ylim=(-0.05, .1))
end

# ╔═╡ c182982b-6f5a-41fe-a708-9e7786e090a5
#savefig("./Plots/egf_esindy_100bt.svg")

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

# ╔═╡ 699c7160-7ca7-46db-917f-400f729277ba
# ╠═╡ disabled = true
#=╠═╡
masks = get_coef_mask(a_res.bootstraps[:,1:1,:])
  ╠═╡ =#

# ╔═╡ fa126d0c-868c-4083-b401-02504c033605
a_res = load("./Data/nfb_esindy_100bt_a.jld2")["results"]

# ╔═╡ d0fd11e2-0f4c-42f4-b90f-fda24269b6e2
masks = get_coef_mask(a_res.bootstraps)

# ╔═╡ 0d136529-28a5-406c-958d-28df99498f66
get_final_coef(a_res.bootstraps)

# ╔═╡ dd11964a-1693-47d5-830f-3c4c632efe83
md"""
###### Plot results
"""

# ╔═╡ 20869d4b-ccd5-479a-b0d9-3271bef9921d
# ╠═╡ disabled = true
#=╠═╡
begin
	# Plot the results
	a_plots = plot_esindy(a_res, sample_idxs=1:9, iqr=false)
	a_plot2 = plot(a_plots[2], ylim=(0, 2))
	plot(a_plots[1], a_plot2, layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB on g1p")
end
  ╠═╡ =#

# ╔═╡ f9f4822e-e5cc-4c15-9131-30c8ec2e0b83
#savefig("./Plots/NFB_esindy_a.svg")

# ╔═╡ 6d7969d7-5753-4f4e-bace-7de3a542103e
md"""
#### Case b
"""

# ╔═╡ a692aaa9-3b35-44cf-9293-ae0e2faf1eeb
md"""
###### Run E-SINDy
"""

# ╔═╡ 973f831a-4050-4c34-a868-091f335bcbb4
#b_res = esindy(b_data, nfb_basis, 1000, coef_threshold=98, data_fraction=1)

# ╔═╡ 6d20c14d-7fed-4812-a677-d7d85fdbe725
md"""
###### Save or load results
"""

# ╔═╡ d34550ae-57fb-44b9-bd0f-e03fdf47b058
#jldsave("./Data/nfb_esindy_100bt_b.jld2"; results=b_res)

# ╔═╡ 99c86351-4a05-45cb-b842-7b7c2d37a7e1
b_res = load("./Data/nfb_esindy_100bt_b.jld2")["results"]

# ╔═╡ 18d05848-5789-414d-ab88-439c60959899
md"""
###### Plot results
"""

# ╔═╡ 34c1f8fd-27de-4d89-a4f4-c8fbf62c54f6
# ╠═╡ disabled = true
#=╠═╡
begin
	# Plot the results
	b_plots = plot_esindy(b_res, sample_ids=1:9)
	b_plot1 = plot(b_plots[1], ylim=(0, 2))
	plot(b_plot1, b_plots[2], layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB on g2p")
end
  ╠═╡ =#

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
# ╠═╡ disabled = true
#=╠═╡
ab_res = esindy(ab_data, nfb_basis, 100, coef_threshold=98, data_fraction=0.5)
  ╠═╡ =#

# ╔═╡ dbbdd00c-c320-48a4-801c-c3be9701832b
#=╠═╡
compute_coef_stat(ab_res.bootstraps, 95)
  ╠═╡ =#

# ╔═╡ a42646a0-a861-4687-9580-eb37e14dc05f
md"""
###### Save or load results
"""

# ╔═╡ b8cfb8e1-1d4d-410e-8629-5c93f4d15d78
#jldsave("./Data/nfb_esindy_100bt_ab.jld2"; results=ab_res)

# ╔═╡ 52485563-9ac1-46df-8c18-ce94507782c0
# ╠═╡ disabled = true
#=╠═╡
ab_res = load("./Data/nfb_esindy_100bt_ab.jld2")["results"]
  ╠═╡ =#

# ╔═╡ 2bf54606-0ad6-4e2f-96aa-5a80d98772a4
md"""
###### Plot results
"""

# ╔═╡ e8c5545d-9988-42f9-bf2c-bbd0199f7655
#=╠═╡
begin
	# Plot the results
	ab_plots = plot_esindy(ab_res, iqr=false)
	plot(ab_plots[1], ab_plots[2], layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB on g1p and g2p")
end
  ╠═╡ =#

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
# ╠═╡ disabled = true
#=╠═╡
begin
	# Plot the results
	nofb_plots = plot_esindy(nofb_res)
	plot(nofb_plots[1], nofb_plots[2], ylim=(0, 2), layout=(2,1), size=(800, 800), plot_title="E-SINDy results with no FB")
end
  ╠═╡ =#

# ╔═╡ f6354d11-84ae-459a-807a-7d759d7c07f7
#savefig("./Plots/nfb_esindy_nofb.svg")

# ╔═╡ c7808cd2-5b58-4eb6-88fe-0a7fbb66717d
md"""
## Accessory functions
"""

# ╔═╡ Cell order:
# ╟─9ebfadf0-b711-46c0-b5a2-9729f9e042ee
# ╠═6024651c-85f6-4e53-be0f-44f658cf9c77
# ╠═b9ff1932-20ef-48da-9d2d-d61f3ee13a4f
# ╠═6b5db2ee-bb76-4617-b6b1-e7b6064b0fb9
# ╠═a806fdd2-5017-11ef-2351-dfc89f69b334
# ╟─d6b1570f-b98b-4a98-b5e5-9d747930a5ab
# ╟─8f821358-68cf-4ffd-88d3-37b64238f5ef
# ╟─666a20eb-8395-4859-ae70-aa8ea22c5c77
# ╟─f21876f0-3124-40ad-ad53-3a53efe77040
# ╟─fd914104-90da-469c-a80f-f068cac51a1c
# ╟─af06c850-2e8b-4b4b-9d0f-02e645a79743
# ╟─19225f7f-257d-4857-a709-6eb623985ba4
# ╟─129f37f3-2399-44ec-9d01-fca76015f755
# ╟─70cf3834-18d3-4732-837c-5af620c5f005
# ╟─b549bff5-d8e9-4f41-96d6-2d562584ccd9
# ╟─dd110815-cb48-4abd-8604-a275e3ae07b8
# ╟─6c7929f5-15b2-4e19-8c26-e709f0da182e
# ╟─447a8dec-ae5c-4ffa-b672-4f829c23eb1f
# ╟─d9b1a318-2c07-4d44-88ec-501b2cfc940b
# ╟─d8eb7ce2-7efb-47e0-9404-e58eea193225
# ╟─c8841434-a7af-4ade-a444-e6bc47575811
# ╟─74ad0ae0-4406-4326-822a-8f3e027077b3
# ╠═9dc0a251-a637-4144-b32a-7ebf5b86b6f3
# ╟─ede9d54f-aaa4-4ff3-9519-14e8d32bc17f
# ╠═dafd3b7d-f1ce-4c1f-8ba4-6ddf09e4f7d4
# ╟─d7cc32e8-9bb0-4d40-98b0-5af57b16e8fd
# ╟─0f5e0785-2778-4fde-b010-7fcf813ceed2
# ╟─219b794d-9f51-441b-9197-b8ef0c4495b4
# ╠═4646bfce-f8cc-4d24-b461-753daff50f71
# ╟─034f5422-7259-42b8-b3b3-e34cfe47b7b7
# ╟─0f7076ef-848b-4b3c-b918-efb6419787be
# ╟─5ef43425-ca26-430c-a62d-e194a8b1aebb
# ╟─c7e825a3-8ce6-48fc-86ac-21810d32bbfb
# ╟─79b03041-01e6-4d8e-b900-446511c81058
# ╠═85c913fa-23be-4d01-9a78-47f786d9a4cd
# ╠═d0fd11e2-0f4c-42f4-b90f-fda24269b6e2
# ╠═0d136529-28a5-406c-958d-28df99498f66
# ╠═1f055fd8-c26e-46e0-bffe-51cadbfb6410
# ╠═4feff5b4-66dd-4e69-ab1e-95d4363c8cb0
# ╠═acf3c904-9784-4b43-983e-8dd35763c100
# ╠═74b2ade4-884b-479d-9fee-828d37d7ab47
# ╟─64246dd6-8aa6-4ebd-8385-624f5bffd7c7
# ╠═97ae69f0-764a-4aff-88cd-91accf6bb3fd
# ╠═74066f54-6e50-4db0-a98f-dc1eb9999899
# ╠═aefb3b75-af6a-4bc9-98f8-713144b24c5a
# ╟─d7e68526-5ba9-41ed-9669-e7a289db1e93
# ╟─cc2395cc-00df-4e5e-8573-e85ce813fd41
# ╠═012a5186-03aa-482d-bb62-ecba49587877
# ╟─569c7600-246b-4a64-bba5-1e74a5888d8c
# ╟─e49b5f76-d04b-4b19-a4c8-a3a3a1b40f1b
# ╟─1f3f1461-09f1-4825-bb99-4064e075e23e
# ╠═b73f3450-acfd-4b9e-9b7f-0f7289a62976
# ╟─3b541b2d-6644-4376-beb8-6e8587be286c
# ╟─df7f0f7e-1bfa-4d7e-a817-8dc20a3ec0c4
# ╟─92f3a5ea-9838-400a-9120-52534d9cad52
# ╟─02c625dc-9d21-488e-983b-c3e2c40e0aad
# ╠═04538fbf-63b7-4394-b281-f047d0c3ea51
# ╠═62d79e8d-6d82-4fea-a596-8c32dd16d78e
# ╟─3bd630ad-f546-4b0b-8f3e-98367de739b1
# ╠═5c400147-353a-43f2-9de6-c234e13f06c9
# ╠═a7741d65-e0b5-4547-be64-59cb736132b5
# ╠═de4c57fe-1a29-4354-a5fe-f4e184de4dd3
# ╟─03993d5c-7900-454a-9fe3-a70bd69456d1
# ╟─e675346b-99dc-441f-913b-19387aaef3ea
# ╟─50d20cc1-ce44-43b6-9d72-1ff36137f6f8
# ╟─1cc0a861-8c92-4a05-85b6-6b1dfe406f2e
# ╟─2e472b7f-0863-470e-a91a-13aafd12ce0b
# ╟─900aabf8-cce0-4aa6-a257-8739b977e9b3
# ╟─8f7e784d-1f35-49bf-ba73-d2306453e258
# ╟─65732e07-69dd-43d5-b04c-72444c78c470
# ╟─34be558a-80bb-41b5-80ae-4d982e612c06
# ╟─4a1ab1e5-90cb-4b87-abdc-7278cbca407f
# ╟─664b3e67-09d3-4e98-a946-ffbcbed0fe90
# ╟─c182982b-6f5a-41fe-a708-9e7786e090a5
# ╟─d0e65b25-0ea7-46c1-ac15-99eea43b6ade
# ╟─77c1d355-8a7e-414e-9e0e-8eda1fbbbf1d
# ╟─5e692d14-c785-4f8c-9e9a-f581c56e1bc8
# ╟─5a8428a3-eba1-4960-8033-bb537eda034f
# ╟─07910dfe-9a66-4fc2-bbb3-66a18ed2b7a6
# ╟─4edf87b9-ee15-47d2-91e1-d7c683d9cc1f
# ╠═699c7160-7ca7-46db-917f-400f729277ba
# ╟─fa126d0c-868c-4083-b401-02504c033605
# ╟─dd11964a-1693-47d5-830f-3c4c632efe83
# ╠═20869d4b-ccd5-479a-b0d9-3271bef9921d
# ╟─f9f4822e-e5cc-4c15-9131-30c8ec2e0b83
# ╟─6d7969d7-5753-4f4e-bace-7de3a542103e
# ╟─a692aaa9-3b35-44cf-9293-ae0e2faf1eeb
# ╟─973f831a-4050-4c34-a868-091f335bcbb4
# ╟─6d20c14d-7fed-4812-a677-d7d85fdbe725
# ╟─d34550ae-57fb-44b9-bd0f-e03fdf47b058
# ╟─99c86351-4a05-45cb-b842-7b7c2d37a7e1
# ╟─18d05848-5789-414d-ab88-439c60959899
# ╠═34c1f8fd-27de-4d89-a4f4-c8fbf62c54f6
# ╟─bcd1baf8-548c-4798-806e-67a92238be07
# ╟─a8d77be8-17f3-42e5-9ba2-b22df0ff2b04
# ╟─215e6f0b-faa1-4e80-b946-3fbc6d048058
# ╠═947b426f-bf18-4c43-bf18-85bc8b7df201
# ╠═dbbdd00c-c320-48a4-801c-c3be9701832b
# ╟─a42646a0-a861-4687-9580-eb37e14dc05f
# ╟─b8cfb8e1-1d4d-410e-8629-5c93f4d15d78
# ╟─52485563-9ac1-46df-8c18-ce94507782c0
# ╟─2bf54606-0ad6-4e2f-96aa-5a80d98772a4
# ╠═e8c5545d-9988-42f9-bf2c-bbd0199f7655
# ╟─8f6071b2-b9d7-46a4-8ec0-aadb3f01692e
# ╟─4ae9802d-fdbc-449c-a560-391913369306
# ╟─9910d0af-d0af-4543-ad4f-29c043b70a63
# ╟─0529134b-963e-4c10-bb1b-848db02a2f99
# ╟─837f137d-be72-4cdd-ad3e-6153754bad08
# ╟─ab383b83-7ff1-4e11-b7b8-256b6c78e703
# ╟─7fdae884-4a02-4f61-8b53-616e14bef473
# ╟─a46a7ac1-4b0c-45cf-a191-240bc4796b20
# ╠═249d11db-7af2-48f8-bc04-961b52816d4a
# ╟─f6354d11-84ae-459a-807a-7d759d7c07f7
# ╟─c7808cd2-5b58-4eb6-88fe-0a7fbb66717d
