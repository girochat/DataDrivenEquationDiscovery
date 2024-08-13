### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ b9ff1932-20ef-48da-9d2d-d61f3ee13a4f
begin 
	import Pkg
	Pkg.activate(".")
	Pkg.Registry.status()
end

# ╔═╡ a5696976-2dfc-4e98-ae0b-571ab8c0e4c3
using Revise

# ╔═╡ c01b0a96-612a-48fd-a982-ebd5c259adc0
begin
	using DataDrivenDiffEq
	using DataDrivenSparse
end

# ╔═╡ a806fdd2-5017-11ef-2351-dfc89f69b334
begin
	# SciML tools
	import ModelingToolkit, Symbolics
	
	# Standard libraries
	using Statistics, Plots, CSV, DataFrames, Printf

	# External libraries
	using HyperTuning, StableRNGs, Distributions, SmoothingSplines, Logging

	# Set a random seed for reproducibility
	rng = StableRNG(1111)

	gr()
end

# ╔═╡ 9ebfadf0-b711-46c0-b5a2-9729f9e042ee
md"""
### Equation discovery with E-SINDy
This notebook is about estimating the equation describing the unknown dynamics approximated by the neural network. It uses the Ensemble SINDy approach described in "Discovering governing equations from data by sparse identification of nonlinear dynamical systems" by Brunton et al. (2016).
"""

# ╔═╡ 6b5db2ee-bb76-4617-b6b1-e7b6064b0fb9
begin
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

# ╔═╡ af06c850-2e8b-4b4b-9d0f-02e645a79743
begin
	# Load the NFB model estimations for various concentrations
	df_nofb_01 = CSV.read("./Data/NFB_nofb_01.csv", DataFrame)
	df_nofb_005 = CSV.read("./Data/NFB_nofb_005.csv", DataFrame)
	df_nofb_001 = CSV.read("./Data/NFB_nofb_001.csv", DataFrame)
	df_nofb_0001 = CSV.read("./Data/NFB_nofb_0001.csv", DataFrame)
	df_nofb = vcat(df_nofb_01, df_nofb_005, df_nofb_001, df_nofb_0001)
	
	df_a_01 = CSV.read("./Data/NFB_a_01.csv", DataFrame)
	df_a_005 = CSV.read("./Data/NFB_a_005.csv", DataFrame)
	df_a_001 = CSV.read("./Data/NFB_a_001.csv", DataFrame)
	df_a_0001 = CSV.read("./Data/NFB_a_0001.csv", DataFrame)
	df_a = vcat(df_a_01, df_a_005, df_a_001, df_a_0001)
	
	df_b_01 = CSV.read("./Data/NFB_b_01.csv", DataFrame)
	df_b_005 = CSV.read("./Data/NFB_b_005.csv", DataFrame)
	df_b_001 = CSV.read("./Data/NFB_b_001.csv", DataFrame)
	df_b_0001 = CSV.read("./Data/NFB_b_0001.csv", DataFrame)
	df_b = vcat(df_b_01, df_b_001, df_b_005, df_b_0001)
	
	df_ab_01 = CSV.read("./Data/NFB_ab_01.csv", DataFrame)
	df_ab_005 = CSV.read("./Data/NFB_ab_005.csv", DataFrame)
	df_ab_001 = CSV.read("./Data/NFB_ab_001.csv", DataFrame)
	df_ab_0001 = CSV.read("./Data/NFB_ab_0001.csv", DataFrame)
	df_ab = vcat( df_ab_001,  df_ab_0001, df_ab_005,df_ab_01)
end

# ╔═╡ b549bff5-d8e9-4f41-96d6-2d562584ccd9
md"""
###### -EGF/NGF model
"""

# ╔═╡ f598564e-990b-436e-aa97-b2239b44f6d8
begin
	# Load the NGF model estimations for various pulse regimes
	df1 = CSV.read("./Data/ngf_highCC_10m_10v.csv", DataFrame)
	df2 = CSV.read("./Data/ngf_highCC_10m_3pulse.csv", DataFrame) # missing NN_GT!!
	df3 = CSV.read("./Data/ngf_highCC_3m_20v.csv", DataFrame)
	df4 = CSV.read("./Data/ngf_highCC_10m.csv", DataFrame)
	df5 = CSV.read("./Data/ngf_highCC_3m_3v.csv", DataFrame)
	df6 = CSV.read("./Data/ngf_lowCC_10m_10v.csv", DataFrame)
	df_NGF = vcat(df1, df3, df4, df5, df6)
end

# ╔═╡ f21876f0-3124-40ad-ad53-3a53efe77040
function create_nfb_data(df, labels = [], smoothing=0.)

	# Retrieve the number of samples
	n_samples = sum(df.time .== 0)
	size_sample = Int(nrow(df) / n_samples)

	time = df.time
	X = [df.g1p_fit df.g2p_fit df.Xact_fit]
	GT = (0.75 .* df.PFB_fit .* 
		(1 .- df.Raf_fit) ./ (0.01 .+ (1 .- df.Raf_fit)))
	Y = df.NN_approx

	if smoothing != 0
		smoothed_Y = zeros(size(Y))
		for i in 0:(n_samples-1)
			x_t = df.time[1 + i*size_sample: (i+1)*size_sample]
			y = df.NN_approx[1 + i*size_sample: (i+1)*size_sample]
			λ = smoothing  # smoothing parameter
			spl = fit(SmoothingSpline, x_t, y, λ)
			smoothed_y = predict(spl)
			smoothed_Y[1 + i*size_sample: (i+1)*size_sample] = smoothed_y
		end
	end

	@assert size(X, 1) == size(Y, 1)
		
	return (time=time, X=X, Y=smoothed_Y, GT=GT, labels=labels)
end

# ╔═╡ 6c7929f5-15b2-4e19-8c26-e709f0da182e
function create_gf_data(df, labels = [], smoothing=0.)

	# Retrieve the number of samples
	n_samples = sum(df.time .== 0)
	size_sample = Int(nrow(df) / n_samples)

	time = df.time
	X = [df.Raf_fit df.PFB_fit]
	GT = (0.75 .* df.PFB_fit .* 
		(1 .- df.Raf_fit) ./ (0.01 .+ (1 .- df.Raf_fit)))
	Y = df.NN_approx

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
		
	return (time=time, X=X, Y=smoothed_Y, GT=GT, labels=labels)
end

# ╔═╡ 5f0fb956-7c84-4ff2-944f-e9ddfccab748
ngf_data = create_gf_data(df_NGF, ["High CC 10'/10'" "High CC 3'/20'" "High CC 10'" "High CC 3'/3'" "Low CC 10'/10'"], 300.)

# ╔═╡ 74ad0ae0-4406-4326-822a-8f3e027077b3
md"""
##### Set up the SINDy library of functions and hyperparameters optimisation
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
	gf_h = DataDrivenDiffEq.polynomial_basis(x[1:2], 2)
	gf_basis = DataDrivenDiffEq.Basis([gf_h; gf_h .* i], x[1:2], implicits=i[1:1])
end

# ╔═╡ 4646bfce-f8cc-4d24-b461-753daff50f71
begin
	sampler = DataDrivenDiffEq.DataProcessing(split = 0.8, shuffle = true, batchsize = 30)
	options = DataDrivenDiffEq.DataDrivenCommonOptions(data_processing = sampler, digits=1, abstol=1e-10, reltol=1e-10, denoise=true)
end

# ╔═╡ 9c4ade80-d5ad-40b4-bf57-53479bb8a74b
ngf_dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(ngf_data.X', ngf_data.Y')

# ╔═╡ e488c35e-d454-4942-b08e-599d27f00e17
res_dd = DataDrivenDiffEq.solve(
			ngf_dd_prob, 
			gf_basis, 
			ImplicitOptimizer(DataDrivenSparse.SR3(0.001, 10)), 
			options = options)

# ╔═╡ 1ec3d1b5-83d9-4093-8c1f-e726441743a6
coef = res_dd.out

# ╔═╡ d7ffe014-f4bf-4236-a553-d0dd6f06f146
begin
	metric_bic = []
	metric_dof = []
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

	push!(metric_dof, res_dd.out[1].dof)
	push!(metric_bic, bic(res_dd))
	return bic(res_dd)

end

# ╔═╡ 0f7076ef-848b-4b3c-b918-efb6419787be
function get_best_hyperparameters(dd_prob, basis, with_implicits)
	
	# Define the range of hyperparameters to consider
	scenario = Scenario(λ = 1e-2:1e-3:9e-1, ν = exp10.(-2:1:3),
		max_trials = 300, 
		sampler=HyperTuning.RandomSampler(), batch_size=1)
	
	hp_res = HyperTuning.optimize(trial -> objective(trial, dd_prob, basis, with_implicits), scenario)

	return hp_res.best_trial.values[:λ], hp_res.best_trial.values[:ν]
end

# ╔═╡ 5ef43425-ca26-430c-a62d-e194a8b1aebb
md"""
##### E-SINDy Bootstrapping
"""

# ╔═╡ c7e825a3-8ce6-48fc-86ac-21810d32bbfb
# Bootstrapping function that estimate optimal library coefficients given data
function sindy_bootstrap(data, basis, n_bstraps)

	n_eqs = size(data.Y, 2)
	l_basis = length(basis)
	bootstrap_coef = zeros(n_bstraps, n_eqs, l_basis)

	for i in 1:n_bstraps
		rand_ind = rand(1:size(data.X, 1), size(data.X, 1))
		X = data.X[rand_ind,:]
		Y = data.Y[rand_ind,:]
		dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(X', Y')

		# Check if the problem involves implicits
		implicits = implicit_variables(basis)
		with_implicits = false
		if !isempty(implicits)
			with_implicits = true
		end
		
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

# ╔═╡ 74b2ade4-884b-479d-9fee-828d37d7ab47
function compute_coef_stat(bootstrap_res, coef_threshold)

	# Retrieve dimensions of the problem
	n_eqs = size(bootstrap_res, 2)
	n_terms = size(bootstrap_res, 3)

	# Compute inclusion probabilities
	inclusion_prob = (mean((bootstrap_res .!= 0), dims=1) * 100)

	# Keep only elements of basis with probabilities above threshold
	mask = inclusion_prob .> coef_threshold
	masked_res = bootstrap_res .* mask

	# Compute the mean and std of ensemble coefficients
	m = zeros(Float64, n_eqs, n_terms)
	sem = zeros(Float64, n_eqs, n_terms)
	for i in 1:n_eqs
		for j in 1:n_terms
			m[i,j] = mean(filter(!iszero, masked_res[:,i,j]))
			sem[i,j] = std(filter(!iszero, masked_res[:,i,j]))
			sample_size = sum(.!(iszero.(masked_res[:,i,j])))
			if sample_size != 0
				sem[i,j] = sem[i,j] / sqrt(sample_size)
			end
		end
	end
	m[isnan.(m)] .= 0
	sem[isnan.(sem)] .= 0

	return (mean = m, SEM = sem)
end

# ╔═╡ 97ae69f0-764a-4aff-88cd-91accf6bb3fd
function build_equations(coef, basis)

	# Build equation
	h = [equation.rhs for equation in DataDrivenDiffEq.equations(basis)]
	final_eqs = [sum(row .* h) for row in eachrow(coef)]

	implicits = implicit_variables(basis)
	if !isempty(implicits)
		final_eqs = ModelingToolkit.solve_for(final_eqs, implicits)
	end

	# Build callable function and print equation
	y = []
	println("Estimated equation(s):")
	for (j, eq) in enumerate(final_eqs)
		println("y$(j) = $(eq)")
		push!(y, Symbolics.build_function(eq, [x[1:2]; i[1]], expression = Val{false}))
	end
	
	return y
end

# ╔═╡ cc2395cc-00df-4e5e-8573-e85ce813fd41
function plot_esindy(data, y, sem, basis, confidence)
	
	# Get the quantile corresponding to the confidence interval
	Z = Normal()
	if 0 <= confidence <= 1
		q = quantile(Z, confidence)
	else
		q = quantile(Z, confidence/100)
	end

	# Obtain the equations for the SEM
	#y_sem = build_equations(sem, basis)

	# Retrieve the number of samples
	n_samples = sum(data.time .== 0)
	size_sample = Int(length(ngf_data.time) / n_samples)
	
	n_eqs = size(y, 1)
	subplots = []
	palette = :default
	for i in 1:n_eqs
		
		p = plot(title="Equation $(i)")
		for sample in 0:(n_samples-1)
			i_start = 1 + sample * size_sample
			i_end = i_start + size_sample - 1
			
			y_vals = [y[i](x) for x in eachrow([ngf_data.X[i_start:i_end,:] ngf_data.Y[i_start:i_end]])] 
			if length(y_vals) == 1
				y_vals = repeat([y_vals], size_sample)
			end
			
			sem_vals = [y[1](x) for x in eachrow([ngf_data.X[i_start:i_end,:] ngf_data.Y[i_start:i_end]])] 
			if length(sem_vals) == 1
				sem_vals = repeat([sem_vals], size_sample)
			end

			plot!(p, ngf_data.time[i_start:i_end], y_vals, label="SINDy (sample $(sample+1))")#, ribbon=q*sem_vals, fillalpha=0.3)#, color=palette[i+1])

			plot!(p, ngf_data.time[i_start:i_end], ngf_data.GT[i_start:i_end],
			linestyle=:dash, label="GT")
		end
		push!(subplots, p)
	end
	#main_p = plot(subplots[1], subplots[2], layout=(2,1), xlabel="Time", 
		#		ylabel="Model species", 
		#		plot_title="E-SINDy results", size=(600, 700))
	
	return subplots
end

# ╔═╡ 012a5186-03aa-482d-bb62-ecba49587877
function e_sindy(data, basis, n_bstrap, coef_threshold, confidence = 65)

	# Run sindy bootstraps
	bootstrap_res = sindy_bootstrap(data, basis, n_bstrap)

	# Compute the mean and std of ensemble coefficients
	e_coef, coef_sem = compute_coef_stat(bootstrap_res, coef_threshold)

	# Build the final equation as callable functions
	y = build_equations(e_coef, basis)

	# Plot result
	plots = plot_esindy(df, y, coef_sem, basis, confidence)
	
	return (equations = y, coef_mean = e_coef, coef_sem = coef_sem, plots=plots)
end

# ╔═╡ 569c7600-246b-4a64-bba5-1e74a5888d8c
md"""
### GFs model
"""

# ╔═╡ cd655445-6bf6-4aee-8968-e324ea66601d
res = sindy_bootstrap(ngf_data, gf_basis, 1)

# ╔═╡ 689cd930-e06d-4077-a433-88430ed2079a
begin
	indices = sortperm(metric_dof)
	scatter(metric_dof[indices], metric_bic[indices])
end

# ╔═╡ f27a340f-8d78-4e8a-8606-82c7e85e50fc
# ╠═╡ disabled = true
#=╠═╡
res = sindy_bootstrap(ngf_data, gf_basis, 10)
  ╠═╡ =#

# ╔═╡ c6e9f4ae-bb48-4f4e-88e5-165e8d1f3c8e
e_coef, coef_sem = compute_coef_stat(res, 65)

# ╔═╡ 50f61ebc-5788-403d-bb62-010abbd1e618
begin
	# Build equation
	h = [equation.rhs for equation in DataDrivenDiffEq.equations(gf_basis)]
	final_eqs = [sum(row .* h) for row in eachrow(e_coef)]

	println(final_eqs[1])

	implicits = implicit_variables(gf_basis)
	final_eqs = ModelingToolkit.solve_for(final_eqs, implicits)
	
	# Build callable function and print equation
	y = []
	println("Estimated equation(s):")
	for (j, eq) in enumerate(final_eqs)
		println("y$(j) = $(eq)")
		push!(y, Symbolics.build_function(final_eqs[j], [x[1:2]; i[1]], expression = Val{false}))
	end
end

# ╔═╡ 0ef4979a-aa1a-48b4-b347-9b71d84e66b3
p = plot_esindy(ngf_data, y, coef_sem, gf_basis, 95)

# ╔═╡ f2e7f963-ba4a-44b7-a24c-313269780a0e
plot(p[1])

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
# ╠═b9ff1932-20ef-48da-9d2d-d61f3ee13a4f
# ╠═a5696976-2dfc-4e98-ae0b-571ab8c0e4c3
# ╠═c01b0a96-612a-48fd-a982-ebd5c259adc0
# ╠═a806fdd2-5017-11ef-2351-dfc89f69b334
# ╠═6b5db2ee-bb76-4617-b6b1-e7b6064b0fb9
# ╟─d6b1570f-b98b-4a98-b5e5-9d747930a5ab
# ╟─666a20eb-8395-4859-ae70-aa8ea22c5c77
# ╟─af06c850-2e8b-4b4b-9d0f-02e645a79743
# ╟─b549bff5-d8e9-4f41-96d6-2d562584ccd9
# ╟─f598564e-990b-436e-aa97-b2239b44f6d8
# ╟─f21876f0-3124-40ad-ad53-3a53efe77040
# ╟─6c7929f5-15b2-4e19-8c26-e709f0da182e
# ╟─5f0fb956-7c84-4ff2-944f-e9ddfccab748
# ╟─74ad0ae0-4406-4326-822a-8f3e027077b3
# ╟─9dc0a251-a637-4144-b32a-7ebf5b86b6f3
# ╟─ede9d54f-aaa4-4ff3-9519-14e8d32bc17f
# ╟─e81310b5-63e1-45b8-ba4f-81751d0fcc06
# ╠═4646bfce-f8cc-4d24-b461-753daff50f71
# ╠═9c4ade80-d5ad-40b4-bf57-53479bb8a74b
# ╠═e488c35e-d454-4942-b08e-599d27f00e17
# ╠═1ec3d1b5-83d9-4093-8c1f-e726441743a6
# ╠═d7ffe014-f4bf-4236-a553-d0dd6f06f146
# ╠═034f5422-7259-42b8-b3b3-e34cfe47b7b7
# ╠═0f7076ef-848b-4b3c-b918-efb6419787be
# ╟─5ef43425-ca26-430c-a62d-e194a8b1aebb
# ╠═c7e825a3-8ce6-48fc-86ac-21810d32bbfb
# ╟─74b2ade4-884b-479d-9fee-828d37d7ab47
# ╟─97ae69f0-764a-4aff-88cd-91accf6bb3fd
# ╠═cc2395cc-00df-4e5e-8573-e85ce813fd41
# ╠═012a5186-03aa-482d-bb62-ecba49587877
# ╟─569c7600-246b-4a64-bba5-1e74a5888d8c
# ╠═cd655445-6bf6-4aee-8968-e324ea66601d
# ╟─689cd930-e06d-4077-a433-88430ed2079a
# ╠═f27a340f-8d78-4e8a-8606-82c7e85e50fc
# ╠═c6e9f4ae-bb48-4f4e-88e5-165e8d1f3c8e
# ╠═50f61ebc-5788-403d-bb62-010abbd1e618
# ╠═0ef4979a-aa1a-48b4-b347-9b71d84e66b3
# ╠═f2e7f963-ba4a-44b7-a24c-313269780a0e
# ╟─77c1d355-8a7e-414e-9e0e-8eda1fbbbf1d
# ╟─f3077d61-cb49-4efb-aac0-66e8de6e15ae
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
