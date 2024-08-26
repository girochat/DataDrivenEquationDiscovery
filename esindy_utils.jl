# Module that contains ERK model-specific functions
module ERKModule
    
    # Standard libraries
    using CSV, DataFrames
    
    # External libraries
    using SmoothingSplines
    
    # Packages under development (debugging)
    using DataDrivenDiffEq, DataDrivenSparse


    export make_labels
    export create_data
    export build_basis


    # Make sample labels for plotting
    function make_labels(files)
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


    # Create E-SINDy compatible data structure out of dataframes
    function create_data(files, smoothing=0.)

        # Load data into dataframe
        df = CSV.read(files[1], DataFrame)
        if length(files) > 1
            for i in 2:length(files)
                df2 = CSV.read(files[i], DataFrame)
                df = vcat(df, df2)
            end
        end
    
    	# Retrieve the number of samples
    	n_samples = sum(df.time .== 0)
    	size_sample = Int(nrow(df) / n_samples)
    
    	# Define relevant data for E-SINDy
    	time = df.time
    	X = [df.Raf_fit df.PFB_fit] #[df.R_fit df.Ras_fit df.Raf_fit df.MEK_fit df.PFB_fit]
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

    	# Create labels for plotting
    	labels = make_labels(files)
    		
    	return (time=time, X=X, Y=smoothed_Y, GT=GT, labels=labels)
    end


    # Function to build a basis
    function build_basis(x, i)
        
        # Define a basis of functions to estimate the unknown equation of GFs model
    	h = DataDrivenDiffEq.polynomial_basis(x, 2)
    	basis = DataDrivenDiffEq.Basis([h; h .* i], x, implicits=i)
        return basis
    end
end


# Module that contains NFB model-specific functions
module NFBModule

    # Standard libraries
    using CSV, DataFrames
    
    # External libraries
    using SmoothingSplines
    
    # Packages under development (debugging)
    using DataDrivenDiffEq, DataDrivenSparse

    export make_labels
    export create_data
    export build_basis


    # Make sample labels for plotting
    function make_labels(files)
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

    
    # Create E-SINDy compatible data structure out of dataframes
    function create_data(files)
    
    	# Load data into dataframe
    	df = CSV.read(files[1], DataFrame)
    	if length(files) > 1
    	    for i in 2:length(files)
    	        df2 = CSV.read(files[i], DataFrame)
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


    # Function to build a basis
    function build_basis(x)
    
        # Define a basis of functions to estimate the unknown equation of NFB model
    	h = DataDrivenDiffEq.monomial_basis(x, 3)
    	basis = DataDrivenDiffEq.Basis(h, x, iv=t)
        return basis
    end
end



# Module that contains E-SINDy functions
module ESINDyModule

    # SciML tools
    import ModelingToolkit, Symbolics
    
    # Standard libraries
    using Statistics, Plots
    
    # External libraries
    using HyperTuning, StableRNGs, Distributions, ColorSchemes
    
    # Packages under development (debugging)
    using DataDrivenDiffEq, DataDrivenSparse

    # Set a random seed for reproducibility
    rng = StableRNG(1111)

    export objective
    export get_best_hyperparameters
    export sindy_bootstrap
    export library_bootstrap
    export compute_coef_stat
    export build_equations
    export get_yvals
    export compute_CI
    export plot_esindy
    export esindy

    # Declare necessary symbolic variables for the bases
    @ModelingToolkit.variables x[1:7] i[1:1]
    @ModelingToolkit.parameters t
    x = collect(x)
    i = collect(i)


    # Define a sampling method and the options for the data driven problem
    global sampler = DataDrivenDiffEq.DataProcessing(split = 0.8, shuffle = true, batchsize = 100)
    global options = DataDrivenDiffEq.DataDrivenCommonOptions(data_processing = sampler, digits=1, 
    abstol=1e-10, reltol=1e-10, denoise=true)
    
    function print_flush(message)
         println(message)
	 flush(stdout)
    end


    # Objective function for hyperparameter optimisation
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
    
    
    
    # Bootstrapping function that estimate optimal library coefficients given data
    function sindy_bootstrap(data, basis, n_bstraps)
    
    	# Initialise the coefficient array
    	n_eqs = size(data.Y, 2)
    	l_basis = length(basis)
    	bootstrap_coef = zeros(n_bstraps, n_eqs, l_basis)

        # Track best hyperparameters
    	hyperparam = (λ = [], ν = [])
    
    	print_flush("Starting E-SINDy Bootstrapping:")
    	for j in 1:n_bstraps
   	   	
                if j % 10 == 0
		    print_flush("Bootstrap $(i)/$(n_bstraps)")
		end
	     
    		# Define data driven problem with bootstrapped data
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
    
    		# Solve problem with optimal hyperparameters
    		best_λ, best_ν = get_best_hyperparameters(dd_prob, basis, with_implicits)
            push!(hyperparam.λ, best_λ)
    		push!(hyperparam.ν, best_ν)
        
    		if with_implicits
    			best_res = DataDrivenDiffEq.solve(dd_prob, basis, 
                ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), 
                options=options)
    
    		else
    			best_res = DataDrivenDiffEq.solve(dd_prob, basis, 
                DataDrivenSparse.SR3(best_λ, best_ν), 
                options = options) 			
    
    		end
    		bootstrap_coef[j,:,:] = best_res.out[1].coefficients	
    	end
    	return bootstrap_coef, hyperparam
    end


    # Bootstrapping function that estimate optimal library terms given data
    function library_bootstrap(data, basis, n_bstraps, n_libterms)
    
    	# Initialise the coefficient array
    	n_eqs = size(data.Y, 2)
    	l_basis = length(basis)
    	bootstrap_coef = zeros(n_bstraps, n_eqs, l_basis)
    
    	print_flush("Starting E-SINDy Library Bootstrapping:")
    	for j in 1:n_bstraps
   	   	
                if j % 10 == 0
		    print_flush("Bootstrap $(j)/$(n_bstraps)")
		end
    		
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
    			best_res = DataDrivenDiffEq.solve(dd_prob, bt_basis,
                ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), 
                options=options)
    		else
    			best_res = DataDrivenDiffEq.solve(dd_prob, bt_basis, 
                DataDrivenSparse.SR3(best_λ, best_ν), 
                options = options) 			
    
    		end
    		bootstrap_coef[j,:,rand_ind] = best_res.out[1].coefficients
    		
    	end
    	return bootstrap_coef 
    end
    
    
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
    


    function get_yvals(data, equations)
    	n_eqs = length(equations)
    
    	yvals = []
    	for eq in equations
    		push!(yvals, [eq(x) for x in eachrow([data.X ngf_data.Y])])
    	end
    	return yvals
    end



    # Fucntion to compute the confidence interval of the estimated equation
    function compute_CI(data, mean_coef, sem_coef, basis, confidence)
    
    	# Build normal distribution for non-zero coefficients
    	indices = findall(!iszero, mean_coef)
    	coef_distrib = [Normal(mean_coef[k], sem_coef[k]) for k in indices]
    
    	# Run MC simulations to estimate the distribution of estimated equation 
    	n_simulations = 1000
    	results = zeros(n_simulations, size(data.time, 1))
    	current_coef = zeros(size(mean_coef))
        for i in 1:n_simulations
    		sample = [rand(distrib) for distrib in coef_distrib]
    		current_coef[indices] = sample
    		current_y = build_equations(current_coef, basis, false)
    		current_y_vals = [current_y[1](x) for x in eachrow([data.X data.Y])]
    	
            # Calculate function value
            results[i, :] = current_y_vals
    	end
    	
    	if confidence > 1
    		confidence = confidence / 100
    	end
    	
        # Calculate confidence interval
        lower_percentile = (1 - confidence) / 2
        upper_percentile = 1 - lower_percentile
        ci = mapslices(row -> quantile(row, [lower_percentile, upper_percentile]),
    		results, dims=1)
    
        return (ci_low=ci[1,:], ci_up=ci[2,:])
    end



    # Plotting function for E-SINDy results
    function plot_esindy(results; sample_ids=nothing, confidence=0)
    
    	# Retrieve the number of samples
    	data, basis, y = results.data, results.basis, results.equations
    	n_samples = sum(data.time .== 0)
    	if isnothing(sample_ids)
    		sample_ids = 1:n_samples
    	end
    	size_sample = Int(length(data.time) / n_samples)
    
    	# Compute confidence interval if necessary
    	if confidence > 0
    		ci_low, ci_up = compute_CI(data, results.mean_coef, results.sem_coef, basis, confidence)
    	end
    	
    	# Plot results
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
    			if sample in sample_ids
    				
    				i_start = 1 + sample * size_sample
    				i_end = i_start + size_sample - 1
    				i_color = ceil(Int, 1 + sample * (length(palette) / n_samples))
                    				
    				y_vals = [y[i](x) for x in eachrow([data.X[i_start:i_end,:] data.Y[i_start:i_end]])] 
    				if length(y_vals) == 1
    					y_vals = repeat([y_vals], size_sample)
    				end
    
    				if confidence > 0
    					plot!(p, data.time[i_start:i_end], y_vals, label=data.labels[sample+1], 
                            color=palette[i_color],
                            ribbon=(y_vals-ci_low[i_start:i_end], ci_up[i_start:i_end]-y_vals), 
                            fillalpha=0.15)
    				else
    					plot!(p, data.time[i_start:i_end], y_vals, label=data.labels[sample+1],
                            color=palette[i_color])
    				end
    				
    				plot!(p, data.time[i_start:i_end], data.GT[i_start:i_end], label="", linestyle=:dash, 
                        color=palette[i_color])
    			end
    		end
    		plot!(p, [], [],  label="GT", color=:black, linestyle=:dash)
    		push!(subplots, p)
    	end
    	return subplots
    end
    
    
    # Complete E-SINDy function 
    function esindy(data, basis, n_bstrap=100; coef_threshold=15)
    
    	# Run sindy bootstraps
    	bootstrap_res, hyperparameters = sindy_bootstrap(data, basis, n_bstrap)
    
    	# Compute the mean and std of ensemble coefficients
    	e_coef, coef_sem = compute_coef_stat(bootstrap_res, coef_threshold)
    
    	# Build the final equation as callable functions
    	println("E-SINDy estimated equations:")
    	y = build_equations(e_coef, basis)
    	
    	return (data=data, 
            basis=basis, 
            equations=y, 
            bootstraps=bootstrap_res, 
            coef_mean=e_coef, 
            coef_sem=coef_sem,
            hyperparameters=hyperparameters)
    end
end
