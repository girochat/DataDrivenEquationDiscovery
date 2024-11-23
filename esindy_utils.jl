# Module that contains ERK model-specific functions
module ERKModule
    
    # Standard libraries
    using CSV, DataFrames
    
    # External libraries
    using SmoothingSplines

    # Data Driven Equation Discovery packages
    using DataDrivenDiffEq, DataDrivenSparse


    export make_labels
    export create_data
    export build_basis


    # Make sample labels for plotting
    function make_labels(files)
    	labels = []
        case = ""
    	for file in files
            words = split(file, "/")
    		words = split(words[end], ".")
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
                elseif occursin("gf", lowercase(word))
                    case = lowercase(word)
                end
    		end
    		push!(labels, label)
    	end
    	return (labels=labels, case=case)
    end


    # Create E-SINDy compatible data structure out of dataframes
    function create_data(files; smoothing=0., var_idxs=[2,4])
    
    	# Load data into dataframe
    	df = CSV.read(files[1], DataFrame)
    	if length(files) > 1
    	    for i in 2:length(files)
    	        df2 = CSV.read(files[i], DataFrame)
    	        df = vcat(df, df2)
    	    end
    	end

    	# Create labels for plotting
    	labels, case = make_labels(files)
    
    	# Retrieve the number of samples
    	n_samples = sum(df.time .== 0)
    	size_sample = Int(nrow(df) / n_samples)
    
    	# Define relevant data for E-SINDy
    	time = df.time
    	X = [df.R_fit df.Raf_fit df.ERK_fit df.PFB_fit df.NFB_fit]
    	if case == "ngf"
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
    
    	return (time=time, X=X[:,var_idxs], Y=Y, GT=GT, labels=labels, case=case)
    end


    # Function to build a basis
    function build_basis(x, i)
        
        # Define a basis of functions to estimate the unknown equation of GFs model
    	h = DataDrivenDiffEq.polynomial_basis(x, 2)
    	basis = DataDrivenDiffEq.Basis([h; h .* i], x, implicits=i)
        return basis
    end


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
    function create_data(files, smoothing=0.)
    
    	# Load data into dataframe
    	df = CSV.read(files[1], DataFrame)
    	if length(files) > 1
    	    for i in 2:length(files)
    	        df2 = CSV.read(files[i], DataFrame)
    	        df = vcat(df, df2)
    	    end
    	end
    
    	# Create labels for plotting and retrieve NFB case
    	labels, case = make_labels(files)
    
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
    function build_basis(x, i)
    
        # Define a basis of functions to estimate the unknown equation of NFB model
    	h = DataDrivenDiffEq.monomial_basis(x, 3)
    	basis = DataDrivenDiffEq.Basis(h, x)
        return basis
    end
end



# Module that contains E-SINDy functions
module ESINDyModule

    # SciML tools
    import ModelingToolkit, Symbolics
    
    # Standard libraries
    using StatsBase, Statistics, Plots
    
    # External libraries
    using HyperTuning, StableRNGs, Distributions, ColorSchemes, ProgressMeter
    
    # Packages under development (debugging)
    using DataDrivenDiffEq, DataDrivenSparse

    # Set a random seed for reproducibility
    rng = StableRNG(1111)

    export objective
    export get_best_hyperparameters
    export sindy_bootstrap
    export library_bootstrap
    export simplify_expression
    export compute_ecoef
    export get_coef_mask
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
    global sampler = DataDrivenDiffEq.DataProcessing(split = 0.8, shuffle = true, batchsize = 400)
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
    	scenario = Scenario(λ = 1e-3:1e-3:9e-1, ν = exp10.(-3:1:3),
    		max_trials = 500, 
    		sampler=HyperTuning.RandomSampler())
    
    	# Find optimal hyperparameters
    	hp_res = HyperTuning.optimize(trial -> objective(trial, dd_prob, basis, with_implicits), scenario)
    
    	return hp_res.best_trial.values[:λ], hp_res.best_trial.values[:ν]
    end
    
    
    
    # Bootstrapping function that estimate optimal library coefficients given data
    function sindy_bootstrap(data, basis, n_bstraps, data_fraction)
    
    	# Initialise the coefficient array
    	n_eqs = size(data.Y, 2)
    	l_basis = length(basis)
    	bootstrap_coef = zeros(n_bstraps, n_eqs, l_basis)

        # Track best hyperparameters
    	hyperparam = (λ = [], ν = [])

        # Track progress while using parallelisation
        progress = []        
 
    	print_flush("Starting E-SINDy Bootstrapping:")
    	Threads.@threads for j in 1:n_bstraps

            # Randomly select 75% of the samples
    		n_samples = sum(data.time .== 0)
    		size_sample = Int(length(data.time) / n_samples)
    		rand_samples = sample(1:n_samples, floor(Int, n_samples * 0.75), replace=false)
    		
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
    		implicits = getfield(basis, :implicit)
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
    				
    				best_res = DataDrivenDiffEq.solve(dd_prob, basis, 
                    ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), options=options)
    				
                    # Simplify the symbolic expression to get the final coefficients
    				simpl_coefs = simplify_expression(best_res.out[1].coefficients[1,:], basis)
    				
    				# Store library coefficient for current bootstrap
    				bootstrap_coef[i,eq,:] = simpl_coefs
    			end
    		else
    			dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(X', Y')
    			
    			# Solve problem with optimal hyperparameters
    			best_λ, best_ν = get_best_hyperparameters(dd_prob, basis, with_implicits)
    			push!(hyperparam.λ, best_λ)
    			push!(hyperparam.ν, best_ν)
    			best_res = DataDrivenDiffEq.solve(dd_prob, basis, DataDrivenSparse.SR3(best_λ, best_ν), 
                options = options)
    			
    			# Store library coefficient for current bootstrap
    			bootstrap_coef[j,:,:] = best_res.out[1].coefficients
    		end

            # Print progress 
            push!(progress, j)
            if length(progress) % 10 == 0
                print_flush("Progress: $(length(progress))/$(n_bstraps)")
            end 
    	end
    	return bootstrap_coef, hyperparam
    end




    # Bootstrapping function that estimate optimal library terms given data
    function library_bootstrap(data, basis, n_bstraps, n_libterms; implicit_id=none)
    
    	# Initialise the coefficient array
    	n_eqs = size(data.Y, 2)
    	l_basis = length(basis)
    	bootstrap_coef = zeros(n_bstraps, n_eqs, l_basis)
    	indices = [1]
    	best_bic = 1000

        # Track progress while using parallelisation
        progress = [] 
    
    	print_flush("Starting Library E-SINDy Bootstrapping:")
    	Threads.@threads for j in 1:n_bstraps
    		for eq in 1:n_eqs

                if n_eqs > 1 && j == 1
                    print_flush("Equation $(eq):")
                end
    		
    			# Check if the problem involves implicits
    			implicits = implicit_variables(basis)
    			with_implicits = false
    
    			# Create bootstrap library basis
    			if !isempty(implicits)
    				with_implicits = true
    				idxs = [1:(implicit_id-1); (implicit_id+1):l_basis]
    				rand_ind = [sample(idxs, n_libterms, replace=false); implicit_id]
    				bt_basis = DataDrivenDiffEq.Basis(basis[rand_ind], 
                    x[1:size(data.X, 2)], implicits=i[1:1])
    			else
    				rand_ind = sample(1:l_basis, n_libterms, replace=false)
    				bt_basis = DataDrivenDiffEq.Basis(basis[rand_ind], x[1:size(data.X, 2)])
    			end
    	
    			# Solve data-driven problem with optimal hyperparameters
    			dd_prob = DataDrivenDiffEq.DirectDataDrivenProblem(data.X', data.Y[:,eq]')
    			best_λ, best_ν = get_best_hyperparameters(dd_prob, bt_basis, with_implicits)
    			if with_implicits
    				best_res = DataDrivenDiffEq.solve(dd_prob, bt_basis, 
                    ImplicitOptimizer(DataDrivenSparse.SR3(best_λ, best_ν)), options=options)
    			else
        			best_res = DataDrivenDiffEq.solve(dd_prob, bt_basis, 
                    DataDrivenSparse.SR3(best_λ, best_ν), options = options)
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

                # Print progress 
                push!(progress, j)
                if length(progress) % 10 == 0
                    print_flush("Progress: $(length(progress))/$(n_bstraps)")
                end 
    		end    
    	end
    	return bootstrap_coef 
    end




    # Function to simplify the expression of the final equation returned by SINDy (limit redundancy of 
    # expressions)
    function simplify_expression(coefs, basis)
    	
    	# Get only right hand side of the library terms
    	h = [equation.rhs for equation in equations(basis)]
    
    	# Get the expression out of the coefficients and the basis
    	expression = sum(coefs .* h)
    
    	# Solve wrt the implicit variable and simplify
    	implicit = getfield(basis, :implicit)
    	implicit_id = findall(x -> (x == 0.1), [substitute(term.rhs, Dict([i[1] => 0.1, 
                        x[1] => 2, x[2] => 3, x[3] => 4, x[4] => 5, x[5]=> 6])) for term in basis])
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
    					push!(masks, (mask=mask, freq=freq/n_bootstraps, coef_mean=mean(coefs),
                            coef_std=coef_std/sqrt(n_bootstraps)))
    				end
    			end
    		end
    		push!(eq_masks, masks)
    	end
    	return eq_masks
    end




    # Function to build the reduced basis based on Library E-SINDy results
    function build_basis(lib_bootstraps, basis)

        # Find all library terms that were picked during Library E-SINDy
        lib_stats = compute_coef_stat(lib_bootstraps, 0)
    	lib_ind = findall(!iszero, lib_stats.median[1,:])

        # Build library basis
    	lib_basis = DataDrivenDiffEq.Basis(basis[lib_ind], x, implicits=i[1:1])
    	return lib_basis
    end



    
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
    


    
    # Function to compute the equations output (y) given the input (x) 
    function get_yvals(data, equations)
    	n_eqs = length(equations)
    
    	yvals = []
    	for eq in equations
    		push!(yvals, [eq(x) for x in eachrow([data.X data.Y])])
    	end
    	return yvals
    end




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
    		current_eqs = build_equations(current_coef, basis, verbose=false)
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




    # Plotting function for E-SINDy results
    function plot_esindy(results; sample_idxs=nothing, iqr=true)
    
    	# Retrieve results
    	data, basis = results.data, results.basis
    	coef_median = results.coef_median
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
    					plot!(p, data.time[i_start:i_end], y, label=data.labels[sample+1], 
                        color=palette[i_color],
    					ribbon=(y - ci_low[i_start:i_end, eq], ci_up[i_start:i_end, eq] - y), 
                        fillalpha=0.15)
    				else
    					plot!(p, data.time[i_start:i_end], y, label=data.labels[sample+1], 
                        color=palette[i_color])
    				end
    				
    				plot!(p, data.time[i_start:i_end], data.GT[i_start:i_end, eq], label="", 
                    linestyle=:dash, color=palette[i_color])
                    i_color = i_color + 1
    			end
                
    		end
    		plot!(p, [], [],  label="GT", color=:black, linestyle=:dash)
    		push!(subplots, p)
    	end
    	return subplots
    end




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
    	y = build_equations(e_coef, basis)
    	
    	return (data=data, 
            basis=basis, 
            equations=y, 
            bootstraps=bootstrap_res, 
            coef_median=e_coef, 
            masks=masks,
            hyperparameters=hyperparameters)
    end
end
