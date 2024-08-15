# SciML tools
import OrdinaryDiffEq, ModelingToolkit, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches #DataDrivenDiffEq, DataDrivenSparse,

# Standard libraries
using Statistics, Plots, CSV, DataFrames, ComponentArrays

# External libraries
using Lux, Zygote, StableRNGs # LuxCUDA

# Set a random seed for reproducibility
rng = StableRNG(1112)

# Explicitly call Plots backend
gr()




### USER-DEFINED parameters (pulse duration, frequency, GF concentration)

# Retrieve file arguments
if length(ARGS) < 4
    error("Error! You need to specify as arguments: \n- Type of input concentration (high/low)\n
        - Type of growth factor (EGF/NGF)\n
        - Pulse duration\n
        - Pulse frequency (Note: enter 100 for only one pulse)")
else
    println("Running UDE approximation of EGF/NGF model for:")
    CC = uppercasefirst(lowercase(ARGS[1]))
    GF = lowercase(ARGS[2])
    pulse_duration = parse(Int, ARGS[3])
    pulse_frequency = parse(Int, ARGS[4])

    # Name output file to save the results according to parameters used
    if pulse_frequency < 100
        println("$(CC) concentration $(GF) with $(pulse_duration)' pulse every $(pulse_frequency)'")
        filename = string(GF, "_", CC, "CC_", pulse_duration, "m_", pulse_frequency, "v")
    else
        println("$(CC) concentration $(GF) with a single pulse of $(pulse_duration)'")
        filename = string(GF, "_", CC, "CC_", pulse_duration, "m")
    end
end

# Set the ODE parameters relative to specific case
GF_PFB = 0
GF_concentration = 0
if GF == "egf"
    GF_PFB = 0
    if CC == "high"
        GF_concentration = 25
    else
        GF_concentration = 1
    end
else
    GF_PFB = 0.75
    if CC == "high"
        GF_concentration = 50
    else
        GF_concentration = 2
    end
end

# Define the signal function
f_signal(t) = sum(tanh(100(t - i))/2 - tanh(100(t - (i + pulse_duration))) / 2 for i in range(0, stop=100, step=(pulse_frequency + pulse_duration)))
@ModelingToolkit.register_symbolic f_signal(t)




##### Define ODE system #####

function erk_dynamics!(du, u, p, t)

    ## PARAMETERS
    k_R, kd_R, GF = p[1:3]    											   # Receptor 
    k_Ras, kd_Ras, Km_Ras, Km_aRas, GAP = p[4:8]   							# Ras
    k1_Raf, k2_Raf, kd_Raf, Km_Raf, Km_aRaf, K_NFB, K_PFB, P_Raf = p[9:16]  # Raf
    k_MEK, kd_MEK, Km_MEK, Km_aMEK, P_MEK = p[17:21] 						# MEK 
    k_ERK, kd_ERK, Km_ERK, Km_aERK, P_ERK = p[22:26] 						# ERK
    k_NFB, kd_NFB, Km_NFB, Km_aR, Km_aNFB, P_NFB = p[27:32] 				# NFB
    k_PFB, kd_PFB, Km_PFB, Km_aPFB, P_PFB = p[33:37] 						# PFB

    ## EQUATIONS
    # Receptor equation
    du[1] = k_R * (1-u[1]) * GF * f_signal(t) - kd_R * u[1]
    
    # Ras equation
    du[2] = (k_Ras * u[1] * (1-u[2]) / (Km_Ras + (1-u[2])) - 
             kd_Ras * GAP * u[2] / (Km_aRas + u[2]))
    
    # Raf equation
    du[3] = (k1_Raf * u[2] * ((1-u[3]) / (Km_Raf + (1-u[3]))) * 
            ((K_NFB)^2 / (K_NFB^2 + u[6]^2)) - 
             kd_Raf * P_Raf * u[3] / (Km_aRaf + u[3]) + 
             k2_Raf * u[7] * (1-u[3]) / (K_PFB + (1-u[3])))
    
    # MEK equation
    du[4] = (k_MEK * u[3] * (1-u[4]) / (Km_MEK + (1-u[4])) - 
             kd_MEK * P_MEK * u[4] / (Km_aMEK + u[4]))
    
    # ERK equation
    du[5] = (k_ERK * u[4] * (1-u[5]) / (Km_ERK + (1-u[5])) -
             kd_ERK * P_ERK * u[5] / (Km_aERK + u[5]))
    
    # NFB equation
    du[6] = (k_NFB * u[5] * ((1-u[6]) / (Km_NFB + (1-u[6]))) *
             (u[1]^2 / (Km_aR^2 + u[1]^2)) - 
             kd_NFB * P_NFB * u[6] / (Km_aNFB + u[6]))
    
    # PFB equation
    du[7] = (k_PFB * u[5] * u[1] * (1-u[7]) / (Km_PFB + (1-u[7])) - 
             kd_PFB * P_PFB * u[7] / (Km_aPFB + u[7]))

end


# Define the experimental parameter for the ODE problem
tspan = (0.0, 100.0)
u0 = repeat([0], 7)
p_fix = [0.5, 0.5, 25,
         40, 3.75, 1, 1, 1,
         10, 0, 3.75, 1, 1, 0.05, 0.01, 1,
         2, 0.5, 1, 1, 1, 
         2, 0.25, 1, 0.1, 1,
         0.0286, 0.0057, 0.01, 0.85, 0.5, 1,
         0.1, 0.005, 0.1, 0.1, 1]

# Set specific parameters to current case
p_fix[3] = GF_concentration
p_fix[10] = GF_PFB




##### Simulate data with noise #####

# Define and solve ODE problem for specific case
prob = ModelingToolkit.ODEProblem(erk_dynamics!, u0, tspan, p_fix)
X = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.125)

# Add noise in terms of the mean
x_ERK = Array(X)[5,begin:10:end]
time = X.t[begin:10:end]

x̄_ERK = mean(x_ERK, dims = 1)
noise_magnitude = 5e-3
xₙ_ERK = abs.(x_ERK .+ (noise_magnitude * x̄_ERK) .* randn(rng, eltype(x_ERK), size(x_ERK)))

plot(X, alpha = 0.75, color = :blue, 
    label = "ERK Ground truth", idxs=5, title="ERK simulated data after $(GF) stimulation ($(GF_concentration)ng/ml)")
scatter!(time, xₙ_ERK, color = :blue, label = "ERK Noisy Data", idxs=4)




##### Solve UDE #####

rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward
const U = Lux.Chain(Lux.Dense(7, 25, rbf), Lux.Dense(25, 25, rbf), Lux.Dense(25, 25, rbf), Lux.Dense(25, 25, rbf), Lux.Dense(25, 1))

# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
const _st = st

# Define the hybrid model
function ode_discovery!(du, u, p, t, p_true)

	# Estimate ODE solution with NN
	û = U(u, p, _st)[1] 

	# Retrieve known parameters
	k_R, kd_R, GF = p_true[1:3]
	k_Ras, kd_Ras, Km_Ras, Km_aRas, GAP = p_true[4:8] 
	k1_Raf, k2_Raf, kd_Raf, Km_Raf, Km_aRaf, K_NFB, K_PFB, P_Raf = p_true[9:16]
	k_MEK, kd_MEK, Km_MEK, Km_aMEK, P_MEK = p_true[17:21]
	k_ERK, kd_ERK, Km_ERK, Km_aERK, P_ERK = p_true[22:26]
	k_NFB, kd_NFB, Km_NFB, Km_aR, Km_aNFB, P_NFB = p_true[27:32]
	k_PFB, kd_PFB, Km_PFB, Km_aPFB, P_PFB = p_true[33:37]

	# Define ODE system with unknown part
	# Receptor equation
    du[1] = k_R * (1-u[1]) * GF * f_signal(t) - kd_R * u[1]

	# Ras equation
    du[2] = (k_Ras * u[1] * (1-u[2]) / (Km_Ras + (1-u[2])) - 
	         kd_Ras * GAP * u[2] / (Km_aRas + u[2]))

	# Raf equation combined with NN function
	du[3] = (k1_Raf * u[2] * ((1-u[3]) / (Km_Raf + (1-u[3]))) * 
	        ((K_NFB)^2 / (K_NFB^2 + u[6]^2)) - 
	         kd_Raf * P_Raf * u[3] / (Km_aRaf + u[3]) + û[1])
	         
			 #k2_Raf * u[7] * (1-u[3]) / (K_PFB + (1-u[3])))

	# MEK equation
	du[4] = (k_MEK * u[3] * (1-u[4]) / (Km_MEK + (1-u[4])) - 
	         kd_MEK * P_MEK * u[4] / (Km_aMEK + u[4]))
			 
	# ERK equation
	du[5] = (k_ERK * u[4] * (1-u[5]) / (Km_ERK + (1-u[5])) -
	         kd_ERK * P_ERK * u[5] / (Km_aERK + u[5]))

	# NFB equation
	du[6] = (k_NFB * u[5] * ((1-u[6]) / (Km_NFB + (1-u[6]))) *
	         (u[1]^2 / (Km_aR^2 + u[1]^2)) - 
	         kd_NFB * P_NFB * u[6] / (Km_aNFB + u[6]))

    # PFB equation
	du[7] = (k_PFB * u[5] * u[1] * (1-u[7]) / (Km_PFB + (1-u[7])) - 
	         kd_PFB * P_PFB * u[7] / (Km_aPFB + u[7]))
end


# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ode_discovery!(du, u, p, t, p_fix)

# Define the problem
prob_nn = ModelingToolkit.ODEProblem(nn_dynamics!, u0, tspan, p)



#--------------------------------------------------------
# Optimisation-specific functions
function predict(θ, T=time)
    _prob = ModelingToolkit.remake(prob_nn, p = θ)
    Array(OrdinaryDiffEq.solve(_prob, OrdinaryDiffEq.Vern7(), saveat = T,
                abstol = 1e-10, reltol = 1e-10,
                sensealg=SciMLSensitivity.QuadratureAdjoint(autojacvec=SciMLSensitivity.ReverseDiffVJP(true))))
end

function loss(θ)
    X̂ = predict(θ)
    if length(X̂[5,:]) == length(xₙ_ERK)
    	mean(abs2, xₙ_ERK .- X̂[5,:])
	else
		1000
	end
end

losses = Float64[]
callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end
#----------------------------------------------------------

# Define the optimisation problem (auto-differentiation method to compute gradient)
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob1 = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

# Empty the loss array
if !isempty(losses)
    empty!(losses)
end

# Solve UDE first by running Adam optimizer
res1 = Optimization.solve(optprob1, OptimizationOptimisers.Adam(), callback = callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

# Finish solving UDE by running LBFGS optimizer
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking()), callback = callback, maxiters = 2000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")



##### Visualise results #####

# Retrieve the trained parameters and get EGF/NGF model estimations
p_trained = res2.u
ts = first(X.t):(mean(diff(X.t))):last(X.t)
X̂ = predict(p_trained, ts)

# Trained on noisy data vs real solution
data_plot = plot(ts, X̂[5,:], xlabel = "t", ylabel = "x(t), y(t)", color = :red, label = "ERK UDE Approximation")
scatter!(data_plot, time, xₙ_ERK, color = :black, label = "ERK Noisy data")
#savefig(data_plot, "./Plots/$(filename)_data_plot.svg")

# Compare unknown part approximated by NN with ground truth
û = U(X̂, p_trained, _st)[1]

# Establish ground truth for unkown part of ODE system
u = (p_fix[10] .* X[7,:] .* (1 .- X[3,:]) ./ (p_fix[15] .+ (1 .- X[3,:])))

# Plot the simulated data, trained solution and ground truth ERK dynamics
p1 = plot(X, color = :skyblue, linewidth=8, label = "Ground truth ERK", idxs=5)
plot!(p1, ts, X̂[5,:], xlabel = "Time [min]", ylabel = "x(t)", left_margin=(7,:mm), color = :black, linewidth=2, label = "ERK Approximation (UDE)", title="Fitted ERK dynamics after $(GF) stimulation")
scatter!(p1, time, xₙ_ERK, color = :darkorange, alpha=0.75, label = "Noisy ERK Data")

# Plot unknown part approximated by NN with ground truth
p2 = plot(ts, û[1,:], label="NN Unknown", linewidth=2, title="Unknown function approximated by NN", xlabel = "Time [min]", ylabel = "x(t)", left_margin=(7,:mm),)
plot!(p2, X.t, u, linewidth=2, label="Ground truth", legend_position=:bottomright)

# Final two panels plot
nn_plot = plot(p1, p2, layout=(2,1), size=(600, 800))
#savefig(nn_plot, "./Plots/$(filename)_nn_plot.png")


# Plot UDE full results for all model species
R_plot = plot(ts, X̂[1,:], label="Predicted", alpha = 0.25, color = :blue, linewidth=2, title="\nR", titlelocation=:left)
plot!(X.t, X[1,:], label="Ground truth", linestyle=:dash, colour=:purple, linewidth=2, ylabel = "x(t)", left_margin=(7,:mm))

Ras_plot = plot(ts, X̂[2,:], label="Predicted", colour=:lightgreen, linewidth=2, title="\nRas", titlelocation=:left)
plot!(X.t, X[2,:], label="Ground truth", linestyle=:dash, colour=:green, linewidth=2)

Raf_plot = plot(ts, X̂[3,:], label="Predicted", colour=:steelblue, alpha=0.75, linewidth=2, title="Raf", titlelocation=:left)
plot!(X.t, X[3,:], label="Ground truth", linestyle=:dash, colour=:darkblue, linewidth=2, ylabel = "x(t)", left_margin=(7,:mm))

MEK_plot = plot(ts, X̂[4,:], label="Predicted", colour=:yellow3, linewidth=2, title="MEK", titlelocation=:left)
plot!(X.t, X[4,:], label="Ground truth", linestyle=:dash, colour=:orange, linewidth=2)

ERK_plot = plot(ts, X̂[5,:], label="Predicted", colour=:blue, linewidth=2, title="ERK", titlelocation=:left)
plot!(X.t, X[5,:], label="Ground truth", linestyle=:dash, colour=:darkblue, linewidth=2)

NFB_plot = plot(ts, X̂[6,:], label="Predicted", colour=:lightpink, linewidth=2, title="NFB", titlelocation=:left)
plot!(X.t, X[6,:], label="Ground truth", linestyle=:dash, colour=:red, linewidth=2, xlabel = "Time [min]", ylabel = "x(t)", left_margin=(7,:mm), legend_position=:bottomright)

PFB_plot = plot(ts, X̂[7,:], label="Predicted", colour=:grey, linewidth=2, title="PFB", titlelocation=:left)
plot!(X.t, X[7,:], label="Ground truth", colour=:black, linewidth=2, linestyle=:dash, legend_position=:bottomright, xlabel = "Time [min]")

# Final multipanel plot
full_plot = plot(R_plot, Ras_plot, Raf_plot, MEK_plot, NFB_plot, PFB_plot, layout=(3,2), size=(800, 1000))

#savefig(full_plot, "./Plots/$(filename)_full_plot.svg")



##### Save results #####

# Save NN approximation with the fitted and GT ODE solution for equation discovery
erk_data = zeros(length(û[1,:]))
time_data = zeros(length(û[1,:]))
erk_data[begin:length(xₙ_ERK)] = xₙ_ERK
time_data[begin:length(time)] = time
df = DataFrame((
    time=ts,
    t_data=time_data,
    ERK_data=erk_data,
    NN_approx=û[1,:],
    NN_GT=u,
    R_fit=X̂[1,:], 
    Ras_fit=X̂[2,:],
    Raf_fit=X̂[3,:],
    MEK_fit=X̂[4,:],
    ERK_fit=X̂[5,:],
    NFB_fit=X̂[6,:],
    PFB_fit=X̂[7,:],
    R_GT=X[1,:],
    Ras_GT=X[2,:],
    Raf_GT=X[3,:],
    MEK_GT=X[4,:],
    ERK_GT=X[5,:],
    NFB_GT=X[6,:],
    PFB_GT=X[7,:]))
CSV.write("./Data/$(filename).csv", df, header=true)


