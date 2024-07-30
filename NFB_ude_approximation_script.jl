#SBATCH --time=03:00:00
#SBATCH --mem=10G
#SBATCH --job-name=ude
#SBATCH --mail-user=giliane.rochat@students.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --output=/home/grochat/DataDrivenEquationDiscovery/log/output_ude_%j.o
#SBATCH --error=/home/grochat/DataDrivenEquationDiscovery/log/error_ude_%j.e
#SBATCH --partition=all
#SBATCH --gres=gpu:rtx2080ti:1

# SciML tools
import DataDrivenDiffEq, DataDrivenSparse, OrdinaryDiffEq, ModelingToolkit, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Standard libraries
using Statistics, Plots, CSV, DataFrames, ComponentArrays

# External libraries
using Lux, Zygote, StableRNGs

# Set a random seed for reproducibility
rng = StableRNG(1112)

gr()



### USER-DEFINED parameters

# Retrieve file arguments
if length(ARGS) < 2
    error("Error! You need to specify as arguments: \n-input concentration\n-Type of NFB (no/a/b/ab)")
else
    input_CC = ARGS[1]
    type_NFB = lowercase(ARGS[2])
end

# Define which case (noFB, a, b, ab) to run the UDE approximation for
string_CC = replace(string(input_CC), "." => "")
filename = string("NFB_", type_NFB, "_", string_CC)



##### Simulate data with noise #####

# Define negative feedback ODE system function
function NFB!(du, u, p, t)

	# Retrieve known parameters
	v1, v2, v3, v4, v5, v6 = p[1:6]
	k1, k2, k3, k4, k5, k6 = p[7:12]
	α, β, input = p[13:15]

	# Define ODE system
	du[1] = v1 * input * (1-u[1]) / (k1 + (1-u[1])) - 
			(v2 * u[1] / (k2 + u[1])) * (1 + α * (u[3] - 1))
    du[2] = v3 * u[1] * (1-u[2]) / (k3 + (1-u[2])) - 
			(v4 * u[2] / (k4 + u[2])) * (1 + β * (u[3] - 1))
	du[3] = v5 * u[2] * (1-u[3]) / (k5 + (1-u[3])) - 
			(v6 * u[3] / (k6 + u[3]))
end

# Define time span and inital conditions of ODE problem
u0 = repeat([0], 3)
tspan = (0., 100.)

# Define parameters
p_ = [0.5, 5, 5, 0.03, 0.1, 0.1,
          0.1, 0.1, 0.1, 0.1, 1, 10,
          0, 0, input_CC]

# Define and solve ODE problem
prob = ModelingToolkit.ODEProblem(NFB!, u0, tspan, p_)
X = OrdinaryDiffEq.§§>21solve(prob, OrdinaryDiffEq.Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.125)

# Add noise in terms of the mean
x_g2p = Array(X)[2,begin:10:end]
time = X.t[begin:10:end]

x̄_g2p = mean(x_g2p, dims = 1)
noise_magnitude = 5e-3
xₙ_g2p = abs.(x_g2p .+ (noise_magnitude * x̄_g2p) .* randn(rng, eltype(x_g2p), size(x_g2p)))

data_plot = plot(X, alpha = 0.75, color = :blue, 
    label = string(type_NFB, " Ground truth"), idxs=2, title="g2p simulated data")
scatter!(data_plot, time, xₙ_g2p_noFB, color = :blue, label = "Noisy Data", idxs=4)




##### Solve UDE #####

# Define a Multilayer FeedForward with Lux.jl
rbf(x) = exp.(-(x .^ 2))
const U = Lux.Chain(Lux.Dense(3, 25, rbf), Lux.Dense(25, 25, rbf), Lux.Dense(25, 25, rbf), Lux.Dense(25, 25, rbf), Lux.Dense(25, 2))

# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
const _st = st

# Define the hybrid model
function ode_discovery!(du, u, p, t, p_true)

    # Estimate ODE solution with NN
    û = U(u, p, _st)[1] 
    
    # Retrieve known parameters
    v1, v2, v3, v4, v5, v6 = p_true[1:6]
    k1, k2, k3, k4, k5, k6 = p_true[7:12]
    α, β, input = p_true[13:15]
    
    # Define ODE system with unknown part
    du[1] = v1 * input * (1-u[1]) / (k1 + (1-u[1])) - 
            (v2 * u[1] / (k2 + u[1])) * û[1] #(1 + α * (u[3] - 1))
    du[2] = v3 * u[1] * (1-u[2]) / (k3 + (1-u[2])) - 
            (v4 * u[2] / (k4 + u[2])) * û[2] #(1 + β * (u[3] - 1))
    du[3] = v5 * u[2] * (1-u[3]) / (k5 + (1-u[3])) - (v6 * u[3] / (k6 + u[3]))
end

# Closure with the known parameters
nn_NFB!(du, u, p, t) = ode_discovery!(du, u, p, t, p_)

# Define the problem with NN
prob_nn = ModelingToolkit.ODEProblem(nn_NFB!, u0, tspan, p)



#--------------------------------------------------------
# Optimisation-specific functions

function predict(θ, T=time)
    _prob = ModelingToolkit.remake(prob_nn, p = θ)
    Array(OrdinaryDiffEq.solve(_prob, OrdinaryDiffEq.Vern7(), saveat = T,
        abstol = 1e-6, reltol = 1e-6, 
		sensealg=SciMLSensitivity.QuadratureAdjoint(autojacvec=SciMLSensitivity.ReverseDiffVJP(true))))
end

function loss(θ)
    X̂ = predict(θ)
    mean(abs2, xₙ_g2p .- X̂[2,:])
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

# Define
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

# Retrieve the trained parameters and get NFB model estimations
p_trained = res2.u
ts = first(X.t):(mean(diff(X.t))):last(X.t) 
X̂ = predict(p_trained, ts)

# Trained on noisy data vs real solution
pl_trajectory = plot(ts, X̂[2,:], xlabel = "Time", ylabel = "x(t)", color = :red, label = "g2p Approximation", linewidth=2, title="g2p fitting")
scatter!(time, xₙ_g2p, color = :black, label = "g2p Noisy data")

# Compare unknown part approximated by NN with ground truth
û = U(X̂, p_trained, _st)[1]
nn_plot = plot(ts, û[1,:], label="Û₁ approximated by NN", colour=:blue, title="NN approximation")
plot!(nn_plot, ts, û[2,:], label="Û₂ approximated by NN", colour=:green)

if occursin("no", type_NFB)
    plot!(nn_plot, ts, repeat([1], length(ts)), label="U₁ and U₂ Ground truth", colour=:black, linestyle=:dash)
elseif type_NFB == "a"
    plot!(nn_plot, ts, X[3,:], label="U₁ Ground truth", colour=:blue, linestyle=:dash)
    plot!(nn_plot, ts, repeat([1], length(ts)), label="U₂ Ground truth", colour=:green, linestyle=:dash)
elseif type_NFB == "b"
    plot!(nn_plot, ts, repeat([1], length(ts)), label="U₁ Ground truth", colour=:blue, linestyle=:dash)
    plot!(nn_plot, ts, X[3,:], label="U₂ Ground truth", colour=:green, linestyle=:dash)
elseif type_NFB == "ab"
    plot!(nn_plot, ts, X[3,:], label="U₁ and U₂ Ground truth", colour=:black, linestyle=:dash)
end

# Compare predicted state variables with their respective ground truth
plot(ts, X̂[1,:], label="Predicted g1p", colour=:green, linewidth=2, title="Full model prediction")
plot!(X.t, X[1,:], label="Ground truth g1p", linestyle=:dash, colour=:lightgreen, linewidth=2)
plot!(ts, X̂[2,:], label="Predicted g2p", colour=:blue, linewidth=2)
plot!(X.t, X[2,:], label="Ground truth g2p", linestyle=:dash, colour=:steelblue, linewidth=2)
plot!(ts, X̂[3,:], label="Predicted Xact", colour=:red, linewidth=2)
plot!(X.t, X[3,:], label="Ground truth Xact", colour=:pink, linewidth=2, linestyle=:dash)




##### Save results #####

# Save NN approximation with the fitted and GT ODE solution for equation discovery
g2p_data = zeros(length(X[1,:]))
time_data = zeros(length(X[1,:]))
g2p_data[begin:length(xₙ_g2p)] = xₙ_g2p
time_data[begin:length(time)] = time
df = DataFrame((
    time=X.t,
    t_data=time_data,
    g2p_data=g2p_data,
    NN1=û[1,:],
    NN2=û[2,:],
    g1p_fit=X̂[1,:],
    g2p_fit=X̂[2,:],
    Xact_fit=X̂[3,:],
    g1p_GT=X[1,:],
    g2p_GT=X[2,:],
    Xact_GT=X[3,:],
    ))
CSV.write("./Data/$(filename).csv", df, header=true)

