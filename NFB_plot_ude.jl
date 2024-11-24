### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 607bbaa4-42a0-11ef-3e76-451221ce0bf2
begin
	# Activate project environment
	import Pkg
	Pkg.activate(".")
	
	using DataFrames, Plots, CSV, JLD2, ColorSchemes

	gr()
end

# ╔═╡ 029eed0b-e3e8-4a53-8fdf-9aa8456a9537
md"""
##### Environment Set-up 
"""

# ╔═╡ 9f679d88-6de8-4cf4-a79f-3ddfcc68a840
md"""
# NFB plotting
This notebook is designed to plot the results of the UDE approximation for the NFB model.
"""

# ╔═╡ 36e2585e-d50d-4353-a7c6-6a7a40c58f08
md"""
#### Single case
"""

# ╔═╡ c1bc713c-446d-445f-9034-8ed0910c5e49
# Retrieve UDE data from file
begin
	filename = "nfb_a_005"
	df = CSV.read("./Data/$(filename).csv", DataFrame)
end

# ╔═╡ 9cc300fa-2810-4d77-a632-d25ae4c01e59
begin 
	# Plot the simulated data, trained solution and ground truth g2p dynamics
	simulated_data_plot = plot(df.time, df.g2p_GT, color = :skyblue, linewidth=8, label = "Ground truth g2p")
	plot!(df.time, df.g2p_fit, xlabel = "Time [min]", ylabel = "x(t)", left_margin=(7,:mm), linewidth=2, color = :black, label = "g2p Approximation (UDE)", title="Fitted g2p dynamics")
	#scatter!(df.t_data, df.g2p_data, label = "Noisy g2p Data", color = :darkorange, alpha=0.75)
	
	# Plot unknown part approximated by NN with ground truth
	nn_plot = plot(df.time, [df.NN1, df.NN2], label=["NN1" "NN2"], title="NN-approximated unknown function", xlabel = "Time [min]", ylabel = "x(t)", left_margin=(5,:mm), linewidth=2)#, ylim=(0, 0.1))

	if occursin("no_fb", filename)
		plot!(nn_plot, df.time, repeat([1], length(df.time)), label="U₁ and U₂ Ground truth", colour=:black, linestyle=:dash)
	elseif occursin("_a_", filename)
		plot!(nn_plot, df.time, df.Xact_fit, label="U₁ Ground truth", colour=:blue, linestyle=:dash)
		plot!(nn_plot, df.time, repeat([1], length(df.time)), label="U₂ Ground truth", colour=:darkorange, linestyle=:dash)
	elseif occursin("_b_", filename)
		plot!(nn_plot, df.time, repeat([1], length(df.time)), label="U₁ Ground truth", colour=:blue, linestyle=:dash)
		plot!(nn_plot, df.time, df.Xact_fit, label="U₂ Ground truth", colour=:darkorange, linestyle=:dash)
	elseif occursin("_ab_", filename)
		plot!(nn_plot, df.time, df.Xact_fit, label="U₁ and U₂ Ground truth", colour=:black, linestyle=:dash)
	end

	# fitted_ERK_plot,
	plot(simulated_data_plot, nn_plot, layout=(2,1), size=(600, 800), plot_title="Case ab")#, ylim=(0.9, 1.1))

	#savefig("./Plots/$(filename)_nn_unknown.svg")
end

# ╔═╡ bfe1b3e0-0f02-416b-981a-35b82b8269d3
begin
	# Plot UDE full results for all model species
	g1p_plot = plot(df.time, df.g1p_fit, label="Predicted", alpha = 0.25, color = :blue, linewidth=2, title="\ng1p", titlelocation=:left)
	plot!(df.time, df.g1p_GT, label="Ground truth", linestyle=:dash, colour=:purple, linewidth=2, ylabel = "x(t)", left_margin=(7,:mm))
	
	g2p_plot = plot(df.time, df.g2p_fit, label="Predicted", colour=:lightgreen, linewidth=2, title="\ng2p", titlelocation=:left)
	plot!(df.time, df.g2p_GT, label="Ground truth", linestyle=:dash, colour=:green, linewidth=2, ylabel = "x(t)", left_margin=(7,:mm))
	
	Xact_plot = plot(df.time, df.Xact_fit, label="Predicted", colour=:steelblue, alpha=0.75, linewidth=2, title="Xact", titlelocation=:left)
	plot!(df.time, df.Xact_GT, label="Ground truth", linestyle=:dash, colour=:darkblue, linewidth=2, ylabel = "x(t)", left_margin=(7,:mm))

	plot(g1p_plot, g2p_plot, Xact_plot, layout=(3,1), size=(600, 800), plot_title="Case ab")

	#savefig("./Plots/$(filename)_full_model.svg")
end

# ╔═╡ c39dacc0-49c6-44be-991b-8a5ee56f7473
md"""
#### Multiple cases
"""

# ╔═╡ 5d520514-2310-49c1-a58f-b37d97733654
# Retrieve UDE data from file
begin
	filename_a = "nfb_a_005"
	filename_b = "nfb_b_005"
	filename_ab = "nfb_ab_005"
	filename_nofb = "nfb_nofb_005"
	df_a = CSV.read("./Data/$(filename_a).csv", DataFrame)
	df_b = CSV.read("./Data/$(filename_b).csv", DataFrame)
	df_ab = CSV.read("./Data/$(filename_ab).csv", DataFrame)
	df_nofb = CSV.read("./Data/$(filename_nofb).csv", DataFrame)
end

# ╔═╡ a871c376-074a-4a73-9ad8-9e642302e1af
begin
	# Plot the simulated data, trained solution and ground truth g2p dynamics
	simulated_data_plot_all = plot(df.time, [df_a.g2p_GT, df_b.g2p_GT, df_ab.g2p_GT, df_nofb.g2p_GT], color = :skyblue, linewidth=8, label = ["Ground truth g2p" "" "" ""], size=(500, 300))
	plot!(df.time, [df_a.g2p_fit, df_b.g2p_fit, df_ab.g2p_fit, df_nofb.g2p_fit], xlabel = "Time [min]", ylabel = "x(t)", left_margin=(7,:mm), linewidth=2, color = :black, label = ["g2p Approximation (UDE)" "" "" ""], title="Simulated g2p data")
	scatter!(df.t_data, [df_a.g2p_data, df_b.g2p_data, df_ab.g2p_data, df_nofb.g2p_data], label = ["Noisy g2p Data" "" "" ""], color = :darkorange, alpha=0.75, markersize=3)
	#savefig("./Plots/nfb_simulated_data_005.svg")
end

# ╔═╡ a7ab7ea7-5dab-45d7-86a7-3a1a92f44c34
begin
	# Plot UDE full results for all model species
	a_plot_all = plot(df_a.time, 
		[df_a.g1p_fit, df_a.g2p_fit, df_a.Xact_fit],  
		label = "", 
		color = [:blue :purple :green], alpha = 0.25,
		linewidth=2, 
		title="Case a", titlelocation=:left, 
		top_margin=(3,:mm), bottom_margin=(3,:mm))
	plot!(df.time, 
		[df_a.g1p_GT, df_a.g2p_GT, df_a.Xact_GT], 
		label="",  
		colour=[:blue :purple :green], alpha = 0.5,
		linestyle=:dash, linewidth=2, 
		ylabel = "x(t)", 
		left_margin=(7,:mm))
	
	b_plot_all = plot(df_b.time, 
		[df_b.g1p_fit, df_b.g2p_fit, df_b.Xact_fit], 
		label="", 
		color = [:blue :purple :green], alpha = 0.25,
		linewidth=2, 
		title="Case b", titlelocation=:left)
	plot!(df.time, 
		[df_b.g1p_GT, df_b.g2p_GT, df_b.Xact_GT], 
		label="", 
		linestyle=:dash, linewidth=2,
		colour=[:blue :purple :green], alpha = 0.5,  
		ylabel = "x(t)", 
		left_margin=(7,:mm))
	
	ab_plot_all = plot(df_ab.time, 
		[df_ab.g1p_fit, df_ab.g2p_fit, df_ab.Xact_fit], 
		label="", color = [:blue :purple :green], alpha = 0.25,
		linewidth=2,
		title="Case ab", titlelocation=:left, 
		top_margin=(3,:mm))
	plot!(df.time, 
		[df_ab.g1p_GT, df_ab.g2p_GT, df_ab.Xact_GT], 
		label="", 
		linestyle=:dash, linewidth=2, 
		colour=[:blue :purple :green], alpha = 0.5,  
		ylabel = "x(t)", 
		left_margin=(7,:mm))

	nofb_plot_all = plot(df_nofb.time, 
		[df_nofb.g1p_fit, df_nofb.g2p_fit, df_nofb.Xact_fit], 
		label="", 
		color = [:blue :purple :green], alpha = 0.25, 
		linewidth=2, 
		title="Case no FB", titlelocation=:left)
	plot!(df.time,
		[df_nofb.g1p_GT, df_nofb.g2p_GT, df_nofb.Xact_GT], 
		label="", 
		linestyle=:dash, linewidth=2, 
		colour=[:blue :purple :green], alpha = 0.5,  
		ylabel = "x(t)", 
		left_margin=(7,:mm))

	# Create an invisible plot for the legend
	legend_plot = plot((1:6)', 
		label=["Predicted g1p" "Predicted g2p" "Predicted Xact" "GT g1p" "GT g2p" "GT Xact"], 
		colour=[:blue :purple :green :blue :purple :green], 
		linestyle=[:solid :solid :solid :dash :dash :dash],
        legend=:outerbottom, 
		framestyle=:none,
		grid=false, 
		showaxis=false, 
		background_color=:transparent,
		legend_column=3)

	plot(a_plot_all, b_plot_all, ab_plot_all, nofb_plot_all, legend_plot, layout=@layout([[A B; C D]; E{0.12h}]), size=(800, 500), plot_title="NFB UDE solution (0.05M input)", plot_titlevspan=0.1, titlefontsize=12)

	#savefig("./Plots/nfb_ude_solution_005.svg")
end

# ╔═╡ 654fb681-32ba-4610-a544-5cf47071e1fc
function add_GT!(nn_plot, case, df)
	if case == "nofb"
		plot!(nn_plot, df.time, repeat([1], length(df.time)), label="", colour=:black, linestyle=:dash)
	elseif case == "a"
		plot!(nn_plot, df.time, df.Xact_fit, label="", colour=:blue, linestyle=:dash)
		plot!(nn_plot, df.time, repeat([1], length(df.time)), label="", colour=:darkorange, linestyle=:dash)
	elseif case == "b"
		plot!(nn_plot, df.time, repeat([1], length(df.time)), label="", colour=:blue, linestyle=:dash)
		plot!(nn_plot, df.time, df.Xact_fit, label="", colour=:darkorange, linestyle=:dash)
	elseif case == "ab"
		plot!(nn_plot, df.time, df.Xact_fit, label="", colour=:black, linestyle=:dash)
	end
end

# ╔═╡ 8bc58089-82c2-4d51-8c86-d57e694b7887
begin 
	# Plot unknown part approximated by NN with ground truth
	a_nn_plot = plot(df.time, 
		[df_a.NN1, df_a.NN2], 
		label="", 
		title="Case a",
		xlabel = "Time [min]", ylabel = "x(t)",
		linewidth=2,
		top_margin=(3, :mm), bottom_margin=(3,:mm), left_margin=(5,:mm))

	add_GT!(a_nn_plot, "a", df_a)

	# Plot unknown part approximated by NN with ground truth
	b_nn_plot = plot(df.time, 
		[df_b.NN1, df_b.NN2], 
		label="", 
		title="Case b", 
		xlabel = "Time [min]", ylabel = "x(t)",
		linewidth=2, left_margin=(5,:mm))

	add_GT!(b_nn_plot, "b", df_b)

	# Plot unknown part approximated by NN with ground truth
	ab_nn_plot = plot(df.time, 
		[df_ab.NN1, df_ab.NN2], 
		label="", 
		title="Case ab", 
		xlabel = "Time [min]", ylabel = "x(t)", 
		left_margin=(5,:mm), 
		linewidth=2)

	add_GT!(ab_nn_plot, "ab", df_ab)

	# Plot unknown part approximated by NN with ground truth
	nofb_nn_plot = plot(df.time, 
		[df_nofb.NN1, df_nofb.NN2], 
		label="", 
		title="Case no FB", 
		xlabel = "Time [min]", ylabel = "x(t)", ylim=(0.99, 1.01),
		left_margin=(5, :mm), 
		linewidth=2)

	add_GT!(nofb_nn_plot, "nofb", df_nofb)

	# Create an invisible plot for the legend
	legend_nn_plot = plot((1:3)', 
		label=["NN1" "NN2" "GT"], 
		colour=[:blue :orange :black], 
		linestyle=[:solid :solid :dash],
        legend=:outerbottom, 
		framestyle=:none,
		grid=false, 
		showaxis=false, 
		background_color=:transparent,
		legend_column=3)

	plot(a_nn_plot, b_nn_plot, ab_nn_plot, nofb_nn_plot, legend_nn_plot,
		layout=@layout([[A B; C D]; E{0.12h}]), 
		size=(800, 500), 
		plot_title="NFB Neural Network output (0.05M input)", 
		plot_titlevspan=0.1, 
		titlefontsize=12, titlelocation=:left)
	#savefig("./Plots/NFB_nn_005.svg")
end

# ╔═╡ f69131ad-7c41-4d49-96ce-9e4a8301a14d
md"""
#### Pulsatile Prediction
"""

# ╔═╡ 9c1fe74d-fed2-4a5c-9352-f4aec1425ee2
# Retrieve UDE data from file
begin
	filename_pred_a_10 = "nfb_a_005_10m_10v_pred"
	filename_pred_b_10 = "nfb_b_005_10m_10v_pred"
	filename_pred_ab_10 = "nfb_ab_005_10m_10v_pred"
	filename_pred_nofb_10 = "nfb_nofb_005_10m_10v_pred"
	df_a_pred_10 = CSV.read("./Data/$(filename_pred_a_10).csv", DataFrame)
	df_b_pred_10 = CSV.read("./Data/$(filename_pred_b_10).csv", DataFrame)
	df_ab_pred_10 = CSV.read("./Data/$(filename_pred_ab_10).csv", DataFrame)
	df_nofb_pred_10 = CSV.read("./Data/$(filename_pred_nofb_10).csv", DataFrame)
	filename_pred_a_3 = "nfb_a_005_3m_3v_pred"
	filename_pred_b_3 = "nfb_b_005_3m_3v_pred"
	filename_pred_ab_3 = "nfb_ab_005_3m_3v_pred"
	filename_pred_nofb_3 = "nfb_nofb_005_3m_3v_pred"
	df_a_pred_3 = CSV.read("./Data/$(filename_pred_a_3).csv", DataFrame)
	df_b_pred_3 = CSV.read("./Data/$(filename_pred_b_3).csv", DataFrame)
	df_ab_pred_3 = CSV.read("./Data/$(filename_pred_ab_3).csv", DataFrame)
	df_nofb_pred_3 = CSV.read("./Data/$(filename_pred_nofb_3).csv", DataFrame)
end

# ╔═╡ 57259924-4d0e-4598-b8a4-28cc6c73a5b7
begin 
	# Plot unknown part approximated by NN with ground truth
	a_pred_nn_plot = plot(df.time, 
		[df_a_pred_3.NN1, df_a_pred_3.NN2], 
		label="", 
		title="Case a",
		xlabel = "Time [min]", ylabel = "x(t)",
		linewidth=3, left_margin=(5,:mm))

	add_GT!(a_pred_nn_plot, "a", df_a_pred_3)

	# Plot unknown part approximated by NN with ground truth
	b_pred_nn_plot = plot(df.time, 
		[df_b_pred_3.NN1, df_b_pred_3.NN2], 
		label="", 
		title="Case b", 
		xlabel = "Time [min]", ylabel = "x(t)",
		linewidth=3, left_margin=(5,:mm))

	add_GT!(b_pred_nn_plot, "b", df_b_pred_3)

	# Plot unknown part approximated by NN with ground truth
	ab_pred_nn_plot = plot(df.time, 
		[df_ab_pred_3.NN1, df_ab_pred_3.NN2], 
		label="", 
		title="Case ab", 
		xlabel = "Time [min]", ylabel = "x(t)", 
		left_margin=(5,:mm), 
		linewidth=3)

	add_GT!(ab_pred_nn_plot, "ab", df_ab_pred_3)

	# Plot unknown part approximated by NN with ground truth
	nofb_pred_nn_plot = plot(df.time, 
		[df_nofb_pred_3.NN1, df_nofb_pred_3.NN2], 
		label="", 
		title="Case no FB", 
		xlabel = "Time [min]", ylabel = "x(t)", ylim=(0.95, 1.05),
		left_margin=(5,:mm), 
		linewidth=2)

	add_GT!(nofb_pred_nn_plot, "nofb", df_nofb_pred_3)

	plot(a_pred_nn_plot, b_pred_nn_plot, ab_pred_nn_plot, nofb_pred_nn_plot, legend_nn_plot,
		layout=@layout([A; B; C; D; E{0.12h}]), 
		size=(800, 1200), 
		plot_title="Neural Network Output\n after 3'/3' pulse (0.05M)", 
		plot_titlevspan=0.1, 
		titlefontsize=12, titlelocation=:left)
	#savefig("./Plots/nfb_005_3m_3v_nn_pred_.svg")
end

# ╔═╡ 632d83b0-b807-443f-85ab-c8af2c2c00aa
begin 
	# Plot unknown part approximated by NN with ground truth
	a_pred_plot = plot(df_a.time, 
		[df_a_pred_3.g1p_fit, df_a_pred_3.g2p_fit, df_a_pred_3.Xact_fit],  
		label = "", 
		color = [:blue :purple :green], alpha = 0.25,
		linewidth=3, 
		title="Case a", titlelocation=:left)
	plot!(df.time, 
		[df_a_pred_3.g1p_GT, df_a_pred_3.g2p_GT, df_a_pred_3.Xact_GT], 
		label="",  
		colour=[:blue :purple :green], alpha = 0.5,
		linestyle=:dash, linewidth=2, 
		ylabel = "x(t)", 
		left_margin=(7,:mm))

	b_pred_plot = plot(df_b_pred_3.time, 
		[df_b_pred_3.g1p_fit, df_b_pred_3.g2p_fit, df_b_pred_3.Xact_fit], 
		label="", 
		color = [:blue :purple :green], alpha = 0.25,
		linewidth=3, 
		title="Case b", titlelocation=:left)
	plot!(df.time, 
		[df_b_pred_3.g1p_GT, df_b_pred_3.g2p_GT, df_b_pred_3.Xact_GT], 
		label="", 
		linestyle=:dash, linewidth=2,
		colour=[:blue :purple :green], alpha = 0.5,  
		ylabel = "x(t)", 
		left_margin=(7,:mm))

	ab_pred_plot = plot(df_ab_pred_3.time, 
		[df_ab_pred_3.g1p_fit, df_ab_pred_3.g2p_fit, df_ab_pred_3.Xact_fit], 
		label="", color = [:blue :purple :green], alpha = 0.25,
		linewidth=3,
		title="Case ab", titlelocation=:left)
	plot!(df.time, 
		[df_ab_pred_3.g1p_GT, df_ab_pred_3.g2p_GT, df_ab_pred_3.Xact_GT], 
		label="", 
		linestyle=:dash, linewidth=2, 
		colour=[:blue :purple :green], alpha = 0.5,  
		ylabel = "x(t)", 
		left_margin=(7,:mm))

	nofb_pred_plot = plot(df_nofb.time, 
		[df_nofb_pred_3.g1p_fit, df_nofb_pred_3.g2p_fit, df_nofb_pred_3.Xact_fit], 
		label="", 
		color = [:blue :purple :green], alpha = 0.25, 
		linewidth=3, 
		title="Case no FB", titlelocation=:left)
	plot!(df.time,
		[df_nofb_pred_3.g1p_GT, df_nofb_pred_3.g2p_GT, df_nofb_pred_3.Xact_GT], 
		label="", 
		linestyle=:dash, linewidth=2, 
		colour=[:blue :purple :green], alpha = 0.5,  
		ylabel = "x(t)", 
		left_margin=(7,:mm))

	plot(a_pred_plot, b_pred_plot, ab_pred_plot, nofb_pred_plot, legend_plot, layout=@layout([A; B; C; D; E{0.12h}]), size=(800, 1200), plot_title="NFB model prediction\nafter 3'/3' pulse (0.05M input)", plot_titlevspan=0.1, titlefontsize=12)

	#savefig("./Plots/nfb_005_3m_3v_pred.svg")
end

# ╔═╡ 507d47a2-3ad7-405e-ac50-a3f7ab3bda2f
md"""
#### Optimisation animation
Dynamic visualisation of the optimisation process.
"""

# ╔═╡ 6cd990df-809e-4075-a26a-f7ee157d661a
anim_data = load("./Data/nfb_anim_data.jld2")["anim_data"]

# ╔═╡ c3365c5a-0d41-41b2-94af-73fd10f1e66d
begin
	nfb_anim = Animation()

	# Select the indices to optimise visualisation
	ind = [1:100:5000; 5000:10:7000]

	ts = 
	
	for (i, t) in enumerate(anim_data.track)
		if i in ind
			
			# Make data plot
			data_plot = plot(anim_data.df.t_data[1:81], t.X̂[2,:], color = :red, label = "g2p Approximation", linewidth=2, title="\ng2p data fitting", ylim=(0, 0.3), titlelocation = :left)
			scatter!(anim_data.df.t_data, anim_data.df.g2p_data, color = :black, label = "g2p Noisy data")
			
			# Make all state variables plot
			ude_plot = plot(anim_data.df.t_data[1:81], t.X̂[1,:], label="Predicted g1p", colour=:green, linewidth=2, title="\nFull model prediction", titlelocation = :left)
			plot!(anim_data.df.time, anim_data.df.g1p_fit, label="Ground truth g1p", linestyle=:dash, colour=:lightgreen, linewidth=2)
			plot!(anim_data.df.t_data[1:81], t.X̂[2,:], label="Predicted g2p", colour=:blue, linewidth=2)
			plot!(anim_data.df.time, anim_data.df.g2p_fit, label="Ground truth g2p", linestyle=:dash, colour=:steelblue, linewidth=2)
			plot!(anim_data.df.t_data[1:81], t.X̂[3,:], label="Predicted Xact", colour=:red, linewidth=2)
			plot!(anim_data.df.time, anim_data.df.Xact_fit, label="Ground truth Xact", colour=:pink, linewidth=2, linestyle=:dash)

			# Make neural network plot
			nn_plot = plot(anim_data.df.t_data[1:81], t.û[1,:], label="Û₁ approximated by NN", colour=:blue, title="\nNN approximation", titlelocation = :left)
			plot!(nn_plot, anim_data.df.t_data[1:81], t.û[2,:], label="Û₂ approximated by NN", colour=:green)
			plot!(nn_plot, anim_data.df.time, anim_data.df.Xact_fit, label="U₁ and U₂ Ground truth", colour=:black, linestyle=:dash)

			# Plot the final panel of plots
			plot(ude_plot, data_plot, nn_plot, layout=(1,3), size=(1800, 600), plot_title="Iteration $(i)", top_margin=(10,:mm), margin=(7, :mm), xlabel = "Time", ylabel = "x(t)",)

			# Save the frame for the animation
			frame(nfb_anim)
		end
	end
end

# ╔═╡ 71bda023-e0be-46bf-bb1f-31f5fac7f78d
mp4(nfb_anim, fps=10) 

# ╔═╡ 66548c5b-adfb-4a7f-87a5-f331bbecec4c
#mp4(nfb_anim, "./Plots/nfb_animation.mp4", fps=10) 

# ╔═╡ Cell order:
# ╟─029eed0b-e3e8-4a53-8fdf-9aa8456a9537
# ╟─607bbaa4-42a0-11ef-3e76-451221ce0bf2
# ╟─9f679d88-6de8-4cf4-a79f-3ddfcc68a840
# ╟─36e2585e-d50d-4353-a7c6-6a7a40c58f08
# ╟─c1bc713c-446d-445f-9034-8ed0910c5e49
# ╟─9cc300fa-2810-4d77-a632-d25ae4c01e59
# ╟─bfe1b3e0-0f02-416b-981a-35b82b8269d3
# ╟─c39dacc0-49c6-44be-991b-8a5ee56f7473
# ╠═5d520514-2310-49c1-a58f-b37d97733654
# ╟─a871c376-074a-4a73-9ad8-9e642302e1af
# ╟─8bc58089-82c2-4d51-8c86-d57e694b7887
# ╟─a7ab7ea7-5dab-45d7-86a7-3a1a92f44c34
# ╟─654fb681-32ba-4610-a544-5cf47071e1fc
# ╟─f69131ad-7c41-4d49-96ce-9e4a8301a14d
# ╟─9c1fe74d-fed2-4a5c-9352-f4aec1425ee2
# ╟─57259924-4d0e-4598-b8a4-28cc6c73a5b7
# ╟─632d83b0-b807-443f-85ab-c8af2c2c00aa
# ╟─507d47a2-3ad7-405e-ac50-a3f7ab3bda2f
# ╠═6cd990df-809e-4075-a26a-f7ee157d661a
# ╟─c3365c5a-0d41-41b2-94af-73fd10f1e66d
# ╠═71bda023-e0be-46bf-bb1f-31f5fac7f78d
# ╠═66548c5b-adfb-4a7f-87a5-f331bbecec4c
