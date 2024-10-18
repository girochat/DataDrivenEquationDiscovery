### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ bde4a618-3ae4-4789-8ce3-3e33b5d1b842
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 75c93ab4-5f36-11ef-0003-d3a7c79083e6
begin
	# SciML tools
	import ModelingToolkit, Symbolics
	
	# Standard libraries
	using Statistics, Plots , CSV, DataFrames

	# External libraries
	using StableRNGs, Distributions, ColorSchemes, JLD2, Combinatorics, HyperTuning, SmoothingSplines, ProgressMeter

	# Packages under development
	using DataDrivenDiffEq, DataDrivenSparse
	
	# Set a random seed for reproducibility
	rng = StableRNG(1111)

	gr()
end

# ╔═╡ 5885afff-a5fe-46d1-9bb6-48742c859aae
begin
	# Import utils function
	include("./esindy_utils.jl")
	using .ESINDyModule
	using .ERKModule
	using .NFBModule
end

# ╔═╡ a695765c-6dd4-4ce1-858d-625a259cae75
md"""
## Notebook to load and plot E-SINDy results
"""

# ╔═╡ 2d2ff0e0-533b-49aa-a3ab-0d18339ec3d3
md"""
#### Load the data
"""

# ╔═╡ c43aad21-5e64-46cf-9aad-1bdf816e6494
ngf_res = load("./Data/ngf_esindy_100bt.jld2")["results"]

# ╔═╡ 0b9893e6-ffc7-42ec-816d-82100443d2e4
ngf_res_full = load("./Data/ngf_esindy_100bt_full.jld2")["results"]

# ╔═╡ af1d7e6d-907f-4e19-a5b4-034999cfffba
egf_res = load("./Data/egf_esindy_100bt.jld2")["results"]

# ╔═╡ da5aafa0-00e7-4524-9759-30c2950dad1c
egf_res_full = load("./Data/egf_esindy_100bt_full.jld2")["results"]

# ╔═╡ f68a85e2-0b1a-4aab-9beb-6ff4ea0589a7
a_res = load("./Data/nfb_esindy_100bt_a.jld2")["results"]

# ╔═╡ 7082540b-efd2-4f4d-8f4d-0c30a1a17a24
ab_res = load("./Data/nfb_esindy_100bt_ab.jld2")["results"]

# ╔═╡ c6247e64-0261-45ce-8bf2-fe122edb7c96
md"""
### Plotting functions
"""

# ╔═╡ d7e41530-b632-451d-8258-da821c164611
function plot_data(data; sample_idxs=nothing, title=nothing, ylabel=nothing)

	# Retrieve the number of samples
	n_samples = sum(data.time .== 0)
	size_sample = Int(length(data.time) / n_samples)
	if isnothing(sample_idxs)
		sample_idxs = 1:n_samples
	end
	
	# Plot results
	subplots = []
	palette = colorschemes[:seaborn_colorblind]
	p = plot(title=title, xlabel="Time", ylabel=ylabel, titlelocation=:left)
	
	# Plot each sample separately
	i_color = 1
	for sample in 0:(n_samples-1)
		if (sample+1) in sample_idxs
			i_start = 1 + sample * size_sample
			i_end = i_start + size_sample - 1
							
			scatter!(p, data.time[i_start:5:i_end], data.X[i_start:5:i_end, 3], label=data.labels[sample+1], color=palette[i_color], alpha=0.7, markerstrokewidth=0)
			
			#plot!(p, data.time[i_start:i_end], data.GT[i_start:i_end, eq], label="", linestyle=:dash, color=palette[i_color])
			i_color = i_color + 1
		end
		
	end
	#plot!(p, [], [],  label="GT", color=:black, linestyle=:dash)
	push!(subplots, p)
	return subplots
end

# ╔═╡ 663a67a7-84ac-406c-9d1e-9adcc9f70e20
function plot_nn(data; sample_idxs=nothing, title=nothing)

	# Retrieve the number of samples
	n_samples = sum(data.time .== 0)
	size_sample = Int(length(data.time) / n_samples)
	if isnothing(sample_idxs)
		sample_idxs = 1:n_samples
	end
	
	# Plot results
	n_eqs = size(data.Y, 2)
	subplots = []
	palette = colorschemes[:seaborn_colorblind]
	for eq in 1:n_eqs
		if n_eqs > 1
			p = plot(title="NN$(eq)", xlabel="Time t", ylabel="NN(t)")
		else
			p = plot(title=title, xlabel="Time", ylabel="NN(t)", titlelocation=:left)
		end

		# Plot each sample separately
		i_color = 1
		for sample in 0:(n_samples-1)
			if (sample+1) in sample_idxs
				i_start = 1 + sample * size_sample
				i_end = i_start + size_sample - 1
								
				plot!(p, data.time[i_start:i_end], data.Y[i_start:i_end, eq], label=data.labels[sample+1], color=palette[i_color], linewidth=3)
				
				#plot!(p, data.time[i_start:i_end], data.GT[i_start:i_end, eq], label="", linestyle=:dash, color=palette[i_color])
				i_color = i_color + 1
			end
			
		end
		#plot!(p, [], [],  label="GT", color=:black, linestyle=:dash)
		push!(subplots, p)
	end
	return subplots
end

# ╔═╡ d1346555-4e30-42da-9c0f-743e7c943b63
md"""
### ERK model
"""

# ╔═╡ aa202167-2eb6-4073-8d2d-9d39e1862422
md"""
#### NGF case
"""

# ╔═╡ 220f3ef8-a0af-456b-bc16-99cc6a4e2297
begin
	ngf_plot = ESINDyModule.plot_esindy(ngf_res, sample_idxs=1:8, iqr=true)[1]
	plot(ngf_plot, title="E-SINDy results for ERK model\nafter NGF stimulation\n", size=(800, 600), legend_position=:topright)
	#savefig("./Plots/ngf_esindy_100bt.svg")
end

# ╔═╡ e23ce1e4-d30c-4aa8-bd97-69b946c25100
begin
	data_ngf_plot = plot_data(ngf_res_full.data, sample_idxs=5:8, title="NGF", ylabel="ERK(t)")[1]
	plot(data_ngf_plot, size=(800, 600), plot_title="ERK data", plot_titlefontsize=14, topmargin=(8,:mm))
end

# ╔═╡ 163adcf0-db25-4b53-a6c3-ce0edabee217
begin
	nn_ngf_plot = plot_nn(ngf_res.data, sample_idxs=1:4, title="NGF")[1]
	plot(nn_ngf_plot, size=(800, 600), plot_title="NN approximation for ERK model\nafter NGF stimulation\n", plot_titlefontsize=14, topmargin=(8,:mm))
	#savefig("./Plots/erk_nn_ngf.svg")
end

# ╔═╡ 4f0b3acb-a691-4e62-bb3c-c1da3ae40be3
md"""
#### EGF case
"""

# ╔═╡ 6d2649da-7b89-42f2-96e7-96e53ce93ccb
begin
	egf_plot = ESINDyModule.plot_esindy(egf_res, sample_idxs=1:8, iqr=false)[1]
	egf_plot_lim = plot(egf_plot, title="E-SINDy results for ERK model\nafter EGF stimulation\n", size=(800, 600), legend_position=:topright, ylim=(-0.085, 0.035))
end

# ╔═╡ 7caaffa8-147b-4c66-b1b4-d8aaee351ad5
begin
	data_egf_plot = plot_data(egf_res_full.data, sample_idxs=5:8, title="EGF", ylabel="ERK(t)")[1]
	plot(data_egf_plot, size=(800, 600), plot_title="ERK data", plot_titlefontsize=14)#, topmargin=(8,:mm))
end

# ╔═╡ ecba5d53-b109-4761-b09e-1eb30469c550
begin
	nn_egf_plot = plot_nn(egf_res.data, sample_idxs=1:4, title="EGF",)[1]
	plot(nn_egf_plot, size=(800, 600), plot_title="NN approximation for ERK model\nafter EGF stimulation\n", plot_titlefontsize=14, topmargin=(8,:mm))
	#savefig("./Plots/erk_nn_egf.svg")
end

# ╔═╡ c2362a43-a0dd-4136-9874-eeb1bab3cb98
md"""
##### Panel of NGF and EGF plots
"""

# ╔═╡ 39f21f88-218d-4daf-8d91-9d84378352cf
begin
	data_erk_plot = plot(data_ngf_plot, data_egf_plot, layout=(2, 1), plot_title="ERK data", plot_titlefontsize=14, size=(800, 800), titlefontsize=10 ,labelfontsize=8, leftmargin=(3,:mm))
	#savefig("./Plots/erk_data_plot.svg")
end

# ╔═╡ 60405c43-f1dd-4065-bab2-17e97e2e9b4c
begin
	nn_erk_plot = plot(nn_ngf_plot, nn_egf_plot, layout=(2, 1), plot_title="NN approximation", plot_titlefontsize=14, size=(800, 800), titlefontsize=10 , labelfontsize=8, leftmargin=(3,:mm))
	#savefig("./Plots/erk_nn_plot_reduced.svg")
end

# ╔═╡ 93fb8928-94fb-47a1-99a6-33ac4f5a0685
begin
	sindy_erk_plot = plot(ngf_plot, egf_plot_lim, layout=(2, 1), plot_title="NN approximation", plot_titlefontsize=14, size=(800, 800), titlefontsize=10 , labelfontsize=8, leftmargin=(3,:mm))
	#savefig("./Plots/erk_esindy_plot.svg")
end

# ╔═╡ c652b8cf-fa98-4cba-acd5-cfb7a0fe338b
md"""
### NFB model
"""

# ╔═╡ 08cad548-fe76-481c-bb2d-588e4ef65051
md"""
#### Case a
"""

# ╔═╡ 274cb2d8-29fc-4e4e-9b8d-75899152dd86
begin
	# Plot the results
	a_plots = ESINDyModule.plot_esindy(a_res, sample_idxs=1:2:9, iqr=true)
	a_plot2 = plot(a_plots[2], ylim=(0, 2))
	plot(a_plots[1], a_plot2, layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB")
end

# ╔═╡ a7c71a8c-23ae-4f09-9635-4d59ddcb8b96
#savefig("./Plots/nfb_esindy_a_reduced.svg")

# ╔═╡ 3e71a1ec-8b59-49c8-8775-f4ea418c7030
begin
	nn_a_plots = plot_nn(a_res.data, sample_idxs=1:2:9)
	plot(nn_a_plots[1], nn_a_plots[2], layout=(2,1), size=(800, 800), plot_title="NN approximation")
end

# ╔═╡ e18eda0b-d339-4883-b173-aac497143ccd
#savefig("./Plots/nfb_nn_a.svg")

# ╔═╡ 9f578ca2-ae00-43b7-af4b-02e1cc3c5dec
md"""
#### Case ab
"""

# ╔═╡ f632a13d-b8f9-4686-8b62-6903e5081e99
begin
	# Plot the results
	ab_plots = ESINDyModule.plot_esindy(ab_res, sample_idxs=1:2:9, iqr=false)
	plot(ab_plots[1], ab_plots[2], layout=(2,1), size=(800, 800), plot_title="E-SINDy results for NFB")
end

# ╔═╡ 0ea6c388-7ab7-4543-a110-0bf3ea77e400
#savefig("./Plots/nfb_esindy_ab_iqr.svg")

# ╔═╡ af2812b9-ca0a-45ce-8365-ab2beaf1e679
begin
	nn_ab_plots = plot_nn(ab_res.data, sample_idxs=1:2:9)
	plot(nn_ab_plots[1], nn_ab_plots[2], layout=(2,1), size=(800, 800), plot_title="NN approximation for NFB")
end

# ╔═╡ 369341ea-8057-4c9e-b38a-2b9dee65325b
#savefig("./Plots/NFB_nn_ab.svg")

# ╔═╡ 3f06ac98-ed86-4d78-82e6-e8a69d95ecda
md"""
#### Animation of optimisation
"""

# ╔═╡ 038fdde7-be83-48b6-8ae8-8bbdbac8c6aa
begin
	# Load data for coefficient optimisation plotting
	coef_data=(load("./Data/sindy_coef_data.jld2"))["coef_data"]
	coefs = coef_data.coefs

	# Retrieve NFB basis
	@ModelingToolkit.variables x[1:7] i[1:1]
	@ModelingToolkit.parameters t
	x = collect(x)
	i = collect(i)
	nfb_basis = NFBModule.build_basis(x[1:3], i)

	# Build equation for each set of coefficients
	eq0 = ESINDyModule.build_equations(coefs.coef0, nfb_basis)
	eq1 = ESINDyModule.build_equations(coefs.coef1, nfb_basis)
	eq2 = ESINDyModule.build_equations(coefs.coef2, nfb_basis)
	eq3 = ESINDyModule.build_equations(coefs.coef3, nfb_basis)
	y0 = ESINDyModule.get_yvals(coef_data.data, eq0)[1]
	y1 = ESINDyModule.get_yvals(coef_data.data, eq1)[1]
	y2 = ESINDyModule.get_yvals(coef_data.data, eq2)[1]
	y3 = ESINDyModule.get_yvals(coef_data.data, eq3)[1]

	# Build row/column labels and accessory data for plotting
	matrices = [abs.(coefs.coef0), [abs.(coefs.coef0); abs.(coefs.coef1)], 
		[abs.(coefs.coef0); abs.(coefs.coef1); abs.(coefs.coef2)], 
		[abs.(coefs.coef0); abs.(coefs.coef1); abs.(coefs.coef2); abs.(coefs.coef3)]]
	col_labels = ["1\n", "x₁\n", "x₁²\n", "x₁³\n", "x₂\n", "x₂²\n", "x₂³\n", "x₃\n", "x₃²\n", "x₃³\n"]
	labels = ["iter1" "iter2" "iter3" "iter4"]
	colors = [:blue :green :lightgreen :purple]
	ys = [y0[1:801], y1[1:801], y2[1:801], y3[1:801]]

	sindy_anim = Animation()
	for n in 1:4
		
		# Create the heatmap
		M = matrices[n]
		row_labels = ["Iter. $i" for i in 1:n]
		hm = heatmap(M, aspect_ratio=:equal, colormap=:dense, ylim=(0.5, n + 0.5), clims=(0,1.7), yticks=(1:n, row_labels), xticks=(1:10, col_labels), yflip=true, grid=false, framestyle=:grid, xtickfont=font(12), xmirror=true, colorbar=false, size=(500, 500))
		
		# Add text annotations
		for i in 1:size(M, 1)
		    for j in 1:size(M, 2)
		        annotate!(j, i, text(string(round(M[i,j], digits=2)), :black, :center, 8))
		    end
		end

		# Plot dynamics
		dynamics = plot(coef_data.data.time[1:801], ys[1:n], label=labels[1:1,1:n], linewidth=2, color=colors[1:1,1:n], linestyle=:dash, xlabel="Time", ylabel="ŷ(t)", size=(700, 500), leftmargin=(10,:mm), bottommargin=(5,:mm))
		plot!(dynamics, coef_data.data.time[1:801], coef_data.data.GT[1:801], label="GT", linewidth=2, color=:black)

		plot(hm, dynamics, layout=(1, 2), plot_title="Iteration $(n)", size=(1200, 500))
		frame(sindy_anim)
	end
	#mp4(sindy_anim, "./Plots/sindy_animation.mp4", fps=0.5)
end

# ╔═╡ Cell order:
# ╟─a695765c-6dd4-4ce1-858d-625a259cae75
# ╟─bde4a618-3ae4-4789-8ce3-3e33b5d1b842
# ╟─75c93ab4-5f36-11ef-0003-d3a7c79083e6
# ╟─5885afff-a5fe-46d1-9bb6-48742c859aae
# ╟─2d2ff0e0-533b-49aa-a3ab-0d18339ec3d3
# ╟─c43aad21-5e64-46cf-9aad-1bdf816e6494
# ╟─0b9893e6-ffc7-42ec-816d-82100443d2e4
# ╟─af1d7e6d-907f-4e19-a5b4-034999cfffba
# ╟─da5aafa0-00e7-4524-9759-30c2950dad1c
# ╟─f68a85e2-0b1a-4aab-9beb-6ff4ea0589a7
# ╟─7082540b-efd2-4f4d-8f4d-0c30a1a17a24
# ╟─c6247e64-0261-45ce-8bf2-fe122edb7c96
# ╟─d7e41530-b632-451d-8258-da821c164611
# ╟─663a67a7-84ac-406c-9d1e-9adcc9f70e20
# ╟─d1346555-4e30-42da-9c0f-743e7c943b63
# ╟─aa202167-2eb6-4073-8d2d-9d39e1862422
# ╟─220f3ef8-a0af-456b-bc16-99cc6a4e2297
# ╟─e23ce1e4-d30c-4aa8-bd97-69b946c25100
# ╟─163adcf0-db25-4b53-a6c3-ce0edabee217
# ╟─4f0b3acb-a691-4e62-bb3c-c1da3ae40be3
# ╟─6d2649da-7b89-42f2-96e7-96e53ce93ccb
# ╟─7caaffa8-147b-4c66-b1b4-d8aaee351ad5
# ╟─ecba5d53-b109-4761-b09e-1eb30469c550
# ╟─c2362a43-a0dd-4136-9874-eeb1bab3cb98
# ╟─39f21f88-218d-4daf-8d91-9d84378352cf
# ╠═60405c43-f1dd-4065-bab2-17e97e2e9b4c
# ╟─93fb8928-94fb-47a1-99a6-33ac4f5a0685
# ╟─c652b8cf-fa98-4cba-acd5-cfb7a0fe338b
# ╟─08cad548-fe76-481c-bb2d-588e4ef65051
# ╟─274cb2d8-29fc-4e4e-9b8d-75899152dd86
# ╟─a7c71a8c-23ae-4f09-9635-4d59ddcb8b96
# ╟─3e71a1ec-8b59-49c8-8775-f4ea418c7030
# ╟─e18eda0b-d339-4883-b173-aac497143ccd
# ╟─9f578ca2-ae00-43b7-af4b-02e1cc3c5dec
# ╟─f632a13d-b8f9-4686-8b62-6903e5081e99
# ╟─0ea6c388-7ab7-4543-a110-0bf3ea77e400
# ╟─af2812b9-ca0a-45ce-8365-ab2beaf1e679
# ╟─369341ea-8057-4c9e-b38a-2b9dee65325b
# ╟─3f06ac98-ed86-4d78-82e6-e8a69d95ecda
# ╟─038fdde7-be83-48b6-8ae8-8bbdbac8c6aa
