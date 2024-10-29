module RBF

	export rbf
	function rbf(x)
		return exp.(-x .^ 2)
	end
end
