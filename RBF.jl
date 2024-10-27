module RBF

export rbf
export tanh

    function rbf(x)
	    return exp.(-x .^ 2)
    end

    function mytanh(x)
	    return tanh.(x)
    end
end
