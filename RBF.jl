module RBF

export rbf
export mytanh

    function rbf(x)
	    return exp.(-x .^ 2)
    end

    function mytanh(x)
	    return tanh.(x)
    end
end
