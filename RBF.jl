module RBF

export rbf
#    export mytanh

function rbf(x)
	return Base.exp.(-x .^ 2)
end

    #function mytanh(x)
	 #   return Base.tanh.(x)
    #end
end
