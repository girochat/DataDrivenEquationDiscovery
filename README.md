# Data Driven Equation Discovery for biological signaling networks 

## Project description
This project aims to estimate unknown parts of an ordinary differential equation (ODE) system in the context of biological signaling networks. The project can be divided into two central parts with their own specific methodology:
1. approximating the unknown part(s) of the ODE system using a neural network (UDE approximation)
2. estimating the formula of the resulting unknown function(s) (equation discovery)

For the first part, a UDE model, i.e. an ODE model with unknown part(s) that are modelled by a neural network, is fitted to real or simulated data. The parameters of the neural network are optimised while solving the ODE problem. The approach used is largely inspired from the tutorial ["Automatically Discover Missing Physics by Embedding Machine Learning into Differential Equations"](https://docs.sciml.ai/Overview/dev/showcase/missing_physics/) of the SciML ecosystem. 
For the second part, the function(s) represented by the neural network are fed into a SINDy-based algorithm to estimate the mathematical formula [ref]. The algorithm uses sparse regression to select the optimal terms and their coefficients among a library of simple functions to build the equation formula. 
The purpose is to gain insights about the signaling mechanisms in a data-driven manner while incorporating some prior knowledge about the system. 

## Project Structure

#### Scripts
The workflow of the project is structured into two consecutive parts:

1. Model-specific UDE approximation: the UDE approximation scripts will output
   the neural network estimation given a specified ODE model. In this project, two different biological models
   were considered: the NFB model and the ERK model. In this project only simulated data with added noise was used. The data simulation as well as UDE optimisation results are saved in CSV format.
2. General-use Equation discovery (E-SINDy): the E-SINDy scripts take as input the previously generated UDE results and output the optimal equation formula. The output contains also various statistics about the E-SINDy run and is saved in JDL2 format.

For each case an interactive Pluto notebook and a Julia script are provided, for easier readability on one hand and  automation on the other.

To better analyse previous results, additional Pluto notebooks have been created for plotting.

#### Folders
The `Data/` folder holds the final and intermediate results (CSV and JDL2 files) that were obtained for the two considered ODE systems, i.e. NFB and ERK model while the `Plots/` one contains all the plots.

## Usage

#### Pluto Notebooks  
  [Pluto.jl](https://plutojl.org/) is an interactive notebook environment for Julia. It has a built-in package manager and each notebook
  contains its own environment and the necessary package dependencies. The provided notebooks can thus be run in Pluto without specific
  requirements. Provided that Julia is already installed, Pluto.jl can be installed using Julia's package manager. After calling `Pkg` using the command `]` in the Julia terminal, type:  

    add Pluto 
And back in the Julia terminal, launch Pluto using:   
    
    Pluto.run()
    
#### Julia Scripts  
  ##### 1. Environment set up
  To ensure version compatibility, the provided `Project.toml` and `Manifest.toml` files should be in the same folder as the Julia scripts to be run. From within the project directory, launch Julia terminal using the command:  

    julia --project=.

Then, in the Julia REPL instantiate the project:

    type "]" 
    instantiate

The project environment should then be installed.

##### 2. Run UDE approximation script
The UDE scripts are specific to the ODE model considered and take some file argument.   
To run the NFB model UDE approximation script:

    julia --project=. NFB_ude_approximation_script.jl <input concentration> <NFB model type (no/a/b/ab)>

To run the ERK model UDE approximation script:

    julia --project=. ERK_ude_approximation_script.jl <Type of input concentration (high/low)> <Growth factor type (EGF/NGF)> <pulse duration> <pulse frequency>

[!NOTE] The E-SINDy script and notebook cannot yet be run due to bugs in the DataDrivenDiffEq.jl package. 


##### 3. Run E-SINDy script
Any output obtained with the UDE approximation scripts (CSV format) can be used to run the E-SINDy script. It also takes file arguments:

    julia --project=. esindy_script.jl <model type (NFB/ERK)> <Number of bootstraps> <Coefficient threshold> <Output filename> <List of CSV files>

## Examples

Provided here are HTML versions of the Pluto notebooks. They can be run without any Pluto installation and allow to explore the code and results:


## Dependencies and Key Libraries
This project relies on several important Julia packages for data-driven modeling and differential equations:

- **ModelingToolkit.jl**  
[ModelingToolkit.jl]() is used for symbolic modeling of dynamical systems. It provides a high-performance, composable modeling framework for automatically parallelized scientific computing.

- **Lux.jl**  
[Lux.jl]() is employed for building neural networks. It offers a flexible and extensible deep learning framework with a focus on compatibility with scientific computing packages.

- **DataDrivenDiffEq.jl**  
[DataDrivenDiffEq.jl]() is central to the data-driven modeling approach of this project. It provides tools for discovering governing equations from data, including implementation of Sparse Identification of Nonlinear Dynamics (SINDy) 

- **DataDrivenSparse.jl**  
[DataDrivenSparse.jl]() was used for sparse regression and feature selection for the models. It implements algorithms such as:
    STLSQ
    ADAM
    SR3

- **HyperTuning.jl**  
[HyperTuning.jl]() was used to optimise the hyperparameters of the sparse regression algorithm which play a crucial part to obtain the optimal solution to the problem.

These libraries form the backbone of this data-driven modeling pipeline, enabling to discover and analyze complex dynamical systems from experimental data.


## Conclusion
This project provides a framework for estimating unknown equations in a biological signaling network using data-driven methods. By leveraging the DataDrivenDiffEq.jl package, users can uncover the underlying dynamics of complex systems without needing a predefined model.

## Additional Resources and References

    SINDy papers
    HyperTuning.jl package
    DataDrivenDiffEq.jl Documentation
    SciML Ecosystem
    SciML Tutorial
    models reference

## License
This project is licensed under the terms of the [MIT license](LICENSE.txt).