# Data Driven Equation Discovery for Biological Signalling Networks 

## Project description
This project aims to estimate unknown parts of an ordinary differential equation (ODE) system in the context of biological signalling networks. The project can be divided into two central parts with their own specific methodology:
1. approximating the unknown part(s) of the ODE system using a neural network (UDE approximation). 
2. estimating the formula of the resulting unknown function(s) (equation discovery)

To test the approach, two biological models were used to simulate data and build a UDE system out of their respective ODE system:
- the Negative FeedBack (NFB) model presented in Chapter 13: Parameter Estimation, Sloppiness, and Model Identifiability by D. Daniels, M. Dobrzyński, D. Fey in "Quantitative Biology: Theory, Computational Methods and Examples of Models" (2018).
- the ERK signalling (ERK) model presented by Ryu et al. in

Here are the schematics of the two models:

<p align="center">
<img width="849" alt="models_schematics" src="https://github.com/user-attachments/assets/50b76b27-c814-4a46-8e05-77bd4c95f4fd">
</p>

For the first part, a UDE model, i.e. an ODE model with unknown part(s) that are modelled by a neural network, is fitted to real or simulated data. Here is an example with the original ODE system and updated UDE system of the NFB model introduced above:  
&nbsp;

<p align="center">
<img width="759" alt="nfb_ode_ude" src="https://github.com/user-attachments/assets/de28b5f6-db27-473a-adc4-c8151b23a780">
</p>
&nbsp;

The parameters of the neural network are optimised while solving the ODE problem. The approach used is largely inspired from the tutorial ["Automatically Discover Missing Physics by Embedding Machine Learning into Differential Equations"](https://docs.sciml.ai/Overview/dev/showcase/missing_physics/) of the SciML ecosystem. Here is a schematic of the optimisation process:

<p align="center">
<img width="871" alt="ude_optimisation" src="https://github.com/user-attachments/assets/3699a7bc-8912-4f55-8d38-bc3912bebbc9">
</p>

For the second part, the function(s) represented by the neural network are fed into a SINDy-based algorithm to estimate the mathematical formula [1]. The algorithm uses sparse regression to select the optimal terms and their coefficients ($\Xi$) among a library of simple functions ( $\Theta (X)$ ) to build the equation formula. Follow a schematic of SINDy algorithm where $Y=(Y_1, Y_2..Y_k)$ is the input data for which the equation is to be retrieved:

<p align="center">
<img width="871" alt="ude_optimisation" src="https://github.com/user-attachments/assets/0387759a-9272-4a3f-b983-694194b6fce3">
</p>

The purpose is to gain insights about the signaling mechanisms in a data-driven manner while incorporating some prior knowledge about the system.

## Project Structure

#### Code
The workflow of the project is structured into two consecutive parts:

1. UDE approximation: the UDE approximation scripts will output
   the neural network estimation given a specified UDE model. In this project, UDE systems for the NFB model and the ERK model were used.
   The data simulation as well as UDE optimisation results are saved in CSV format.
3. Equation discovery (E-SINDy): the E-SINDy scripts take as input the previously generated UDE results and output the optimal equation formula.
   The output contains also various statistics about the E-SINDy run and is saved in JDL2 format.

For each case interactive Pluto/Jupyter notebooks and Julia scripts are provided, for easier readability on one hand and automation on the other.

#### Folders
This repository contains three main directories:

- **Code**:  
  Pluto/Jupyter notebooks and Julia scripts.
- **Data**:  
  Final and intermediate results (CSV and JDL2 files) that were obtained for the two considered ODE systems, i.e. NFB and ERK model.
- **Plots**:  
  All plots of the results.

## Installation

#### Install Julia
To run the code in this repository, you need to have Julia installed which can be downloaded at https://julialang.org/downloads/. Please note that Julia Version 1.10.4 was used as referenced in the Manifest.toml file.

#### Clone the Repository

Clone this repository to your local machine.

git clone https://github.com/girochat/DataDrivenEquationDiscovery.git

#### Install Dependencies
For reproducibility, it is recommended to use the directory of the project as a Julia environment: 
1. Go to the directory:  
   `cd /your_path_to_the_repo/DataDrivenEquationDiscovery`
2. Open Julia and open the REPL by pressing ']'  
3. In the REPL, activate the local environment and instantiate the packages:  
   `pkg> activate .`  
   `pkg> instantiate`


## Usage

#### Notebooks  
[Pluto.jl](https://plutojl.org/) is an interactive notebook environment for Julia similar to Jupyter for Python. It provides interactivity for running the UDE approximation and E-SINDy and visualising the results. To open the notebook, follow these steps:  
Start the Pluto notebook in Julia:  

    using Pluto
    Pluto.run()

In the Pluto interface, open the desired notebook file to start exploring the method and visualising the results.

If you prefer using Jupyter Notebook, the Jupyter version of the Pluto notebooks are also provided.
  
#### Julia Scripts  

##### 1. Run UDE approximation script
The UDE scripts are specific to the ODE model considered and take some file argument.   
To run the NFB model UDE approximation script:

    julia --project=. NFB_ude_approximation_script.jl <NFB model type (no/a/b/ab)> <input concentration> <save parameters (y/n)> <load parameters (y/n)>

To run the ERK model UDE approximation script:

    julia --project=. ERK_ude_approximation_script.jl <Growth factor type (EGF/NGF)> <Type of input concentration (high/low)> <pulse duration> <pulse frequency>
    <save parameters (y/n)> <load parameters (y/n)>
    

##### 2. Run E-SINDy script

Any output obtained with the UDE approximation scripts (CSV format) can be used to run the E-SINDy script. It also takes file arguments:

    julia --project=. esindy_script.jl <model type (NFB/ERK)> <Number of bootstraps> <Coefficient threshold> <Output filename> <List of CSV files>

## Examples
In this section is provided an example of the output and plots after running the method on a specific case of the NFB model. 

## Dependencies and Key Libraries
This project relies on several important Julia packages for data-driven modeling and differential equations:

- **ModelingToolkit.jl**  
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) is used for symbolic modeling of dynamical systems. It provides a high-performance, composable modeling framework for automatically parallelized scientific computing.

- **Lux.jl**  
[Lux.jl](https://lux.csail.mit.edu/stable/) is employed for building neural networks. It offers a flexible and extensible deep learning framework with a focus on compatibility with scientific computing packages.

- **DataDrivenDiffEq.jl**  
[DataDrivenDiffEq.jl](https://docs.sciml.ai/DataDrivenDiffEq/stable/) is central to the data-driven modeling approach of this project. It provides tools for discovering governing equations from data, including implementation of Sparse Identification of Nonlinear Dynamics (SINDy) 

- **DataDrivenSparse.jl**  
[DataDrivenSparse.jl](https://docs.sciml.ai/DataDrivenDiffEq/stable/libs/datadrivensparse/sparse_regression/) was used for sparse regression and feature selection for the models. It implements algorithms such as:
    - STLSQ
    - ADAM
    - SR3

- **HyperTuning.jl**  
[HyperTuning.jl](https://jmejia8.github.io/HyperTuning.jl/dev/) was used to optimise the hyperparameters of the sparse regression algorithm which play a crucial part to obtain the optimal solution to the problem.

These libraries form the backbone of this data-driven modeling pipeline, enabling to discover and analyse complex dynamical systems from experimental data.

## Additional Resources and References

[1]   Brunton et al. (2016). "Discovering governing equations from data by sparse identification of nonlinear dynamical systems". [DOI](https://doi.org/10.1098/rspa.2021.0904)"  
[2]   

    models reference

## License
This project is licensed under the terms of the [MIT license](LICENSE.txt).
