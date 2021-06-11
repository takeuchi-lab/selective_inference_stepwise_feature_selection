# More Powerful and General Selective Inference for Stepwise Feature Selection using the Homotopy Continuation Approach

<div align="center">
    <img src="./thumbnail.svg", width=700>
</div>

This package implements a more powerful and general conditional Selective Inference (SI) approach for stepwise feature selection (SFS) method. The basic idea of SI is to make conditional inferences on the selected hypotheses under the condition that they are selected. The main limitation of the existing methods is the loss of power due to over-conditioning, which is required for computational tractability. In this paper, we develop a more powerful and general conditional SI method for SFS by using homotopy method which enables us to resolve the over-conditioning issue. With the homotopy method, even when the SFS algorithm is extended to more complicated ones, it is still possible to perform conditional SI without losing power.

## Simple Demonstration 
For simple demonstration example without the need of installing any package, please see "ex0_simple_demonstration.html".

## Installation & Requirements

Our package is implemented in [Julia Programming Language](https://julialang.org)     

All the required packages are listed in the file Project.toml.

To automatically install all the required packages, please run the following command line
```
>> julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```


## Reproducibility

Since we have already got the results in advance, all the figures are saved in "./img" folder.

To reproduce the figures, please run the file "ex1_reproduce_plots.ipynb" using Jupyter notebook.

NOTE: to start Jupyter notebook, please run the following command
```
>> julia --project=@. -e "using IJulia; IJulia.notebook(dir=pwd())"
```

To reproduce the results, please see the following instructions.

- For simple demonstration example of computing p-value and confidence interval for each feature selected by forward stepwise feature selection algorithm, please run "ex2_simple_demonstration.ipynb" using Jupyter notebook.


- False Positive Rate comparison (FPR) (Forward SFS) (Figure 1a)
    ```
	>> julia --project=@. ex3_fpr_forward.jl
	``` 

- True Positive Rate comparison (TPR) (Forward SFS) (Figure 1b)
    ```
	>> julia --project=@. ex4_tpr_forward.jl
	``` 

- Length of confidence interval (CI) (Forward SFS) (Figure 1c)
    ```
	>> julia --project=@. ex5_ci_length_forward.jl
	```

- False Positive Rate comparison (FPR) (Forward-Backward SFS) (Figure 4a)
    ```
	>> julia --project=@. ex6_fpr_forward_backward.jl
	``` 

- True Positive Rate comparison (TPR) (Forward-Backward SFS) (Figure 4b)
    ```
	>> julia --project=@. ex7_tpr_forward_backward.jl
	``` 

- Length of confidence interval (CI) (Forward-Backward SFS) (Figure 4c)
    ```
	>> julia --project=@. ex8_ci_length_forward_backward.jl
	```
