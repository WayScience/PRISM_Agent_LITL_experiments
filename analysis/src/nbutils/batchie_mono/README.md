# Minimal fixed-dose monotherapy screen modelling

This sub-package adapts the [BATCHIE](https://doi.org/10.1038/s41467-024-55287-7) framework to model viability responses in fixed-dose monotherapy drug screens.

## Fixed-effect response model

We want to decompose the response matrix (log fold change in cell count against varied cell line and treatment) using factorization:

$$
y_{ij} \sim \mathcal{N}(\mu_{ij}, \tau^{-1}),
\qquad
\mu_{ij} = \alpha + a_i + b_j + U_i^\top V_j.
$$

Here:

- \($y_{ij}$\) is the observed LFC for cell line \(i\) treated with compound \(j\).
- \($\alpha$\) is the global LFC offset for the screen.
- \($a_i$\) and \($b_j$\) are cell-line and compound-specific offsets.
- \($U_i$\) and \($V_j$\) are learned vectors whose dot product captures cell-line-specific drug responses.
- \($\tau$\) is the observation-noise precision (inverse of variance).


## Why going for Bayesian inference with all the added complexity? 

Fitting this model by maximum likelihood estimation (MLE) would be straightfoward in the sense that it requires no custom solver implementation: the solution can be approximated by minimizing prediction errors through stochastic gradient descent. 

Like the original BATCHIE, our goal extends beyond predicting unmeasured response as point estimates. 
We wanted to know **how uncertain every unseen response prediction is, given the experiments collected so far**. 

Two treatments can have similar predicted responses but very different levels of supporting evidence. 
That distinction matters when deciding which experiments to perform next (if wet lab resource is limited, we would wish to devote them all to physically performing experiments for predictions we are lest confident with).
The Gaussian observation model describes variability around a treatment’s expected response. 
By itself, it does not quantify uncertainty about that expected response when the fitted parameters are treated as fixed.
Bayesian inference is the precise method that answers what we wanted to know about, by asking how observed responses support our best effort modelling of the screen:

$$
p(\theta \mid y)
$$

where \($\theta$\) denotes the model parameters. 

Rather than retaining only one fitted parameter set, we want to approximate the posterior distribution of parameter values compatible with the data. 
Propagating these values through the response model gives a distribution of plausible expected responses for each unmeasured treatment. Adding observation noise gives a predictive distribution for a future measurement.

## Where the implementation complexity comes from

Most of the additional machinery supports Bayesian shrinkage and posterior inference, regularizing for model complexity and drawing parameter sets conditional on other parameters and the observed data.
This requires prior distribution specifications, pipelines for conditional sampling updates, and data structure holding results of posterior draws for later uncertainty estimation. 
