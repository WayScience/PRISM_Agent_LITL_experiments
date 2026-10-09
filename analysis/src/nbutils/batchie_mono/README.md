# Minimal fixed-dose monotherapy screen modelling

This sub-package adapts the [BATCHIE](https://doi.org/10.1038/s41467-024-55287-7) framework to model viability responses in fixed-dose monotherapy drug screens.

## Simple response model

The model itself is extremely simple, we want to decompose the response matrix (log fold change in cell count against varied cell line and treatment) using factorization:

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

Fitting this model by maximum likelihood estimation (MLE) would be straightforward to implement: with a shared noise variance, estimating the response parameters amounts to minimizing squared prediction errors over observed treatments. A standard optimizer could learn the offsets and embeddings, yielding one fitted parameter set and a point prediction for each unmeasured response.

## Why Bayesian inference?

Like the original BATCHIE, our goal extends beyond predicting some unmeasured response. 
We wanted to know **how uncertain every unseen response prediction is, given the experiments collected so far**. 

Two treatments can have similar predicted responses but very different levels of supporting evidence. 
That distinction matters when deciding which experiments to perform next (if wet lab resource is limited, we would wish to devote them all to physically performing experiments for predictions we are lest confident with).
The Gaussian observation model describes variability around a treatment’s expected response. By itself, it does not quantify uncertainty about that expected response when the fitted parameters are treated as fixed.
Bayesian inference is the precise method that answers what we wanted to know about, by asking how observed responses support our best effort modelling of the screen:

$$
p(\theta \mid y)
$$

where \($\theta$\) denotes the model parameters. 

Rather than retaining only one fitted parameter set, we want to approximate the posterior distribution of parameter values compatible with the data. 
Propagating these values through the response model gives a distribution of plausible expected responses for each unmeasured treatment. Adding observation noise gives a predictive distribution for a future measurement.

## Where the implementation complexity comes from

The response equation is simple; most of the additional machinery supports posterior inference. 
MLE searches for parameter values that maximize the likelihood. 
Our Bayesian implementation instead uses Gibbs sampling to repeatedly draw parameter sets conditional on the current values of the other parameters and the observed data.

This requires prior specifications, conditional sampling updates, noise and shrinkage parameter updates, and management of posterior draws. 
It also requires checking whether the sampler has adequately explored the posterior. 
These draws preserve uncertainty and dependencies among parameters, allowing downstream experiment selection to account for what the model has, and has not, learned.

## Module overview

The modules connect the response model and its uncertainty estimates to an adaptive screening loop:

- **Data interface**
	- [batchie_config.py](batchie_config.py) names the columns describing experiments and responses.
- **Model formulation**
	- [models.py](models.py) defines the common response-model interfaces.
	- [batchie.py](batchie.py) connects the fixed-dose response model to fitting and uncertainty-aware prediction.
	- [model_utils.py](model_utils.py) provides supporting numerical helpers.
- **Posterior sampling**
	- [pymc_backend.py](pymc_backend.py) expresses the Bayesian formulation in PyMC and samples its posterior with NUTS.
	- [posterior.py](posterior.py) holds the sampled parameter sets used to represent uncertainty.
- **Selection objective and policy**
	- [batchie_selector.py](batchie_selector.py) determines which compounds to test next: randomly or by lowest predicted response.
- **Orchestration**
	- [simulation.py](simulation.py) runs the fit–select–reveal loop and tracks discovery of the strongest-response compounds.
