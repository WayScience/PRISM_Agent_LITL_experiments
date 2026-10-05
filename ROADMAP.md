# DSPy + LITL Agentic AI Sandbox
## Previous Roadmap

- [x] Scaffold repo structure  
  - [x] Defined the `agentic_system` and `analysis` compartment.
  - [x] Added early stage conda env, pyproject toml and uv lock.
- [x] Define minimal DSPy agent with placeholder tool
  - [x] Define backend for DepMap PRISM IC50 retrieval and task dispatching 
  - [x] Define agentic signature
- [x] Implement demo experiments
  - [x] Demo experiment with subset of PRISM data and GPT5-nano model
  - [x] Demo experiment with subset of PRISM data and Llama-3.1-8B model 
- [ ] Expand tool library with academic use cases  
  - [x] Tool rate limiting
  - [x] Tool call caching
  - [ ] Chembl Query tools
  - [ ] Pubchem Query tools
- [ ] Iterative experiments on hypothetical PRISM-style LITL data  

## New Roadmap
- [ ] Re-scaffold repo
  - [ ] Better way of structuring both analysis and implementation of multiple models (including already implemented agents).
- [ ] Data
  - [x] PRSIM secondary LFC input data download and wrangling 
  - [ ] Additional data downloading/wrangling on the fly (for augmenting existing data/acquiring pre-clinical/clinical labels)
- [ ] Implement models for modelling adaptive screens
  - [ ] Implement randomized tree regressor predictive of viability reduction from chemcial fingerprint
  - [ ] Implement bayesian dose response model predictive of vaibility reduction and uncertainty from just screen outcomes
  - [ ] Refactor agentic models to serve as additional base lines
