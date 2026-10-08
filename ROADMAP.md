# Adaptive screening simulated retrospectively with PRISM

## Roadmap
- [ ] Re-scaffold repo
  - [ ] Better way of structuring both analysis and implementation of machine learning models (including already implemented agents).
- [ ] Data
  - [x] PRSIM secondary LFC input data download and wrangling 
  - [ ] Additional data downloading/wrangling on the fly (for augmenting existing data/acquiring pre-clinical/clinical labels)
- [ ] Implement machine learning models for adaptive screens
  - [ ] Implement randomized tree regressor predictive of viability reduction from chemcial fingerprint
  - [x] Implement bayesian dose response model predictive of vaibility reduction and uncertainty from just screen outcomes
  - [ ] Refactor agentic models to serve as additional base lines
