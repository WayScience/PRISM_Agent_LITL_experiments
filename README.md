# DSPy + LITL Agentic AI Sandbox

**Preliminary Repository** – This project is in the early exploratory stage.

---

## Overview

This repository is dedicated to exploring **agentic AI infrastructure** using [DSPy](https://github.com/stanfordnlp/dspy) on **hypothetical LITL (Learning-in-the-Loop) data**, simulated by the [DepMap PRISM drug repurposing dataset](https://depmap.org/portal/prism/).

The goal is to **research how agentic systems can be improved through tool-augmented iterations**, not necessarily to provide a high-utility product.

---

## Planned Components

- **Agentic Infrastructure**  
  - DSPy-based backbone for building and orchestrating agents.  
  - Support for modular tool integration.  

- **Tool Library**  
  - A collection of domain-agnostic and domain-specific tools made available to agents.  
  - Designed for flexible composition in different experimental runs.  

- **Experiments**  
  - Jupyter notebooks, Python scripts, and results testing different agent-tool configurations.  
  - Benchmarks of system performance over successive **LITL iterations**.  

---

## Roadmap (Tentative)

- [ ] Scaffold repo structure  
  - [x] Defined the `agentic_system` and `analysis` compartment.
  - [x] Added early stage conda env, pyproject toml and uv lock.
  - [ ] To be decided.
- [ ] Define minimal DSPy agent with placeholder tool  
- [ ] Implement first toy experiment  
- [ ] Expand tool library with academic use cases  
- [ ] Iterative experiments on hypothetical PRISM-style LITL data  

---

## License

This project is licensed under the **BSD 3-Clause License**.  
See the [LICENSE](LICENSE) file for full details.
