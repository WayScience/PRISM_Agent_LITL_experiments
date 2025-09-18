# DSPy + LITL Agentic AI Sandbox


---

## Overview

**Agentic AI** refers to systems that autonomously perform tasks, make decisions, and adapt over time. It builds on the concept of AI agents—models that can process unstructured inputs, reason over context, and execute sequences of actions to achieve an objective. While individual AI agents excel at focused tasks such as information retrieval or text generation, **agentic AI** organizes multiple agents within a collaborative framework. This allows complex problems to be decomposed into smaller, manageable tasks that individual agents can solve.  

Recently, there has been growing interest in applying agentic systems to **scientific discovery**. By combining reasoning, tool use, and iterative feedback, these systems can accelerate hypothesis generation, automate complex analyses, and adapt to evolving experimental data. 

This repository explores one aspect of building **agentic AI system** for the specific scientific problem of predicting in-vivo compound toxicity and efficacy from textual descriptions of compounds, cell lines, and experimental designs. We will focus on experimenting how to best incorporate **Lab-In-The-Loop (LITL)** feedback of true experimental toxicity measurements plus a suite of tools to refine future predictions.

Our objective is not to deliver a polished product, but to **research how agentic systems can be improved through tool-augmented iterations**. A successful outcome would be identifying the most effective tools and feedback strategies that enable the system to generate toxicity predictions with sufficient fidelity (e.g. ranking of compounds) to inform compound prioritization.  

We will use [DSPy](https://github.com/stanfordnlp/dspy) as the framework and simulate LITL feedback using the [DepMap PRISM drug repurposing dataset](https://depmap.org/portal/prism/).  


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

## License

See the [LICENSE](LICENSE) file for full details.
