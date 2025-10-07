# Modal LLM Deployment

Optional script for deploying open-source language models using [Modal](https://modal.com/) for use in the language system experiments.

## Scripts

The `deploy_llama31_8b.py` script deploys a Llama 3.1 8B Instruct model as a web service using vLLM for efficient inference. This provides a scalable alternative to local GPU deployment.

## Alternative Deployment

Instead of using Modal, you can:
- Deploy the language models locally on your GPU
- Modify the analysis scripts to point to your local endpoints
- Use any other cloud provider or containerized deployment

## Requirements

- Modal account and API key
- Hugging Face token for model access
- The deployment uses A100 GPUs and Modal's volume system for caching

## Usage

Install Modal and set up your credentials:

1. Create a Modal account at [modal.com](https://modal.com/) if you don't have one

2. Install the Modal Python package. Follow installation to register your modal credientials to environment.
    ```bash
    pip install modal
    ```

3. Configure Huggin Face token for model access
    ```bash
    modal secret create ...
    ```

4. Deploy the service with:
    ```bash
    modal deploy deploy_llama31_8b.py
    ```
    and confirm with:
    ```bash
    modal app list
    ```

5. The service will be available as an OpenAI-compatible API endpoint for integration as the `DSPy` language model backend for the agentic system experiments. 
    ```python
    MODAL_BASE = "https://{account}--hf-endpoint-llama31-8b-serve-llama31-8b.modal.run"

    LM_CONFIG = {
        "model": "openai/meta-llama/Llama-3.1-8B-Instruct",
        "api_key": "",
        "temperature": 1.0,
        "max_tokens": 20000,
        "seed": 42
    }
    ```
