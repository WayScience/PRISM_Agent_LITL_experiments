# deploy_llama31_8b.py

import modal, os, subprocess

# ---- Image with vLLM + CUDA 12.8 wheels ----
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_USE_V1": "1",  # keep new engine
    })
)

# ---- Model + caching ----
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("hf-endpoint-llama31-8b")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000
FAST_BOOT = True  # faster cold starts

HF_SECRET = modal.Secret.from_name(
    "huggingface-token")  # contains HUGGING_FACE_HUB_TOKEN

@app.function(
    image=vllm_image,
    gpu=f"A100:{N_GPU}",
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[HF_SECRET],  # <-- inject HF token into env
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)  # give first download more time
def serve_llama31_8b():
    # vLLM / transformers automatically read HUGGING_FACE_HUB_TOKEN from env.
    env = os.environ.copy()

    cmd = [
        "vllm", "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name", MODEL_NAME,  # what /v1/models will report
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
    ]

    # cold-start speed vs throughput
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print("Launching:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True, env=env)
