#!/usr/bin/env python

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from transformers.models.auto import configuration_auto
from huggingface_hub import login
import os
import torch
from transformers import pipeline
import mlflow
import mlflow.transformers
import subprocess

login(token=os.environ["HF_TOKEN"])

# Define both the repository ID and the specific GGUF file name
model_id = "jc-builds/Qwen3.5-9B-Q4_K_M-GGUF"
gguf_file = "Qwen3.5-9B-Q4_K_M.gguf"

# 1. Clear any leftover cache from the previous crash
torch.cuda.empty_cache()

# 2. Initialize the pipeline with GGUF configurations
qa_model = pipeline(
    "text-generation", 
    model=model_id,
    model_kwargs={"gguf_file": gguf_file},  # Tells transformers to extract the GGUF binary
    torch_dtype=torch.bfloat16,             
    device_map="auto"                       # Strongly recommended to handle device allocation automatically
)

# 3. Use a context manager to make sure gradients aren't tracked
with torch.no_grad():
    with mlflow.start_run() as run:
        mlflow.transformers.log_model(
            transformers_model=qa_model,
            artifact_path="model",
            tasks="text-generation"
        )

        # Fetch and print the run ID
        run_id = run.info.run_id
        print(f"\n" + "="*40)
        print(f"Model logged successfully!")
        print(f"Your Run ID is: {run_id}")
        print(f"Serve it using: mlflow models serve -m 'runs:/{run_id}/model'")
        print("="*40 + "\n")

        # Define the exact CLI command you want to execute
        serve_command = [
            "mlflow", "models", "serve",
            "-m", f"runs:/{run_id}/model",
            "--port", "5000",
            "--host", "0.0.0.0",
            "--env-manager", "local"
        ]
        
        try:
            # subprocess.run will execute the command and keep the script alive,
            # printing the live MLflow server logs directly to your terminal.
            subprocess.run(serve_command, check=True)
        except KeyboardInterrupt:
            print("\nServer stopped manually by user.")