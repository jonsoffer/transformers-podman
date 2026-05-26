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

model_id = "google/gemma-4-E2B"

# 1. Clear any leftover cache from the previous crash
torch.cuda.empty_cache()

# 2. Initialize the pipeline with memory optimizations
qa_model = pipeline(
    "text-generation", 
    model=model_id,
    torch_dtype=torch.bfloat16,     # Cuts the memory footprint in half (use torch.float16 if your GPU is older)
    # device_map="auto"               # Dynamically manages memory allocation
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
        # Using 127.0.0.1 or 0.0.0.0 depending on your network needs
        serve_command = [
            "mlflow", "models", "serve",
            "-m", f"runs:/{run_id}/model",
            "--port", "5000",
            # "--host", "127.0.0.1" 
            "--host", "0.0.0.0",
            "--env-manager", "local"
        ]
        
        try:
            # subprocess.run will execute the command and keep the script alive,
            # printing the live MLflow server logs directly to your terminal.
            subprocess.run(serve_command, check=True)
        except KeyboardInterrupt:
            print("\nServer stopped manually by user.")

