#!/usr/bin/env python

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from transformers.models.auto import configuration_auto
from huggingface_hub import login
import os
import torch
from transformers import pipeline

login(token=os.environ["HF_TOKEN"])

model_id = "google/gemma-4-E2B"

# 1. Clear any leftover cache from the previous crash
torch.cuda.empty_cache()

# 2. Initialize the pipeline with memory optimizations
qa_model = pipeline(
    "text-generation", 
    model=model_id,
    torch_dtype=torch.bfloat16,     # Cuts the memory footprint in half (use torch.float16 if your GPU is older)
    device_map="auto"               # Dynamically manages memory allocation
)

# 3. Use a context manager to make sure gradients aren't tracked
with torch.no_grad():
    qam = qa_model("what is the meaning of life?", max_new_tokens=50)
    print(qam)
