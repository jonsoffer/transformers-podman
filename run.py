#!/usr/bin/env python

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from transformers.models.auto import configuration_auto
from huggingface_hub import login
import os

login(token = os.environ["HF_TOKEN"])

print(configuration_auto.CONFIG_MAPPING.items())

# This will AUTOMATICALLY download the model from Hugging Face!
# No need to manually download anything!
model_name = "chiedo/hello-world"  # Replace with your actual model name

print("Downloading model... (this happens only once)")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Test the model
output = model.generate_hello_world()
print(output)  # "Hello World!"


