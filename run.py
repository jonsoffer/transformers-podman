#!/usr/bin/env python

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from transformers.models.auto import configuration_auto
from huggingface_hub import login
import os

login(token = os.environ["HF_TOKEN"])

from transformers import pipeline

qa_model = pipeline("text-generation", model="HuggingFaceTB/SmolVLM-256M-Instruct")
qam = qa_model(text_inputs="what is the meaning of life?")

print(qam)
