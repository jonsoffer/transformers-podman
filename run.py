#!/usr/bin/env python

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from transformers.models.auto import configuration_auto
from huggingface_hub import login
import os
from time import sleep
import transformers

print(transformers.__version__)

login(token = os.environ["HF_TOKEN"])

from transformers import pipeline

model = "google/gemma-4-E2B"

qa_model = pipeline("text-generation", model=model)
qam = qa_model(text_inputs="what is the meaning of life?")

print(qam)

sleep(1000000)
