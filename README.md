# LLMsFusionMixtral7B_NexusFlowTest
Fusion LLMS Mixtral y Nexis FLow Starlink

Repository reference:
https://github.com/arcee-ai/mergekit

Si se quiere activar por comandos:

# Command to execute merge llms 
(fusion_llms) (base) Ruben_MACPRO@MacBook-Pro-Ruben FUSION LLMs % mergekit-yaml mergeconfig.yaml merged_folder /
list conf parameters....allow crimes   copy tokenizer etc...

# models llms:
# huggingface opensource:

Nexusflow/Starling-LM-7B-beta:
https://huggingface.co/Nexusflow/Starling-LM-7B-beta

mistralai/Mistral-7B-Instruct-v0.2
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

Para hacerlo por el codigo:

# Description: Configuration file for fusionllmmixtral_llama27b.py
# Author: Ruben Jimenez
# Last Modified: 1-4-24
# Notes: This file is a modified version of the original fusionllmmixtral_llama27b.py file. The original file can be found at

OUTPUT_PATH = "/Users/Ruben_MACPRO/Desktop/IA DevOps/FUSION LLMs/merged"  # folder to store the result in
LORA_MERGE_CACHE = "/tmp"  # change if you want to keep these for some reason
CONFIG_YML = "/Users/Ruben_MACPRO/Desktop/IA DevOps/FUSION LLMs/mergeconfig.yml"  # merge configuration file
COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

# Importing necessary libraries for the configuration file
import torch
import yaml

# Using the mergekit library to import the necessary functions for the configuration
# file
# The mergekit library is a library that allows for the merging of two models
# The library is a part of the LLAMA project
# The LLAMA project is a project that aims to create a large language model
# The project is a part of the OpenAI project

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

# Loading the configuration file
with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

# Running the merge process with the configuration file and the specified options
    run_merge(
    merge_config,
    out_path=OUTPUT_PATH, # where to store the result in
    options=MergeOptions(
      lora_merge_cache=LORA_MERGE_CACHE, # where to store the intermediate files
      cuda=torch.cuda.is_available(), # use CUDA if available
      copy_tokenizer=COPY_TOKENIZER, # copy the tokenizer from the first model
      lazy_unpickle=LAZY_UNPICKLE, # experimental low-memory model loader
      low_cpu_memory=LOW_CPU_MEMORY # enable if you somehow have more VRAM than RAM+swap
    ),  
)
print("Done!")

**********************************
Es necesario el aerchivo yaml de configuracion:

slices:
  - sources:
      - model: mistralai/Mistral-7B-Instruct-v0.2
        layer_range: [0, 40]
      - model: Nexusflow/Starling-LM-7B-beta
        layer_range: [0, 40]
# models: 2 models to merge, MIXTRAL and NexusFlow
#   - model: mistralai/Mistral-7B-Instruct-v0.2
#   - model: Nexusflow/Starling-LM-7B-beta
# layer_range: [0, 40] for both models, to merge first 40 layers
# Slerp is Spherical Linear Interpolation for smooth transition between models
# base_model: mistralai/Mistral-7B-Instruct-v0.2
# merge_method: slerp
# parameters: t for interpolation between models
# Self-attention weights are interpolated between 0 and 1
#   - filter: self_attn
# Value of MLP weights are interpolated between 1 and 0
#     value: [0, 0.5, 0.3, 0.7, 1]

merge_method: slerp
base_model: mistralai/Mistral-7B-Instruct-v0.2
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5 # fallback for rest of tensors
dtype: float16
# dtype: float16 for faster inference and less memory usage 
