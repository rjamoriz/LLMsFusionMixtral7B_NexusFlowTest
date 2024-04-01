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



