import copy
import json
import random
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, theme_minimal
from matplotlib.ticker import MaxNLocator

import torch
import transformers
from datasets import Dataset
from transformers import Trainer

# Update pyreft imports to use relative imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyreft import (
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    ReftDataCollator,
    ReftSupervisedDataset,
    make_last_position_supervised_data_module,
    ConsreftIntervention,
    LoreftIntervention
)

from .finetuning.load_model import load_model_and_tokenizer
from .finetuning.get_dataset import get_goody2_dataset
from .finetuning.generate_response import generate_base_response, generate_reft_response
from .finetuning.finetune import train_reft_model

# Configure matplotlib
plt.rcParams.update({'font.size': 20, 'font.family': 'Sans'})

# Constants
IGNORE_INDEX = -100
device = "cuda" if torch.cuda.is_available() else "cpu"

def max_char_match_length(retrieved, golden):
    n_c, n = 0, 0
    for char in retrieved:
        if char == golden[n]:
            n_c += 1
        else:
            break
        n += 1 
    if len(retrieved) == 0:
        return 0.0
    return round(n_c/len(retrieved), 2)

make_supervised_data_module = make_last_position_supervised_data_module

def main():
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Example of original model response
    print("Original model response:")
    original_response = generate_base_response(
        model, tokenizer,
        "Which dog breed do people think is cuter, poodle or doodle?",
        device
    )
    print(original_response)
    print("\n" + "="*50 + "\n")
    
    # Train ReFT model
    dataset = get_goody2_dataset()
    reft_model = train_reft_model(model, tokenizer, dataset)
    
    # Example of ReFT model response
    print("ReFT model response:")
    reft_response = generate_reft_response(
        reft_model, tokenizer,
        "Which dog breed do people think is cuter, poodle or doodle?"
    )
    print(reft_response)

if __name__ == "__main__":
    main() 