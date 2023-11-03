from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("../data/raw/filtered.tsv")

# Preprocessing function
def preprocess_function(examples):
    inputs = [prefix + ref for ref in examples["reference"]]
    targets = [tsn for tsn in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_overflowing_tokens=False)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, return_overflowing_tokens=False)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize and preprocess dataset
batch_size = 256
tokenized_datasets = dataset.map(preprocess_function, batched=True, batch_size=batch_size)

