#!/usr/bin/env python
# coding: utf-8

"""
Configuration file for FedALT training.
Modify these parameters according to your setup.
"""

# Model Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
CACHE_PATH = "/path/to/model/cache"

# Training Configuration
GLOBAL_ROUNDS = 20
LOCAL_EPOCHS = 5
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1

# LoRA Configuration
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_N = 2  # Number of LoRA adapters (local + RoW)
TARGET_MODULES = ["q_proj", "v_proj"]
ASYMMETRIC = False

# Federated Learning Configuration
CLIENT_NUM = 8
ADAPTIVE_M = 1  # Parameter for adaptive selection

# Data Configuration
DATA_PATH = "path/to/data"
TEST_PATH = "path/to/test"
RESULT_DIR = "path/to/results"

# Evaluation Configuration
EVAL_BATCH_SIZE = 2
EVAL_TEMPERATURE = 0.1
EVAL_TOP_P = 0.75
EVAL_TOP_K = 40
EVAL_NUM_BEAMS = 4
EVAL_MAX_NEW_TOKENS = 80
EVAL_FREQUENCY = 5  # Evaluate every N rounds

# Quantization Configuration
LOAD_IN_8BIT = True
LLM_INT8_THRESHOLD = 6.0
BNB_8BIT_COMPUTE_DTYPE = "float16"
BNB_8BIT_USE_DOUBLE_QUANT = True

# Prompt Configuration
PROMPT_TEMPLATE = "alpaca_short"

# Dataset Mapping
TEST_CLIENT_PAIRS = {
    0: 'ag_news_subset',
    1: 'snli',
    2: 'openbookqa',
    3: 'glue_mrpc',
    4: 'story_cloze',
    5: 'common_gen',
    6: 'sentiment140',
    7: 'definite_pronoun_resolution'
}
