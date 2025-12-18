#!/usr/bin/env python
# coding: utf-8

import torch
import json
from typing import Dict, List
from torch.utils.data import Dataset


def print_gpu_memory():
    """Print current GPU memory usage."""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"Allocated GPU memory: {allocated / (1024**2):.2f} MB")
    print(f"Reserved GPU memory: {reserved / (1024**2):.2f} MB")


def write_file(content, filepath):
    """Append content to a file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(content + '\n')


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    """Prepare model for k-bit quantized training."""
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    
    for name, param in model.named_parameters():
        param.requires_grad = False

    if not is_gptq_quantized:
        for param in model.parameters():
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)

    if (loaded_in_kbit or is_gptq_quantized) and use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    return model


def tokenize(tokenizer, prompt, add_eos_token=True, max_length=512):
    """Tokenize a prompt."""
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    if (result["input_ids"][-1] != tokenizer.eos_token_id 
        and len(result["input_ids"]) < max_length 
        and add_eos_token):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point, tokenizer, prompter, train_on_inputs=False):
    """Generate and tokenize a prompt from a data point."""
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"] if "input" in data_point else None,
        data_point["output"]
    )
    
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)
    
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"] if "input" in data_point else None
        )
        tokenized_user_prompt = tokenize(tokenizer, user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + \
            tokenized_full_prompt["input_ids"][user_prompt_len:]
    
    return tokenized_full_prompt


def prepare_local_dataset(local_data, tokenizer, prompter, local_val_set_size=0):
    """Prepare local training dataset."""
    def tokenize_fn(data_point):
        return generate_and_tokenize_prompt(data_point, tokenizer, prompter)
    
    if local_val_set_size > 0:
        local_train_val = local_data["train"].train_test_split(
            test_size=local_val_set_size, shuffle=True, seed=42
        )
        local_train_dataset = (
            local_train_val["train"]
            .shuffle()
            .map(tokenize_fn, remove_columns=local_train_val["train"].column_names)
        )
        local_eval_dataset = (
            local_train_val["test"]
            .shuffle()
            .map(tokenize_fn, remove_columns=local_train_val["test"].column_names)
        )
    else:
        local_train_dataset = local_data["train"].shuffle().map(
            tokenize_fn, remove_columns=local_data["train"].column_names
        )
        local_eval_dataset = None

    return local_train_dataset


def get_round_specific_paths(eval_files, score_files, round_idx):
    """Generate round-specific file paths."""
    round_eval_files = {
        client_id: file_path.replace('.jsonl', f'_round{round_idx+1}.jsonl')
        for client_id, file_path in eval_files.items()
    }
    round_score_files = {
        client_id: file_path.replace('.json', f'_round{round_idx+1}.json')
        for client_id, file_path in score_files.items()
    }
    return round_eval_files, round_score_files


class EvalDataset(Dataset):
    """Dataset for evaluation."""
    def __init__(self, file, prompter, tokenizer, max_len=512):
        self.prompter = prompter
        self.tokenizer = tokenizer
        with open(file, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        ques = json.loads(line)
        sample = ques['instruction']
        prompt = self.prompter.generate_prompt(sample, None)
        return prompt, sample
