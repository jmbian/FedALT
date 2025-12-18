#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse
from datasets import load_dataset
from transformers import LlamaTokenizer
from tqdm import tqdm

from utils.prompter import Prompter
from utils import prepare_local_dataset, get_round_specific_paths, print_gpu_memory
from client import Client
from server import Server


def train_federated(
    clients,
    server,
    global_rounds,
    local_epochs,
    test_files, 
    eval_files,  
    score_files,
    result_dir,
    lr=1e-8
):
    """
    Main federated training loop.
    
    Args:
        clients: List of Client instances
        server: Server instance
        global_rounds: Number of global communication rounds
        local_epochs: Number of local training epochs per round
        test_files: Dictionary mapping client_id to test file paths
        eval_files: Dictionary mapping client_id to evaluation output paths
        score_files: Dictionary mapping client_id to score output paths
        result_dir: Directory to save results
        lr: Learning rate
    """
    all_client_scores = {client.client_id: [] for client in clients}
    
    for round_idx in tqdm(range(global_rounds), desc="Global Rounds"):
        print(f"\nGlobal Round {round_idx + 1}/{global_rounds}")
        
        round_eval_files, round_score_files = get_round_specific_paths(
            eval_files, score_files, round_idx
        )
        
        # Local training phase
        client_params = []
        for client in tqdm(clients, desc="Client Training"):
            client.load_model()
            if round_idx > 0:
                client.load_params(aggregated_params[client.client_id])
            client.local_training(lr=lr, epochs=local_epochs, batch_size=8)
            params = client.get_lora_params()
            client_params.append(params['params'])
            print("Before unloading:")
            print_gpu_memory()
            client.unload_model()
            print("After unloading:")
            print_gpu_memory()
            
        # Server aggregation phase
        if round_idx >= 0:
            aggregated_params = server.aggregation(
                route_aggregation=False,
                params=client_params
            )
        
        # Evaluation phase (every 5 rounds)
        if (round_idx + 1) % 5 == 0:
            print(f"\nRound {round_idx + 1} Evaluation Scores:")
            round_scores = {}
            for client in clients:
                client_id = client.client_id
                client.load_model()
                client.load_params(aggregated_params[client_id])
                scores = client.evaluate_and_score(
                    test_file=test_files[client_id],
                    output_file=round_eval_files[client_id], 
                    score_file=round_score_files[client_id]   
                )
                all_client_scores[client_id].append(scores)
                round_scores[client_id] = scores
                print(f"\nClient {client_id} ROUGE Scores:")
                print(scores)
                client.unload_model()
                
            summary_file = os.path.join(result_dir, f"round_summary_{round_idx + 1}.json")
            with open(summary_file, 'w') as f:
                json.dump(round_scores, f, indent=2)
    
    return all_client_scores


def main():
    """Main entry point for FedALT training."""
    parser = argparse.ArgumentParser(description='FedALT: Federated Fine-Tuning with Adaptive Local Training')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf', help='Base model name')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to training data directory')
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--rounds', type=int, default=20, help='Number of global communication rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local training epochs')
    parser.add_argument('--client_num', type=int, default=8, help='Number of clients')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    
    args = parser.parse_args()
    
    # Configuration from arguments
    local_epochs = args.local_epochs
    rounds = args.rounds
    client_num = args.client_num
    model_name = args.model_name
    data_path = args.data_path
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize tokenizer and prompter
    prompter = Prompter("alpaca_short")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    # Dataset mapping
    test_client_pairs = {
        0: 'ag_news_subset',
        1: 'snli',
        2: 'openbookqa',
        3: 'glue_mrpc',
        4: 'story_cloze',
        5: 'common_gen',
        6: 'sentiment140',
        7: 'definite_pronoun_resolution'
    }
    
    # Initialize clients
    clients = []
    for client_id in range(client_num):
        local_data_path = os.path.join(data_path, f"local_training_{client_id}.json")
        client_data = load_dataset("json", data_files=local_data_path)
        local_data = prepare_local_dataset(client_data, tokenizer, prompter)
        clients.append(Client(
            client_id, 
            local_data, 
            tokenizer,
            prompter,
            model_name, 
            rank=args.rank, 
            lora_n=2, 
            asymmetric=False
        ))
    
    # Initialize server
    server = Server(clients_num=len(clients))
    
    # Setup file paths
    eval_files = {
        client_id: f"{result_dir}/eval_client{client_id}.jsonl"
        for client_id in range(len(clients))
    }
    score_files = {
        client_id: f"{result_dir}/scores_client{client_id}.json"
        for client_id in range(len(clients))
    }
    test_files = {
        client_id: os.path.join(data_path, "test", f"local_testing_{client_id}.jsonl")
        for client_id in range(len(clients))
    }
    
    # Run federated training
    all_client_scores = train_federated(
        clients=clients,
        server=server,
        global_rounds=rounds,
        local_epochs=local_epochs,
        test_files=test_files,
        eval_files=eval_files,
        score_files=score_files,
        result_dir=result_dir,
        lr=args.lr
    )
    
    print("\nTraining completed!")
    print("Final Evaluation Scores for each client:", all_client_scores)
    
    # Save final results
    with open(f"{result_dir}/final_scores.json", 'w') as f:
        json.dump(all_client_scores, f, indent=2)


if __name__ == "__main__":
    main()
