#!/usr/bin/env python
# coding: utf-8

import torch
from typing import List, Dict


class Server:
    """Federated Learning Server for parameter aggregation."""
    
    def __init__(self, clients_num: int, device: str = "cuda"):
        self.clients_num = clients_num
        self.device = device
        self.select_result = None

    def aggregation(self, route_aggregation: bool, params: List) -> List[Dict]:
        """
        Aggregate client parameters using FedALT strategy.
        
        Args:
            route_aggregation: Whether to aggregate routing parameters
            params: List of client parameter dictionaries
            
        Returns:
            List of aggregated parameters for each client
        """
        gpu_params = [
            {k: v.to(self.device) for k, v in client_params.items()}
            for client_params in params
        ]

        num_clients = len(gpu_params)
        aggregated_results = [{} for _ in range(num_clients)]
        param_names = gpu_params[0].keys()

        for client_idx in range(num_clients):
            for param_name in param_names:
                # Handle routing parameters
                if 'lora_route' in param_name:
                    if route_aggregation:
                        stacked_params = torch.stack([
                            gpu_params[i][param_name]
                            for i in range(num_clients)
                        ]).to(self.device)
                        aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                    continue

                # Aggregate local LoRA parameters from other clients (Rest-of-World)
                if 'lora_A1' in param_name or 'lora_B1' in param_name:
                    aggregated_name = param_name.replace('A1', 'A0').replace('B1', 'B0')
                    stacked_params = torch.stack([
                        gpu_params[i][param_name]
                        for i in range(num_clients) if i != client_idx 
                    ]).to(self.device)
                    aggregated_results[client_idx][aggregated_name] = stacked_params.mean(dim=0)

        return aggregated_results
    

