import torch
import torch.nn as nn
import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import yaml
from typing import Tuple, Optional
from task import Net, get_weights


class FedAvgWithMetrics(FedAvg):
    def initialize_parameters(self, client_manager):
        print("[Global Server] Initializing global parameters from Net() model...")
        model = Net()
        weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
        print(weights[0])
        return ndarrays_to_parameters(weights)

    def aggregate_evaluate(
        self,
        rnd: int,
        results: list,
        failures: list,
    ) -> Optional[float]:
        if not results:
            return None

        accuracies = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
        losses = [r.metrics["loss"] for _, r in results if "loss" in r.metrics]
        maes = [r.metrics["mae"] for _, r in results if "mae" in r.metrics]
        rmses = [r.metrics["rmse"] for _, r in results if "rmse" in r.metrics]

        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
        avg_loss = sum(losses) / len(losses) if losses else None
        avg_mae = sum(maes) / len(maes) if maes else None
        avg_rmse = sum(rmses) / len(rmses) if rmses else None

        print(f"Round {rnd} - Avg Loss (MSE): {avg_loss:.4f}, Avg RMSE: {avg_rmse}, Avg MAE: {avg_mae}")

        return super().aggregate_evaluate(rnd, results, failures)



if __name__ == "__main__":
    print("[Global Server] worked")

    with open("global_server_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    server_cfg = config["server"]
    strategy_cfg = config["strategy"]

    # Read from config
    num_rounds = server_cfg["global_rounds"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Build strategy params dynamically
    strategy = FedAvgWithMetrics(
        fraction_fit=strategy_cfg["fraction_fit"],
        fraction_evaluate=strategy_cfg["fraction_evaluate"],
        min_fit_clients=strategy_cfg["min_fit_clients"],
        min_evaluate_clients=strategy_cfg["min_evaluate_clients"],
        min_available_clients=strategy_cfg["min_available_clients"],
        initial_parameters=parameters,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=server_cfg["address"],
        config=fl.server.ServerConfig(num_rounds=server_cfg["global_rounds"]),
        strategy=strategy,
    )
