import argparse
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.arff_loader import load_arff_relational
from utils.metrics import evaluate
from data_class.EMADataset import EMADataset
from train import build_model


def run_test(model_name: str, config_path: str = "config/config.yaml"):
    # carica config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if cfg["training"]["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )

    # path checkpoint: artifacts/<model>_best.pt
    ckpt_path = os.path.join(cfg["paths"]["artifacts_dir"], f"{model_name}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)

    # carica test set
    test_path = os.path.join(cfg["paths"]["data_dir"], cfg["files"]["test_arff"])
    X_test, y_test = load_arff_relational(test_path)
    y_test = y_test.astype(np.int64) - 1

    test_dl = DataLoader(
        EMADataset(X_test, y_test),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
    )

    # ricostruisce modello
    model = build_model(
        cfg,
        model_name,
        ckpt["input_dim"],
        ckpt["num_classes"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    # valuta su test
    metrics = evaluate(model, test_dl, torch.nn.CrossEntropyLoss(), device)

    print("*" * 80)
    print(f"TEST RESULTS ({model_name.upper()})")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("*" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rnn", "lstm", "gru"])
    args = parser.parse_args()

    run_test(args.model)
