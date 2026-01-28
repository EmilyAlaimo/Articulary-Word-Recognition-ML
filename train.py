import os
import time
import random
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.arff_loader import load_arff_relational
from utils.metrics import evaluate
from data_class.EMADataset import EMADataset

from model_classes.RNN import RNNClassifier
from model_classes.LSTM import LSTMClassifier
from model_classes.GRU import GRUClassifier


def seed_everything(seed: int):
    # Imposta i seed per rendere gli esperimenti riproducibili
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, model_name: str, input_dim: int, num_classes: int):
    # Crea il modello richiesto leggendo i parametri dal config
    mcfg = cfg["model"][model_name]
    hidden_dim = int(mcfg["hidden_dim"])
    num_layers = int(mcfg["num_layers"])
    dropout = float(mcfg["dropout"])

    if model_name == "rnn":
        return RNNClassifier(input_dim, hidden_dim, num_classes, num_layers, dropout)

    if model_name == "lstm":
        bidirectional = bool(mcfg["bidirectional"])
        return LSTMClassifier(input_dim, hidden_dim, num_classes, num_layers, dropout, bidirectional)

    if model_name == "gru":
        bidirectional = bool(mcfg["bidirectional"])
        return GRUClassifier(input_dim, hidden_dim, num_classes, num_layers, dropout, bidirectional)

    raise ValueError(f"Modello non supportato: {model_name}")


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch: int):
    # Esegue 1 epoca di training e ritorna loss media e accuracy
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        #Clear existing gradients
        optimizer.zero_grad(set_to_none=True)
        #forward pass
        logits = model(x)
        #compute loss
        loss = criterion(logits, y)
        #backward pass
        loss.backward()
        #update weights
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += bs

    return {
        "loss": total_loss / max(1, total_samples),             # loss media su tutta l'epoca
        "accuracy": total_correct / max(1, total_samples)      # accuracy media su tutta l'epoca
    }


def run_training(model_name: str, config_path: str = "config/config.yaml"):
    # Training completo (train/val) con checkpoint, early stopping e scheduler

    # 1) Legge il file di configurazione
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2) Seed per riproducibilità
    seed_everything(int(cfg["training"]["seed"]))

    # 3) Selezione device
    device = torch.device(
        "cuda" if cfg["training"]["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )

    # 4) Caricamento dati e split train/val
    train_path = os.path.join(cfg["paths"]["data_dir"], cfg["files"]["train_arff"])
    X, y = load_arff_relational(train_path)

    # CrossEntropyLoss richiede label intere 0..C-1 (nel dataset sono 1..25)
    y = y.astype(np.int64) - 1

    input_dim = int(X.shape[-1])      # F=9
    num_classes = int(y.max() + 1)    # C=25

    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=float(cfg["training"]["val_ratio"]),      # percentuale di validation
        stratify=y,                                         # mantiene distribuzione classi
        random_state=int(cfg["training"]["seed"]),          # rende lo split riproducibile
    )

    train_dl = DataLoader(
        EMADataset(X[tr_idx], y[tr_idx]),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
    )

    val_dl = DataLoader(
        EMADataset(X[va_idx], y[va_idx]),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
    )

    # 5) Costruzione modello
    model = build_model(cfg, model_name, input_dim, num_classes).to(device)

    # 6) Loss, ottimizzatore e scheduler
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    # Metrica usata per best model + early stopping + scheduler
    monitor_key = str(cfg["training"]["monitor"])

    # Decide se massimizzare o minimizzare la metrica
    mode = "min" if monitor_key == "loss" else "max"

    # Scheduler: usa la metrica scelta
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=float(cfg["scheduler"]["factor"]),
        patience=int(cfg["scheduler"]["patience"]),
        min_lr=float(cfg["scheduler"]["min_lr"]),
        threshold=1e-4,
    )

    # 7) Setup checkpoint ed early stopping
    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["paths"]["artifacts_dir"], f"{model_name}_best.pt")

    best = float("inf") if mode == "min" else -float("inf")                 # valore best iniziale (inf per min, -inf per max)
    best_epoch = -1                                                         # epoca in cui si è ottenuto il best
    no_improve = 0                                                          # contatore epoche senza miglioramento
    early_patience = int(cfg["training"]["early_stopping_patience"])        # pazienza per early stopping

    start = time.time()

    print("*" * 80)
    print(model_name.upper())

    # 8) Loop epoche: train -> evaluate -> scheduler -> checkpoint
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        tr = train_one_epoch(model, train_dl, criterion, optimizer, device, epoch)
        va = evaluate(model, val_dl, criterion, device)

        monitor_value = float(va[monitor_key])
        scheduler.step(monitor_value)
        lr = optimizer.param_groups[0]["lr"]

        # print per epoca: train loss/acc + val loss/metriche + lr
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {tr['loss']:.4f}, Train Acc: {tr['accuracy']:.4f} | "
            f"Val Loss: {va['loss']:.4f} | "
            f"Val Acc: {va['accuracy']:.4f}, "
            f"Precision: {va['precision_macro']:.4f}, "
            f"Recall: {va['recall_macro']:.4f}, "
            f"F1: {va['f1_macro']:.4f} | "
            f"LR: {lr:.2e}"
        )

        improved = (monitor_value < best) if mode == "min" else (monitor_value > best)       # verifica se la metrica è migliorata

        # Salva il best model sulla metrica scelta (monitor_key)
        if improved:
            best = monitor_value
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "monitor_key": monitor_key,
                    "best_metric": best,
                    "config": cfg,
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        # Early stopping se non migliora per N epoche
        if no_improve >= early_patience:
            print("Early stopping")
            break

    elapsed = time.time() - start                   # tempo totale trascorso

    print(
        f"Best validation {monitor_key} for {model_name.upper()}: "
        f"{best:.4f} at epoch {best_epoch}"
    )
    print(f"Training took {elapsed:.4f} seconds")
    print("*" * 80)

    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rnn", "lstm", "gru"])           # richiede modello tra rnn/lstm/gru
    args = parser.parse_args()
    run_training(args.model)