"""

Analisi e descrizione del dataset ArticularyWordRecognition.

Il codice:
- carica i file ARFF (formato UCR/UEA relational)
- descrive struttura, dimensioni e classi
- verifica se è stata applicata una z-normalizzazione
  controllando media ≈ 0 e varianza ≈ 1 per ciascuna feature
"""

import os
import argparse
import numpy as np
import yaml

from utils.arff_loader import load_arff_relational


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def describe_dataset(nome: str, X: np.ndarray, y: np.ndarray):
    N, T, F = X.shape
    labels, counts = np.unique(y, return_counts=True)

    print(f"\n[{nome}]")
    print(f"• Numero campioni (N): {N}")
    print(f"• Lunghezza sequenza (T): {T}")
    print(f"• Numero feature (F): {F}  -> ogni campione è una matrice ({T}, {F})")
    print(f"• Numero classi: {len(labels)}")
    print(f"• Label presenti: {int(labels.min())} .. {int(labels.max())}")
    print(f"• Tutte le serie hanno lunghezza fissa {T}")
    print(f"• Valori NaN: {'PRESENTI' if np.isnan(X).any() else 'ASSENTI'}")
    print(f"• Range valori: min={float(X.min()):.4f}, max={float(X.max()):.4f}")

    return {
        "N": N,
        "T": T,
        "F": F,
        "per_classe": int(counts[0])
    }


def check_z_normalization(nome: str, X: np.ndarray):
    """
    Verifica se i dati sono z-normalizzati:
    - media ≈ 0
    - varianza ≈ 1
    per ciascuna feature
    """
    _, _, F = X.shape
    flat = X.reshape(-1, F)   # (N*T, F)

    mean_f = flat.mean(axis=0)
    var_f  = flat.var(axis=0)

    tol_mean = 0.10
    tol_var  = 0.10

    ok = (np.abs(mean_f) < tol_mean) & (np.abs(var_f - 1.0) < tol_var)

    print(f"\n[NORMALIZZAZIONE - {nome}]")
    print(f"• Media per feature:    {np.round(mean_f, 4)}")
    print(f"• Varianza per feature: {np.round(var_f, 4)}")
    print(f"• Feature con mean≈0 e var≈1: {int(ok.sum())}/{F}")

    if ok.sum() == F:
        print("• Conclusione: i dati sono compatibili con una normalizzazione z-score.")
    else:
        print("• Conclusione: i dati NON sono completamente compatibili con una z-normalizzazione.")



def show_sample(X: np.ndarray, y: np.ndarray, idx: int = 0):
    print("\n======================================================================")
    print("ESEMPIO DI UN SINGOLO CAMPIONE")
    print("======================================================================")

    campione = X[idx]
    label = int(y[idx])
    T, F = campione.shape

    print(f"Indice campione: {idx}")
    print(f"Classe: {label}")
    print(f"Forma campione: ({T}, {F})")

    print("\nPrime 3 righe (time step 0..2):")
    np.set_printoptions(precision=4, suppress=True)
    print(campione[:3, :])

    print("\nPrimi 10 valori della feature 0:")
    print(campione[:10, 0])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    data_dir = cfg["paths"]["data_dir"]
    train_path = os.path.join(data_dir, cfg["files"]["train_arff"])
    test_path  = os.path.join(data_dir, cfg["files"]["test_arff"])

    print("======================================================================")
    print("ANALISI DATASET: ArticularyWordRecognition")
    print("======================================================================")

    Xtr, ytr = load_arff_relational(train_path)
    Xte, yte = load_arff_relational(test_path)

    info_tr = describe_dataset("TRAIN", Xtr, ytr)
    info_te = describe_dataset("TEST", Xte, yte)
    
    check_z_normalization("TRAIN", Xtr)
    check_z_normalization("TEST", Xte)

    print("\n======================================================================")
    print("BILANCIAMENTO CLASSI")
    print("======================================================================")
    print(f"TRAIN: {info_tr['per_classe']} campioni per classe (25×{info_tr['per_classe']} = {info_tr['N']})")
    print(f"TEST:  {info_te['per_classe']} campioni per classe (25×{info_te['per_classe']} = {info_te['N']})")

    show_sample(Xtr, ytr, idx=0)


if __name__ == "__main__":
    main()