# ============================================================
# FEATURE ENGINEERING per ArticularyWordRecognition (UEA/UCR)
#
# SCOPO:
#   Convertire un dataset di serie temporali multivariate (N, T, F)
#   in un dataset tabellare (N, K) per usare modelli "tabulari"
#   come SVM, RandomForest, kNN.
# ============================================================

import os                
import math               
import argparse           
import numpy as np        
import pandas as pd      
import yaml             
from utils.arff_loader import load_arff_relational


# ============================================================
# FUNZIONI BASE SU UN SEGNALE 1D (un singolo canale)
# ============================================================

def energy(x: np.ndarray) -> float:
    # Energia = somma dei quadrati -> misura l'ampiezza complessiva del segnale
    # E = x1^2 + x2^2 + ... + xT^2
    return float(np.sum(x * x)) if x.size else 0.0


def rms(x: np.ndarray) -> float:
    # RMS = sqrt(energia / numero_campioni)
    # misura l'ampiezza "media" del segnale
    if x.size == 0:
        return 0.0
    return float(math.sqrt(energy(x) / x.size))


def skewness(x: np.ndarray) -> float:
    # Skewness = asimmetria della distribuzione dei valori del segnale
    # Se std=0 (segnale costante) la skewness non è definita -> mettiamo 0
    if x.size < 2:
        return 0.0
    mu = float(np.mean(x))
    s = float(np.std(x))
    if s == 0:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 3))


def kurtosis(x: np.ndarray) -> float:
    # Kurtosis = "quanto è appuntita" la distribuzione (picchi/code)
    # Anche qui: se std=0, non definita -> 0
    if x.size < 2:
        return 0.0
    mu = float(np.mean(x))
    s = float(np.std(x))
    if s == 0:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 4))


def zero_crossings(v: np.ndarray) -> int:
    # Conta quante volte il segnale cambia segno.
    # Qui la usiamo su dx (derivata discreta) per stimare quante volte cambia direzione.
    if v.size < 2:
        return 0

    s = np.sign(v)  # converte v in segni: -1, 0, +1

    # Se ci sono zeri, li sostituiamo col segno precedente per evitare crossing finti
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]

    # crossing se segni consecutivi hanno prodotto negativo (es. +1 * -1 = -1)
    return int(np.sum(s[1:] * s[:-1] < 0))


def autocorrelation_lag1(x: np.ndarray) -> float:
    # Autocorrelazione a lag 1: corr(x[t], x[t+1])
    # Se è alta, il segnale cambia lentamente (molto "liscio")
    # Se è bassa, è più "irregolare"
    if x.size < 2:
        return 0.0

    x0 = x[:-1]
    x1 = x[1:]

    # Se uno dei due vettori è costante, la correlazione non è definita
    if np.std(x0) == 0 or np.std(x1) == 0:
        return 0.0

    return float(np.corrcoef(x0, x1)[0, 1])


# ============================================================
# FEATURE PER UN CANALE 1D: da vettore lungo T -> a dizionario feature
# ============================================================

def extract_features_1d(x: np.ndarray, prefix: str) -> dict:
    # x è la serie temporale di UN canale: shape (T,)
    # prefix serve a nominare le feature in modo univoco, es. ch1_mean, ch1_std, ...

    # 1) Conversione + pulizia numerica (evita NaN/inf)
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) STATISTICHE BASE (descrivono il "livello" e l'ampiezza)
    mean = float(np.mean(x)) if x.size else 0.0
    std = float(np.std(x)) if x.size else 0.0
    minv = float(np.min(x)) if x.size else 0.0
    maxv = float(np.max(x)) if x.size else 0.0
    value_range = maxv - minv
    median = float(np.median(x)) if x.size else 0.0

    # 3) STATISTICHE ROBUSTE (meno sensibili a outlier)
    q25 = float(np.quantile(x, 0.25)) if x.size else 0.0
    q75 = float(np.quantile(x, 0.75)) if x.size else 0.0
    iqr = q75 - q25
    mean_abs = float(np.mean(np.abs(x))) if x.size else 0.0

    # 4) INTENSITÀ (quanto segnale totale c'è)
    sig_energy = energy(x)
    sig_rms = rms(x)

    # 5) FORMA DELLA DISTRIBUZIONE (asimmetria e "picchi")
    sig_skew = skewness(x)
    sig_kurt = kurtosis(x)

    # 6) CONTINUITÀ TEMPORALE (quanto x[t] somiglia a x[t+1])
    acf1 = autocorrelation_lag1(x)

    # 7) "TIMING" GROSSOLANO DI EVENTI IMPORTANTI
    #    NON salviamo tutta la sequenza,
    #    ma salviamo *dove* (inizio/metà/fine) avvengono min e max.
    if x.size:
        start = float(x[0])  # valore al primo istante
        end = float(x[-1])  # valore all'ultimo istante

        # normalizzazione indice: dividiamo per (T-1) -> intervallo [0,1]
        denom = float(max(1, x.size - 1))
        argmax_norm = float(np.argmax(x) / denom)
        argmin_norm = float(np.argmin(x) / denom)
    else:
        start = end = argmax_norm = argmin_norm = 0.0

    # 8) DINAMICA: derivata discreta dx[t] = x[t+1] - x[t]
    #    Questo cattura "come si muove" il segnale (sale/scende/oscilla)
    dx = np.diff(x) if x.size >= 2 else np.array([], dtype=np.float64)

    dx_mean = float(np.mean(dx)) if dx.size else 0.0
    dx_std = float(np.std(dx)) if dx.size else 0.0
    dx_min = float(np.min(dx)) if dx.size else 0.0
    dx_max = float(np.max(dx)) if dx.size else 0.0
    dx_mean_abs = float(np.mean(np.abs(dx))) if dx.size else 0.0
    dx_energy = energy(dx)
    dx_rms = rms(dx)
    dx_zero_cross = zero_crossings(dx)  # quante volte cambia direzione

    # 9) OUTPUT: dizionario di feature nominato con prefix
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": minv,
        f"{prefix}_max": maxv,
        f"{prefix}_range": value_range,
        f"{prefix}_median": median,
        f"{prefix}_q25": q25,
        f"{prefix}_q75": q75,
        f"{prefix}_iqr": float(iqr),
        f"{prefix}_mean_abs": mean_abs,
        f"{prefix}_energy": sig_energy,
        f"{prefix}_rms": sig_rms,
        f"{prefix}_skew": sig_skew,
        f"{prefix}_kurt": sig_kurt,
        f"{prefix}_acf1": acf1,
        f"{prefix}_start": start,
        f"{prefix}_end": end,
        f"{prefix}_argmax_norm": argmax_norm,
        f"{prefix}_argmin_norm": argmin_norm,
        f"{prefix}_dx_mean": dx_mean,
        f"{prefix}_dx_std": dx_std,
        f"{prefix}_dx_min": dx_min,
        f"{prefix}_dx_max": dx_max,
        f"{prefix}_dx_mean_abs": dx_mean_abs,
        f"{prefix}_dx_energy": dx_energy,
        f"{prefix}_dx_rms": dx_rms,
        f"{prefix}_dx_zero_cross": dx_zero_cross,
    }


# ============================================================
# FEATURE MULTIVARIATE: correlazioni tra canali nello stesso esempio
# Estrae feature multivariate di coordinazione tra i canali di un singolo esempio.
# Calcola la correlazione di Pearson tra ogni coppia di canali nel tempo.
# Valori positivi indicano movimenti sincroni, negativi movimenti opposti.
# Valori prossimi a zero indicano assenza di relazione lineare.
# Queste feature catturano informazioni non presenti nelle serie per-canale.
# ============================================================


def extract_cross_channel_corr(X_i: np.ndarray) -> dict:
    # X_i è UN esempio: matrice shape (T, F)

    T, F = X_i.shape                    # ricava dimensioni (T time-step, F canali)
    feats = {}                          # dizionario output

    # Conversione + pulizia (evita NaN/inf)
    X_i = np.asarray(X_i, dtype=np.float64)
    X_i = np.nan_to_num(X_i, nan=0.0, posinf=0.0, neginf=0.0)

    # Se c'è 0 o 1 canale, non esistono coppie da correlare
    if F < 2:
        return feats

    # corrcoef con rowvar=False: considera ogni colonna come variabile (canale)
    # produce matrice F x F
    corr = np.corrcoef(X_i, rowvar=False)

    # Se alcuni canali sono costanti, corr potrebbe essere NaN -> li mettiamo a 0
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Estraiamo solo il triangolo superiore (a<b) per non duplicare:
    # corr(ch1,ch2) è uguale a corr(ch2,ch1)
    for a in range(F):
        for b in range(a + 1, F):
            feats[f"corr_ch{a+1}_ch{b+1}"] = float(corr[a, b])

    return feats


# ============================================================
# TRASFORMA DATASET INTERO: (N,T,F) -> DataFrame (N,K)
# ============================================================

def engineer_to_dataframe(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    # X: (N,T,F) = N esempi, ciascuno una serie multivariata T×F
    # y: (N,)    = label associata a ciascun esempio

    N, T, F = X.shape           # legge dimensioni
    rows = []                   # lista di "righe" finali (dizionari)

    # Loop su ogni esempio del dataset
    for i in range(N):
        features = {}           # conterrà TUTTE le feature di questo esempio i

        # 1) FEATURE PER CANALE:
        #    per ciascun canale f estraiamo feature su X[i,:,f] (vettore lungo T)
        for f in range(F):
            x_channel = X[i, :, f]  # serie 1D del canale f (shape T,)
            features.update(
                extract_features_1d(x_channel, prefix=f"ch{f+1}")
            )

        # 2) FEATURE CROSS-CANALE:
        #    calcoliamo correlazioni tra le colonne di X[i,:,:] (T×F)
        features.update(extract_cross_channel_corr(X[i, :, :]))

        # 3) LABEL:
        #    la label NON è dentro la matrice (T×F), è esterna (y[i])
        #    qui la aggiungiamo come colonna target per la classificazione
        try:
            features["label"] = int(y[i])  # prova a convertirla in intero (se già numerica)
        except Exception:
            features["label"] = y[i]       # se non si può, la lascia così

        # 4) Salviamo la riga completa
        rows.append(features)

    # Converte lista di dizionari in DataFrame:
    # ogni chiave del dizionario diventa una colonna
    df = pd.DataFrame(rows)

    return df


# ============================================================
# LETTURA CONFIG (percorsi e nomi file)
# ============================================================

def load_config(path: str) -> dict:
    # Apre config.yaml e lo trasforma in dict python
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# MAIN: (carica -> featurizza -> salva)
# ============================================================

def main():
    # 1) Legge argomenti
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")  # file yaml con path e nomi file
    parser.add_argument("--subdir", default="engineered")          # sottocartella dentro artifacts
    args = parser.parse_args()

    # 2) Carica configurazione
    cfg = load_config(args.config)

    # 3) Recupera cartelle dal config
    data_dir = cfg["paths"]["data_dir"]          
    artifacts_dir = cfg["paths"]["artifacts_dir"]

    # 4) Costruisce path completi di TRAIN e TEST
    train_path = os.path.join(data_dir, cfg["files"]["train_arff"])
    test_path = os.path.join(data_dir, cfg["files"]["test_arff"])

    # 5) Crea directory di output (se non esiste)
    output_dir = os.path.join(artifacts_dir, args.subdir)
    os.makedirs(output_dir, exist_ok=True)

    print("FEATURE ENGINEERING: ArticularyWordRecognition")

    # 6) Carica i dati ARFF:
    #    Xtr/Xte: (N,T,F)
    #    ytr/yte: (N,)
    Xtr, ytr = load_arff_relational(train_path)
    Xte, yte = load_arff_relational(test_path)

    # 7) Feature engineering sul TRAIN (trasforma (N,T,F) -> DataFrame tabellare)
    print("[1/2] TRAIN")
    df_train = engineer_to_dataframe(Xtr, ytr)

    # 8) Salva TRAIN in CSV
    train_csv = os.path.join(output_dir, "arwr_train_engineered.csv")
    df_train.to_csv(train_csv, index=False)
    print(f"Saved: {train_csv}  shape={df_train.shape}")

    # 9) Feature engineering sul TEST
    print("[2/2] TEST")
    df_test = engineer_to_dataframe(Xte, yte)

    # 10) Salva TEST in CSV
    test_csv = os.path.join(output_dir, "arwr_test_engineered.csv")
    df_test.to_csv(test_csv, index=False)
    print(f"Saved: {test_csv}  shape={df_test.shape}")

    # 11) Controlla che TRAIN e TEST abbiano esattamente le stesse colonne
    #     (serve perché poi in sklearn devi avere stesso schema)
    if list(df_train.columns) != list(df_test.columns):
        print("WARNING: TRAIN e TEST hanno colonne diverse")
    else:
        print("OK: schema coerente")

    print("Done.")


if __name__ == "__main__":
    main()
