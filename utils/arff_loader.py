import re                  
import numpy as np         


def load_arff_relational(path: str):
    # Legge un ARFF UEA "relational" e ritorna:
    # X: (N, T, F) float32  |  y: (N,) float32 (poi in train -> int64 e -1)

    with open(path, "r", encoding="utf-8", errors="replace") as f:  # apre file
        lines = f.readlines()                                       # legge tutte le righe

    data_lines = []      # conterrà solo righe dati
    in_data = False      # diventa True dopo @data

    for line in lines:
        s = line.strip()                       # pulisce spazi e newline
        if not s or s.startswith("%"):         # salta vuote e commenti
            continue
        if not in_data:                        # finché non arrivo a @data
            if s.lower().startswith("@data"):  # inizio sezione dati
                in_data = True
            continue
        data_lines.append(s)                   # salvo la riga dati

    X_list, y_list = [], []  # liste campioni e label

    for line in data_lines:
        # Formato riga: '<payload_con_\\n>',1.0
        m = re.match(r"\s*'(.+)'\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$", line)
        if m is None:
            raise ValueError(f"Riga non parsabile: {line[:200]}")

        payload = m.group(1)          # stringa con righe separate da "\n"
        label = float(m.group(2))     # label numerica (1..25 nel dataset)

        # Ogni riga nel payload = una feature lungo il tempo
        feature_rows = [r.strip() for r in payload.split("\\n") if r.strip()]

        features = []  
        for r in feature_rows:
            vals = [float(v) for v in r.split(",") if v.strip()]   # valori nel tempo
            features.append(np.array(vals, dtype=np.float32))      # (T,)

        # features è (F, T) -> trasponiamo a (T, F)
        mat_tf = np.stack(features, axis=0).T

        X_list.append(mat_tf)  # aggiungo un campione (T, F)
        y_list.append(label)   # aggiungo la label

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, T, F) con N campioni, T time-step, F feature
    y = np.array(y_list, dtype=np.float32)           # (N,)

    return X, y
