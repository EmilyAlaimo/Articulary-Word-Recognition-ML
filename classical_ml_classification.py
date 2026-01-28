# ============================================================
# Confronto di modelli ML classici su dataset ingegnerizzato
#
# Modelli:
# - SVM (RBF)
# - kNN
# - Random Forest
#
# Il dataset NON è più una serie temporale:
# ogni riga è -> vettore di feature + label
# ============================================================

import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# ============================================================
# Caricamento dataset ingegnerizzato
# ============================================================

def load_engineered_csv(path: str):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"Colonna 'label' non trovata in {path}")
    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float64)
    return X, y


# ============================================================
# Valutazione su test set
# ============================================================

def evaluate_on_test(name: str, estimator, X_test, y_test):

    # predizione sul test set
    y_pred = estimator.predict(X_test)

    # metriche globali
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 70)
    print(f"MODEL: {name}")
    print("=" * 70)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    # report per classe (precision / recall / f1)
    report = classification_report(
        y_test, y_pred, digits=4, output_dict=True
    )

    # stampa ordinata per classe (1..N)
    class_labels = sorted([k for k in report.keys() if k.isdigit()], key=int)
    for lbl in class_labels:
        p = report[lbl]["precision"]
        r = report[lbl]["recall"]
        f1 = report[lbl]["f1-score"]
        print(f"Class {int(lbl):>2} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

    # confusion matrix
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    print("\nConfusion matrix:")
    print(cm_df)

    return {
        "model": name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1)
    }


# ============================================================
# Definizione dei modelli + griglie iperparametri
# ============================================================
def build_model_configs():

    # Lista che conterrà tutte le configurazioni dei modelli
    # Ogni elemento sarà: (nome_modello, stimatore, param_grid)
    configs = []


    # SVM 
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),      # scala le feature (media=0, std=1)
        ("clf", SVC(kernel="rbf"))         # SVM con kernel RBF (non lineare)
    ])

    svm_grid = {
        # C controlla la regolarizzazione:
        # valori alti → modello più rigido
        # valori bassi → modello più tollerante agli errori
        "clf__C": [1, 3, 10, 30],

        # gamma controlla il raggio di influenza del kernel RBF:
        # gamma alto → confini complessi
        # gamma basso → confini più morbidi
        "clf__gamma": ["scale", 0.01, 0.03],

    }

    # Aggiunge la configurazione SVM alla lista
    configs.append(("SVM (RBF)", svm_pipe, svm_grid))


    # k-Nearest Neighbors (kNN)
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),      # normalizza le feature
        ("clf", KNeighborsClassifier())    # classificatore kNN
    ])

    knn_grid = {
        # Numero di vicini considerati per la classificazione
        # pochi vicini → modello più locale
        # molti vicini → modello più stabile
        "clf__n_neighbors": [3, 5, 7, 9, 11],

        # Peso dei vicini:
        # uniform → tutti contano uguale
        # distance → i più vicini contano di più
        "clf__weights": ["uniform", "distance"],

        # Metrica di distanza usata nello spazio delle feature
        "clf__metric": ["euclidean", "manhattan"]
    }

    # Aggiunge la configurazione kNN
    configs.append(("kNN", knn_pipe, knn_grid))



   
    # Random Forest
    rf = RandomForestClassifier(
        random_state=42,   # rende i risultati riproducibili
        n_jobs=-1          # usa tutti i core disponibili (parallelismo)
    )

    rf_grid = {
        # Numero di alberi nella foresta
        # più alberi → minore varianza
        "n_estimators": [200, 300, 500],

        # Profondità massima degli alberi
        # controlla la complessità del modello
        "max_depth": [None, 20, 40],

        # Numero minimo di campioni per dividere un nodo
        "min_samples_split": [2, 5, 10],

        # Numero minimo di campioni in una foglia
        # valori più alti → modello più regolarizzato
        "min_samples_leaf": [1, 2, 4],

        # Numero di feature considerate a ogni split
        # serve a decorrelare gli alberi
        "max_features": ["sqrt", "log2"]
    }

    # Aggiunge la configurazione Random Forest
    configs.append(("Random Forest", rf, rf_grid))

    # Ritorna tutte le configurazioni
    return configs

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Flusso:
      1) carica train/test
      2) definisce CV stratificata
      3) GridSearch per ogni modello
      4) valutazione su test
      5) confronto finale
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="artifacts/engineered/arwr_train_engineered.csv")
    parser.add_argument("--test_csv", default="artifacts/engineered/arwr_test_engineered.csv")
    args = parser.parse_args()

    # caricamento dati
    X_train, y_train = load_engineered_csv(args.train_csv)
    X_test, y_test = load_engineered_csv(args.test_csv)

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test : X={X_test.shape}, y={y_test.shape}")

    # cross-validation stratificata
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # costruzione modelli
    configs = build_model_configs()
    results = []

    # ciclo sui modelli
    for name, estimator, param_grid in configs:

        print("\n" + "=" * 70)
        print(f"GRID SEARCH: {name}")
        print("=" * 70)

        # GridSearchCV:
        # - tuning su train
        # - refit automatico sul train completo
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        gs.fit(X_train, y_train)

        print("Best params:", gs.best_params_)
        print(f"Best CV Macro F1: {gs.best_score_:.4f}")

        # valutazione finale su test
        res = evaluate_on_test(f"{name} [GridSearch]", gs, X_test, y_test)
        results.append(res)

    # --------------------------------------------------------
    # Summary finale
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r['model']:22s} | Accuracy: {r['accuracy']:.4f} | Macro F1: {r['macro_f1']:.4f}")

    best_acc = max(results, key=lambda d: d["accuracy"])
    best_f1 = max(results, key=lambda d: d["macro_f1"])

    print("\nBEST MODEL BY ACCURACY:", best_acc["model"])
    print("BEST MODEL BY MACRO F1:", best_f1["model"])


if __name__ == "__main__":
    main()
