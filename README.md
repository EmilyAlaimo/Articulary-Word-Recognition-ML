# Articulary Word Recognition

| | |
| --- | --- |
| **Descrizione** | Classificazione di serie temporali usando Reti Neurali Ricorrenti e ML Classico |
| **Autore** | Emily Alaimo |
| **Corso** |Machine Learning|
| **Dataset** | [ArticularyWordRecognition](https://www.timeseriesclassification.com/description.php?Dataset=ArticularyWordRecognition) |

---

### Indice

- [Introduzione](#introduzione)
- [Dataset](#dataset)
- [Requisiti](#requisiti)
- [Struttura del Codice](#struttura-del-codice)
- [Utilizzo](#utilizzo)
- [Risultati](#risultati)

---

### Introduzione

Questo progetto implementa la classificazione del **riconoscimento di parole articolatorie** usando dati di serie temporali multivariate. Il dataset è stato raccolto utilizzando un Electromagnetic Articulograph (EMA) per tracciare i movimenti degli articolatori durante la produzione del parlato. Sono stati utilizzati dodici sensori, ciascuno dei quali fornisce dati di posizione tridimensionali (X, Y, Z) con una frequenza di campionamento di 200 Hz. I sensori sono posizionati sulla lingua (quattro sensori lungo la linea mediana dalla punta al dorso, T1–T4), sulle labbra (superiore e inferiore), sulla mandibola e sulla testa. Tre sensori di riferimento sulla testa (Head Center, Head Left e Head Right), montati su una montatura di occhiali, sono utilizzati per compensare il movimento della testa e ottenere traiettorie articolatorie indipendenti dal movimento globale. Sebbene siano disponibili 36 dimensioni complessive, il dataset include solo 9 dimensioni selezionate.

Il progetto confronta due approcci:
- **Deep Learning**: modelli RNN, LSTM e GRU
- **ML Classico**: Feature engineering + SVM, kNN, Random Forest



---

### Dataset

**ArticularyWordRecognition** dall'archivio [UCR/UEA Time Series Archive](https://www.timeseriesclassification.com/description.php?Dataset=ArticularyWordRecognition).

- **Classi**: 25 parole
- **Train**: 275 campioni (11 per classe)
- **Test**: 300 campioni (12 per classe)
- **Formato**: Ogni campione è una matrice (144, 9)
  - T = 144 passi temporali
  - F = 9 caratteristiche articolatorie
- **Preprocessing**: I dati sono già z-normalizzati (media ≈ 0, std ≈ 1)

---

### Requisiti

Il progetto è basato su **Python 3.13.11**.

Installare le dipendenze:
```bash
pip install -r requirements.txt
```

Dipendenze:
- `torch` - PyTorch per il deep learning
- `numpy` - Calcolo numerico
- `scikit-learn` - Algoritmi di ML classico e metriche
- `pyyaml` - Gestione della configurazione
- `tqdm` - Barre di progresso
- `pandas` - Manipolazione dei dati (per feature engineering)

---

### Struttura del Codice

```
.
├── config/
│   └── config.yaml                    # File di configurazione
├── data/
│   ├── ArticularyWordRecognition_TRAIN.arff
│   └── ArticularyWordRecognition_TEST.arff
├── data_classes/
│   └── EMADataset.py                  # Dataset PyTorch
├── model_classes/
│   ├── RNN.py                         
│   ├── LSTM.py                        
│   └── GRU.py                        
├── utils/
│   ├── arff_loader.py                 # Parser file ARFF
│   └── metrics.py                     # Metriche di valutazione
├── artifacts/                         # Cartella generata automaticamente
│   ├── rnn_best.pt                    # Best model RNN
│   ├── lstm_best.pt                   # Best model LSTM
│   ├── gru_best.pt                    # Best model GRU
│   └── engineered/                    # Feature ingegnerizzate
│       ├── arwr_train_engineered.csv
│       └── arwr_test_engineered.csv
├── EMADataset_analysis.py             # Esplorazione del dataset
├── EMADataset_engineer.py             # Feature engineering
├── train.py                           # Script di training (Deep Learnig)
├── test.py                            # Script di testing (Deep Learning)
├── classical_ml_classification.py     # ML classico
├── requirements.txt
└── README.md
```

> **Nota**: La cartella `artifacts/` viene creata automaticamente durante l'esecuzione degli script di training e feature engineering. Contiene i modelli addestrati (`*_best.pt`) e i file CSV con le feature ingegnerizzate nella sottocartella `engineered/`.

---

### Utilizzo



#### 1. Analisi del Dataset

Esplorare la struttura e le proprietà del dataset:

```bash
python EMADataset_analysis.py
```

#### 2. Approccio Deep Learning

Addestrare modelli RNN, LSTM o GRU:

```bash
python train.py --model rnn
python train.py --model lstm
python train.py --model gru
```

Valutare sul test set:

```bash
python test.py --model rnn
python test.py --model lstm
python test.py --model gru
```

I modelli vengono salvati in `artifacts/<model>_best.pt`.

#### 3. Approccio ML Classico

Estrarre feature ingegnerizzate:

```bash
python EMADataset_engineer.py
```

Questo genera i file `artifacts/engineered/arwr_train_engineered.csv` e `artifacts/engineered/arwr_test_engineered.csv`.

Addestrare e confrontare SVM, kNN e Random Forest:

```bash
python classical_ml_classification.py
```

> [NOTE]
> Tutti gli iperparametri possono essere configurati in `config/config.yaml`.

---

### Risultati

I modelli vengono valutati usando:
- **Accuracy**: Accuratezza complessiva di classificazione
- **Macro Precision/Recall/F1**: Metriche medie su tutte le classi 

Prestazioni attese:
- **Deep Learning**: LSTM e GRU tipicamente raggiungono un'accuratezza dell'85-90%
- **ML Classico**: Random Forest spesso performa meglio tra i metodi classici

Risultati dettagliati incluse metriche per classe e matrici di confusione vengono stampati durante la valutazione.

---
