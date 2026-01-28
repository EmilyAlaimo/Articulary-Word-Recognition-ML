import torch
from torch.utils.data import Dataset

class EMADataset(Dataset):
    # Dataset per serie temporali
    # X: (N, T, F) con N campioni, T time-step, F feature
    # y: (N,) etichette di classe intere (0..C-1)

    def __init__(self, X, y):
        # Converte i dati in tensori PyTorch
        # float32 per input rete neurali, long per CrossEntropyLoss (etichette di classe)
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, T, F)
        self.y = torch.tensor(y, dtype=torch.long)     # (N,)

    def __len__(self):
        # Ritorna il numero di campioni N
        return len(self.y)

    def __getitem__(self, index):
        # Ritorna il campione index-esimo:
        # x: (T, F), y: etichetta
        return self.X[index], self.y[index]