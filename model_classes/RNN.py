import torch


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout):
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):           # x: (Batch size (32), lunghezza sequenza (144), input_dim (9))
        out, _ = self.rnn(x)        # calcola tutti i passi nel tempo e attraverso i due layer
        h = out[:, -1, :]           # (32, 128) stato finale -> tiene l'ultimo stato per la classificazione
        logits = self.fc(h)         # (32, 25) punteggi di classe -> per ogni sequenza produce 25 punteggi
        return logits
