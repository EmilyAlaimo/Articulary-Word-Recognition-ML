import torch


class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = torch.nn.Linear(hidden_dim * direction_factor, num_classes)


    def forward(self, x):                             #x: (Batch size (32), lunghezza sequenza (144), input_dim (9))
        if self.bidirectional:
            _, (h_n, _) = self.lstm(x)                # h_n: (2*num_layers (per ogni layer due direzioni), 32, 128)
            h = torch.cat([h_n[-2], h_n[-1]], 1)      # (32, 2*128) stato finale-> h_n[-2] e h_n[-1] stati finali (da destra verso sinistra e viceversa)
        else:
            _, (h_n, _) = self.lstm(x)                # h_n: (num_layers, 32, 128)
            h = h_n[-1]                               # h: (32, 128) stato finale (solo 1 nel caso unidirezionale)

        logits = self.fc(h)                           # self.fc: Linear(2*128 -> 25) se è bidirezionale oppure (128 -> 25)
        
        return logits                                 # (32, 25) punteggi di classe
