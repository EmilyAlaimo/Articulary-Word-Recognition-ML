import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(preds, refs):
    # preds, refs: liste di classi (0..C-1)

    acc = accuracy_score(refs, preds)  # accuracy globale (su tutto il set di dati (validation/test))

    # metriche macro (media non pesata sulle classi)
    p, r, f1, _ = precision_recall_fscore_support(
        refs, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
    }


#valuta il modello su un dataset (validation o test) e ritorna loss+metriche
def evaluate(model, dataloader, criterion, device):
    model.eval()  # modalità evaluation

    total_loss = 0.0          
    total_samples = 0         
    preds, refs = [], []     

    with torch.no_grad():     # disabilita gradienti 
        for x, y in dataloader:   
            x = x.to(device) # input su device
            y = y.to(device) # label su device

            logits = model(x)           # forward -> (32, 25)
            loss = criterion(logits, y) # CrossEntropy loss, confronta logits e y (etichette reali)

            bs = y.size(0)                      # batch size
            total_loss += loss.item() * bs
            total_samples += bs

            pred = torch.argmax(logits, dim=1)  # classe predetta
            preds.extend(pred.cpu().tolist())   # salva predizioni
            refs.extend(y.cpu().tolist())       # salva label vere

    metrics = compute_metrics(preds, refs)                # accuracy/precision/recall/f1
    metrics["loss"] = total_loss / max(1, total_samples)  # loss media

    return metrics
