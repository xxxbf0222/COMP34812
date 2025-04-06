import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
import os

from ClaimEvicenceDataset import ClaimEvidenceLstmDataset
from LstmAttentionClassifier import LstmAttentionClassifier
from eval import evaluate_model, compute_metrics, plot_training_graph, FocusLoss

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    epoch_labels = []
    epoch_preds = []

    for batch in dataloader:
        claim_ids, claim_lens, evidence_ids, evidence_lens, labels = batch["claim"], \
                                                                     batch["claim_length"], \
                                                                     batch["evidence"],\
                                                                     batch["evidence_length"],\
                                                                     batch["label"]
        claim_ids = claim_ids.to(device)
        claim_lens = claim_lens.to(device)
        evidence_ids = evidence_ids.to(device)
        evidence_lens = evidence_lens.to(device)
        labels = labels.to(device)
        # print(claim_lens)
        # print(evidence_lens)
        optimizer.zero_grad()
        logits = model(claim_ids, claim_lens, evidence_ids, evidence_lens)
        # loss = nn.CrossEntropyLoss(weight=torch.tensor([0.273, 0.727]).to(device))(logits, labels)
        # print(logits, labels)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * claim_ids.size(0)
        total += labels.size(0)

        preds = torch.argmax(logits, dim=-1)
        epoch_labels.extend(labels.cpu().numpy())
        epoch_preds.extend(preds.cpu().numpy())

    metrics = compute_metrics(epoch_preds, epoch_labels)
    acc = metrics["accuracy"]
    f1 = metrics["f1"]
    
    return total_loss / total, acc, f1
    
if __name__ == "__main__":

    batch_size = 10
    epochs = 30
    start_lr = 1e-4
    hidden_size = 256
    dropout_rate = 0.5
    use_focus_loss = False
    save_path = "./models/attLstm"+str(datetime.datetime.now())

    os.mkdir(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = ClaimEvidenceLstmDataset(load_from="./data/pt/train_dataset.pt")
    vocab = train_dataset.vocab
    embedding_mat = train_dataset.embedding_mat

    val_dataset = ClaimEvidenceLstmDataset(load_from="./data/pt/dev_dataset.pt")

    class_weights = torch.tensor([1.0,1.0]).to(device)
    if use_focus_loss:
        loss_fn = FocusLoss(weight=class_weights, gamma=1.5, device=device)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    model = LstmAttentionClassifier(len(vocab),
                                    embed_dim=len(embedding_mat[0]),
                                    hidden_size=hidden_size,
                                    num_classes=2,
                                    embedding_mat=embedding_mat,
                                    train_embedding=False,
                                    dropout_rate=dropout_rate)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=0,

    )

    hyper_params = {
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "start_lr": start_lr,
                    "hidden_size": hidden_size,
                    "dropout_rate": dropout_rate,
                    "use_focus_loss": use_focus_loss,
                }
    
    print(f"\nStart training with {hyper_params}")
    
    train_info_recorder = []
    best_score = 0
    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        scheduler.step(train_f1)
        with torch.no_grad():
            dev_loss, dev_acc, dev_f1 = evaluate_model(model, val_dataloader, loss_fn, device)
        
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs}, lr_now={lr_now}, Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, Dev Loss={dev_loss:.4f}, Acc={dev_acc:.4f}, F1={dev_f1:.4f}")
        
        train_info_recorder.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": dev_loss,
            "val_acc": dev_acc
        })
        
        if best_score < dev_f1:
            best_score = dev_f1
            print(f"    New best model! Saved to:", save_path+"/best.pt")
            torch.save(model, save_path+"/best.pt")
            with open(save_path+"/best_info.txt", 'w') as file:

                info = f"Hyper-params:\n{hyper_params}"+ \
                        f"\nPerformance:\nEpoch [{epoch+1}/{epochs}], Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, " + \
                        f"Dev Loss={dev_loss:.4f}, Acc={dev_acc:.4f}, F1={dev_f1:.4f}"
                file.write(info)

    plot_training_graph(train_info_recorder, save_path)
    

