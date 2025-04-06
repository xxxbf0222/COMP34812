import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from LstmAttentionClassifier import LstmAttentionClassifier
from TransformerLstmClassifier import BertLstmAttentionClassifier

class FocusLoss(nn.Module):
    def __init__(self, weight, gamma, device):
        super(FocusLoss, self).__init__()
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(self.weight).to(device)
        self.weight = weight
        self.gamma = gamma
    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets)
        pt = torch.exp(-ce_loss)
        at = self.weight.gather(0, targets)
        loss = at * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total = 0
    epoch_labels = []
    epoch_preds = []

    for batch in dataloader:
        if type(model) is LstmAttentionClassifier:
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

            
            logits = model(claim_ids, claim_lens, evidence_ids, evidence_lens)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            
            total += labels.size(0)

            preds = torch.argmax(logits, dim=-1)
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())
        elif type(model) is BertLstmAttentionClassifier:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
            labels = batch['labels'].to(torch.int64).to(device)
            logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            
            preds = torch.argmax(logits, dim=-1)
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / total
    metrics = compute_metrics(epoch_preds, epoch_labels)
    acc = metrics["accuracy"]
    f1 = metrics["f1"]
    
    return avg_loss, acc, f1

def model_predict(model, dataset, batch_size=150, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            if type(model) is LstmAttentionClassifier:
                claim_ids, claim_lens, evidence_ids, evidence_lens = batch["claim"], \
                                                                        batch["claim_length"], \
                                                                        batch["evidence"],\
                                                                        batch["evidence_length"]
                claim_ids = claim_ids.to(device)
                claim_lens = claim_lens.to(device)
                evidence_ids = evidence_ids.to(device)
                evidence_lens = evidence_lens.to(device)
                logits = model(claim_ids, claim_lens, evidence_ids, evidence_lens)
            elif type(model) is BertLstmAttentionClassifier:
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
                logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
    
    return all_preds

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

def plot_training_graph(train_info_recorder, save_path):
    epochs = [row["epoch"] for row in train_info_recorder]
    all_train_loss = [row["train_loss"] for row in train_info_recorder]
    all_val_loss = [row["val_loss"] for row in train_info_recorder]
    all_train_acc = [row["train_acc"] for row in train_info_recorder]
    all_val_acc = [row["val_acc"] for row in train_info_recorder]

    # === Train vs Val Loss Graph ===
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, all_train_loss, label='Train Loss', marker='o', linestyle='-', color='b')
    plt.plot(epochs, all_val_loss, label='Val Loss', marker='s', linestyle='-', color='r')
    plt.title('Train Loss vs Val Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(save_path+'/train_val_loss.png', dpi=300, bbox_inches='tight')
    plt.clf()  # 清空当前 figure

    # === Train vs Val Acc Graph ===
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, all_train_acc, label='Train Acc', marker='o', linestyle='-', color='b')
    plt.plot(epochs, all_val_acc, label='Val Acc', marker='s', linestyle='-', color='r')
    plt.title('Train Acc vs Val Acc', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(save_path+'/train_val_acc.png', dpi=300, bbox_inches='tight')
    plt.clf()

    plt.close()