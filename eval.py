import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from Solution1.data.ClaimEvicenceDataset import ClaimEvidenceLstmDataset
from Solution2.data.ClaimEvidenceTransformerDataset import ClaimEvidenceTransformerDataset
from Solution1.LstmAttentionClassifier import LstmAttentionClassifier
from Solution2.TransformerLstmClassifier import BertLstmAttentionClassifier

class FocusLoss(nn.Module):
    '''
    A custom loss class implementing focal loss, emphasizing hard-to-classify samples.

    With class weight implemented.
    '''
    def __init__(self, weight, gamma, device):
        super(FocusLoss, self).__init__()
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(self.weight).to(device)
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets)
        pt = torch.exp(-ce_loss)
        at = self.weight.gather(0, targets) # get class weight for each sample
        loss = at * (1 - pt) ** self.gamma * ce_loss # focus loss formula
        return loss.mean()

def evaluate_model(model, dataloader, loss_fn, device):
    """
    Evaluates the given model on validation set dataLoader, returning average loss, accuracy, and F1-score.

    Use loss_fn to calculate validation loss
    """
    model.eval()
    # recorder for all batch
    total_loss = 0.0
    total = 0
    epoch_labels = []
    epoch_preds = []

    for batch in dataloader:
        # Distinguish between LSTM-based model and Transformer-based model
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

            # Forward pass through the LSTM-attention classifier
            logits = model(claim_ids, claim_lens, evidence_ids, evidence_lens)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            preds = torch.argmax(logits, dim=-1) # get predict label based on largest logit index in sample
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())

        elif type(model) is BertLstmAttentionClassifier:
            # For Transformer-based model, fetch input_ids, mask, and token_type_ids
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
            labels = batch['labels'].to(torch.int64).to(device)

            # Forward pass through the Transformer + LSTM model
            logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            preds = torch.argmax(logits, dim=-1) # get predict label based on largest logit index in sample
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())

    # comput metircs
    avg_loss = total_loss / total
    metrics = compute_metrics(epoch_preds, epoch_labels)
    acc = metrics["accuracy"]
    f1 = metrics["f1"]

    return avg_loss, acc, f1

def model_predict(model, dataset, batch_size=150, device='cpu'):
    """
    Generates predictions for a given dataset using the specified model and batch size.

    Used for test dataset.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)
    all_preds = []

    with torch.no_grad():
        batch_idx = 0
        for batch in dataloader:
            batch_idx += 1
            print(f"Predicting... => batch {batch_idx}/{len(dataloader)}", end='\r')

            # Data fetching logic for LSTM-attention
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

            # Data fetching logic for Bert-based model
            elif type(model) is BertLstmAttentionClassifier:
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
                logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
    print("\nDone!")
    return all_preds

def model_predict_text(model, claim, evidence, preprocessor):
    """
    Builds a small dataset with a single (claim, evidence) entry,
    and uses model_predict to get a single prediction.
    
    Used for letting model predict a single claim-evidence text pair
    """
    dataset = pd.DataFrame([[claim, evidence]], columns=["Claim","Evidence"])
    # Automatically detect the model type and use corrensponding dataset class
    if type(model) is LstmAttentionClassifier:
        dataset = ClaimEvidenceLstmDataset(dataset, dataset_type='test', vocab=preprocessor)
    elif type(model) is BertLstmAttentionClassifier:
        dataset = ClaimEvidenceTransformerDataset(dataset, tokenizer=preprocessor, dataset_type='test')
    result = model_predict(model, dataset)[0]
    return bool(result)

def predict_to_csv(labels, save_path):
    """
    Writes the list of predicted labels into a CSV file at the specified path.
    """
    output_dataset = [labels[idx] for idx in range(len(labels))]
    pd.DataFrame(output_dataset, columns = ['prediction']).to_csv(save_path,index=False)
    print("Prediction saved to:", save_path)

def compute_metrics(preds, labels):
    """
    Computes accuracy and weighted F1 score for the predictions.
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

def plot_training_graph(train_info_recorder, save_path):
    """
    Generates and saves training vs validation Loss and Accuracy plots from the given train_info_recorder.
    """
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
    plt.clf()

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
