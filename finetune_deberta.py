import torch
import itertools
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW 
from torch.utils.data import DataLoader
import datetime
import os

from TransformerLstmClassifier import BertLstmAttentionClassifier
from eval import FocusLoss, evaluate_model, compute_metrics, plot_training_graph



class ClaimEvidenceTransformerDataset():
    def __init__(self, file_path, tokenizer, max_length=256, with_label=True, load_from=None):
        self.data = []
        self.with_label = with_label

        if load_from!=None:
            print("Load dataset from:", load_from)
            loaded_dataset = torch.load(load_from, weights_only=False)
            self.__dict__.update(loaded_dataset.__dict__)
            print("Done!")
            return
        
        file = pd.read_csv(file_path)
        for idx in range(len(file)):
            claim_text, evidence_text = file["Claim"][idx], file["Evidence"][idx]
            if with_label:
                label = file["label"][idx]
            
            encoded_claim_evidence = tokenizer(
                    text=claim_text,
                    text_pair=evidence_text,
                    truncation=True, 
                    padding='max_length',
                    max_length = max_length,
                    return_tensors='pt'
                )
            
            if with_label:
                encoded_claim_evidence["labels"] = torch.tensor(label, dtype=torch.float)
            
            self.data.append(encoded_claim_evidence)
            print(f"Loading dataset...   row ({idx}/{file.shape[0]})",end="\r")
        print("\nDone!")
        
    def save_dataset(self, save_path):
        torch.save(self, save_path, pickle_protocol=5)
        print("Dataset saved to:",save_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_transformer_model(
    model,
    epochs,
    batch_size,
    start_lr,
    dropout,
    loss_fn,
    train_dataset,
    val_dataset,
    save_path,
    device
):
    
    save_path += str(datetime.datetime.now())[:-7]
    os.makedirs(save_path, exist_ok = True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=start_lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)  # 10%的steps做warmup，可自行调节
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    train_info_recorder = []
    best_f1 = 0
    for epoch in range(epochs):
        # ============== 1. 训练阶段 ==============
        model.train()
        epoch_train_losses = []
        epoch_labels = []
        epoch_preds = []
    
        batch_idx = 0
        for batch in train_loader:
            batch_idx += 1
            print(f"Epoch [{epoch+1}/{epochs}] => batch {batch_idx}/{len(train_loader)}", end='\r')
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(1).to(device)

            labels = batch['labels'].to(torch.int64).to(device)
    
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # print(labels)
            
            optimizer.zero_grad()
            
            # AutoModelForSequenceClassification 可以直接返回 loss 和 logits
            logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = loss_fn(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 记录loss、预测值，用于后续度量
            epoch_train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=-1)
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())
        
        epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        
        metrics = compute_metrics(epoch_preds, epoch_labels)
        epoch_train_acc = metrics["accuracy"]
        epoch_train_f1 = metrics["f1"]
        
        # ============== 2. 验证阶段 ==============
        epoch_val_loss, epoch_val_acc, epoch_val_f1 = evaluate_model(model, val_loader, loss_fn, device)
        

        print(f"<{str(datetime.datetime.now())[:-7]}> Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, F1: {epoch_train_f1:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {epoch_val_f1:.4f}")

        if epoch_val_f1 > best_f1:
            print("  => New Best Model! saved to:", save_path + "/best.pt")
            best_f1 = epoch_val_f1
            torch.save(model, save_path + "/best.pt")
            with open(save_path+"/best_info.txt", 'w') as file:
                info = f"Hyper-params:\nlr=>{start_lr} batch_size=>{batch_size} dropout_rate=>{dropout}"+ \
                        f"\nPerformance:\nEpoch [{epoch+1}/{epochs}], Train Loss={epoch_train_loss:.4f}, Acc={epoch_train_acc:.4f}, " + \
                        f"Dev Loss={epoch_val_loss:.4f}, Acc={epoch_val_acc:.4f}, F1={epoch_val_f1:.4f}"
                file.write(info)
            
    
        train_info_recorder.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "train_f1": epoch_train_f1,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "val_f1": epoch_val_f1
        })

    plot_training_graph(train_info_recorder, save_path)

    return train_info_recorder

model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = ClaimEvidenceTransformerDataset("./data/train.csv", tokenizer)
val_dataset = ClaimEvidenceTransformerDataset("./data/dev.csv", tokenizer)

param_grid = {
    'lr': [1e-5, 5e-6],
    'batch_size': [5, 10],
    'dropout': [0.5],
    'use_focus_loss': [True, False]
}

param_combinations = list(itertools.product(
    param_grid['lr'],
    param_grid['batch_size'],
    param_grid['dropout'],
    param_grid['use_focus_loss']
))

print(f"Got {len(param_combinations)} param combinations.")

save_path = "./models/deberta-lstm-att-girdsearch/"
best_f1 = 0
best_combs = None

for combs in param_combinations:
    start_lr, batch_size, dropout, use_focus_loss = combs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = BertLstmAttentionClassifier(model_name)
    
    model = model.to(device)

    class_weights = torch.tensor([1.38, 3.66]).to(device)
    if use_focus_loss:
        loss_fn = FocusLoss(weight=class_weights, gamma=2, device=device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"\n===> [{param_combinations.index(combs)+1}/{len(param_combinations)}] Start training with {combs}")

    epochs = 15
    train_info = train_transformer_model(
        model,
        epochs,
        batch_size,
        start_lr,
        dropout,
        loss_fn,
        train_dataset,
        val_dataset,
        save_path,
        device
    )

    val_f1 = train_info[-1]["val_f1"]

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_combs = combs
    
    del model
    torch.cuda.empty_cache()
    
print("\n\nGird Search Finished! Best Params:", best_combs, " Best F1:", best_f1)