import torch
import itertools
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW 
from torch.utils.data import DataLoader
import datetime
import os

from Solution2.data.ClaimEvidenceTransformerDataset import ClaimEvidenceTransformerDataset
from Solution2.TransformerLstmClassifier import BertLstmAttentionClassifier
from Solution2.train_config import train_configs
from eval import FocusLoss, evaluate_model, compute_metrics, plot_training_graph

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
    """
    Train a Transformer-based (BERT) model (with LSTM+Attention) using the specified hyperparameters
    and record the training & validation process (loss, accuracy, F1) for each epoch.
    """
    
    # Append a timestamp to the save_path for organization
    save_path += current_datetime()
    os.makedirs(save_path, exist_ok=True)
    
    # Create DataLoaders based on chosen for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize optimizer (AdamW) with a starting LR
    optimizer = AdamW(model.parameters(), lr=start_lr)
    
    # Calculate total steps (for LR scheduler) = #batches * epochs
    total_steps = len(train_loader) * epochs
    # Warmup steps as 10% of total
    warmup_steps = int(0.1 * total_steps)
    
    # Create a linear schedule with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Keep track of training info across epochs
    train_info_recorder = []
    best_f1 = 0
    
    # ======== Main training loop ========
    for epoch in range(epochs):
        # Switch model to training mode
        model.train()
        epoch_train_losses = []
        epoch_labels = []
        epoch_preds = []
    
        batch_idx = 0
        for batch in train_loader:
            batch_idx += 1
            print(f"Epoch [{epoch+1}/{epochs}] => batch {batch_idx}/{len(train_loader)}", end='\r')
            
            # Unpack batch elements into input tensors
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
            labels = batch['labels'].to(torch.int64).to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass through the model and calculate loss
            logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = loss_fn(logits, labels)
            
            # loss Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Collect training loss & predictions
            epoch_train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=-1)
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())
        
        # Compute average training loss in this epoch
        epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        
        # Evaluate training accuracy & F1 using the collected preds
        metrics = compute_metrics(epoch_preds, epoch_labels)
        epoch_train_acc = metrics["accuracy"]
        epoch_train_f1 = metrics["f1"]
        
        # ======== Evaluation ========
        epoch_val_loss, epoch_val_acc, epoch_val_f1 = evaluate_model(model, val_loader, loss_fn, device)
        
        # ======== Report Result ========
        print(f"<{current_datetime()}> Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, F1: {epoch_train_f1:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {epoch_val_f1:.4f}")

        # Check if new best F1 is found, then save model and its hyper-params
        if epoch_val_f1 > best_f1:
            print("  => New Best Model! saved to:", save_path + "/best.pt")
            best_f1 = epoch_val_f1
            torch.save(model.cpu(), save_path + "/best.pt")
            model.to(device)
            with open(save_path+"/best_info.txt", 'w') as file:
                info = (f"Hyper-params:\nlr=>{start_lr} batch_size=>{batch_size} dropout_rate=>{dropout}"
                        f"\nPerformance:\nEpoch [{epoch+1}/{epochs}], Train Loss={epoch_train_loss:.4f}, "
                        f"Acc={epoch_train_acc:.4f}, Dev Loss={epoch_val_loss:.4f}, "
                        f"Acc={epoch_val_acc:.4f}, F1={epoch_val_f1:.4f}")
                file.write(info)
            
        # Record training info for plotting/analysis
        train_info_recorder.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "train_f1": epoch_train_f1,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "val_f1": epoch_val_f1
        })

    # Plot the training process (loss/acc) over epochs
    plot_training_graph(train_info_recorder, save_path)

    return train_info_recorder

def current_datetime():
    """
    Returns the current timestamp string truncated to seconds, 
    used for naming folders or files with unique IDs.
    """
    return  str(datetime.datetime.now())[:-7]

if __name__ == '__main__':
    # Load training configurations
    model_name = train_configs["bert_model"]
    epochs = train_configs["epochs"]
    batch_size = train_configs["batch_size"]
    start_lr = train_configs["start_lr"]
    dropout_rate = train_configs["dropout_rate"]
    use_focus_loss = train_configs["use_focus_loss"]
    save_path = train_configs["save_path"] + current_datetime()
    device = train_configs["device"]

    # Initialize the tokenizer of selected BERT model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create and preprocessing training and validation datasets
    train_dataset = ClaimEvidenceTransformerDataset("./data/train.csv", tokenizer)
    val_dataset = ClaimEvidenceTransformerDataset("./data/dev.csv", tokenizer, dataset_type="validation")

    # record best f1 score in epochs
    best_f1 = 0

    # Build the BertLstmAttentionClassifier model
    model = BertLstmAttentionClassifier(model_name)
    model = model.to(device)

    # Set class weights for imbalance, then define the selected loss function
    class_weights = torch.tensor([1.374, 3.674]).to(device)
    if use_focus_loss:
        loss_fn = FocusLoss(weight=class_weights, gamma=2, device=device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Log the chosen hyper-params
    hyper_params = {
        "bert_model": model_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "start_lr": start_lr,
        "dropout_rate": dropout_rate,
        "use_focus_loss": use_focus_loss,
    }
    
    print(f"\n[{current_datetime()}] Start training with {hyper_params}")
    # Train and get the record of metrics across epochs
    train_info = train_transformer_model(
        model,
        epochs,
        batch_size,
        start_lr,
        dropout_rate,
        loss_fn,
        train_dataset,
        val_dataset,
        save_path,
        device
    )

    # Retrieve the latest validation F1
    val_f1 = train_info[-1]["val_f1"]
    if val_f1 > best_f1:
        best_f1 = val_f1
    
    # Free CUDA memory to reduce CudaOutOfMemoryError :)
    del model
    torch.cuda.empty_cache()

    