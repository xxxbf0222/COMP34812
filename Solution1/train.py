import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
import os

from Solution1.data.ClaimEvicenceDataset import ClaimEvidenceLstmDataset
from Solution1.LstmAttentionClassifier import LstmAttentionClassifier
from eval import evaluate_model, compute_metrics, plot_training_graph, FocusLoss
from Solution1.train_config import train_configs

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Trains the LSTM-attention model for one epoch. Returns average loss, accuracy, and F1.
    """
    model.train()
    total_loss = 0.0
    total = 0
    epoch_labels = []
    epoch_preds = []

    batch_idx = 0
    # We'll retrieve epoch and epochs from the outer scope inside the loop, so pass them in or use a closure
    for batch in dataloader:
        batch_idx += 1
        
        # Extract claim/evidence sequences, lengths, and labels, move to training device
        claim_ids = batch["claim"].to(device)
        claim_lens = batch["claim_length"].to(device)
        evidence_ids = batch["evidence"].to(device)
        evidence_lens = batch["evidence_length"].to(device)
        labels = batch["label"].to(device)
        
        # Remove gradient
        optimizer.zero_grad()
        
        # Forward pass to obtain logits
        logits = model(claim_ids, claim_lens, evidence_ids, evidence_lens)
        
        # Calculate loss and backpropagation
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        # Accumulate total_loss for average
        total_loss += loss.item() * claim_ids.size(0)
        total += labels.size(0)

        # Record predictions & ground truth for metric calculations
        preds = torch.argmax(logits, dim=-1)
        epoch_labels.extend(labels.cpu().numpy())
        epoch_preds.extend(preds.cpu().numpy())

    # Compute metrics for this epoch
    metrics = compute_metrics(epoch_preds, epoch_labels)
    acc = metrics["accuracy"]
    f1 = metrics["f1"]
    
    return total_loss / total, acc, f1

def current_datetime():
    """
    Returns a string of the current datetime truncated to seconds.
    used for naming folders or files with unique IDs.
    """
    return str(datetime.datetime.now())[:-7]

if __name__ == "__main__":

    # Load hyperparameters from train_configs.py
    epochs = train_configs["epochs"]
    batch_size = train_configs["batch_size"]
    start_lr = train_configs["start_lr"]
    hidden_size = train_configs["hidden_size"]
    dropout_rate = train_configs["dropout_rate"]
    use_focus_loss = train_configs["use_focus_loss"]
    save_path = train_configs["save_path"] + current_datetime()
    device = train_configs["device"]

    # path to save the training result
    os.makedirs(save_path, exist_ok=True)

    # Build training dataset & vocab & embedding mat
    train_dataset = ClaimEvidenceLstmDataset("./data/train.csv")
    vocab = train_dataset.vocab
    embedding_mat = train_dataset.embedding_mat

    # Validation dataset based on vocab build from training set
    val_dataset = ClaimEvidenceLstmDataset("./data/dev.csv", dataset_type="validation", vocab=vocab)

    # Define selected loss function (either FocusLoss or weighted CrossEntropy)
    class_weights = torch.tensor([1.374, 3.674]).to(device)
    if use_focus_loss:
        loss_fn = FocusLoss(weight=class_weights, gamma=1.5, device=device)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Create DataLoaders for training/validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False) # batch_size=100 for faster evaluation

    # Initialize LstmAttentionClassifier with the loaded vocabulary and embedding
    model = LstmAttentionClassifier(
        len(vocab),
        embed_dim=len(embedding_mat[0]),
        hidden_size=hidden_size,
        num_classes=2,
        embedding_mat=embedding_mat,
        train_embedding=False,
        dropout_rate=dropout_rate
    )
    model.to(device)

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    # Use a scheduler to reduce LR on plateau (based on val F1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=0,
    )

    # Log chosen hyperparams
    hyper_params = {
        "batch_size": batch_size,
        "epochs": epochs,
        "start_lr": start_lr,
        "hidden_size": hidden_size,
        "dropout_rate": dropout_rate,
        "use_focus_loss": use_focus_loss,
    }
    
    print(f"\n[{current_datetime()}] Start training with {hyper_params}")
    
    train_info_recorder = []
    best_score = 0

    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_acc, train_f1 = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        
        # Evaluate on validation set
        with torch.no_grad():
            dev_loss, dev_acc, dev_f1 = evaluate_model(model, val_dataloader, loss_fn, device)

        # Use F1 to step the scheduler
        scheduler.step(dev_f1)
        
        # report and record training result in this epoch
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{current_datetime()}] Epoch {epoch+1}/{epochs}, lr_now={lr_now}, "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
              f"Dev Loss={dev_loss:.4f}, Acc={dev_acc:.4f}, F1={dev_f1:.4f}")
        
        train_info_recorder.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": dev_loss,
            "val_acc": dev_acc
        })
        
        # Save model if F1 is improved
        if best_score < dev_f1:
            best_score = dev_f1
            print(f"    New best model! Saved to: {save_path}/best.pt")
            torch.save(model.cpu(), save_path + "/best.pt")
            model.to(device)
            with open(save_path + "/best_info.txt", 'w') as file:
                info = (f"Hyper-params:\n{hyper_params}"
                        f"\nPerformance:\nEpoch [{epoch+1}/{epochs}], "
                        f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                        f"Dev Loss={dev_loss:.4f}, Acc={dev_acc:.4f}, F1={dev_f1:.4f}")
                file.write(info)

    # Plot training progress
    plot_training_graph(train_info_recorder, save_path)

    # Free CUDA memory to reduce CudaOutOfMemoryError :)
    del model
    torch.cuda.empty_cache()
