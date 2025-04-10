# Evidence Detection for Claim-Evidence Pairs

This repository contains two different solutions for a claim-evidence classification task, which determines whether a piece of evidence is relevant to a given claim. Both approaches implement different neural network architectures, where the first one is traditional deep-learning appreach without Transformer architecture, and the second introduced Transformer.

[Access model file here](https://livemanchesterac-my.sharepoint.com/:f:/g/personal/fan_mo-4_student_manchester_ac_uk/ElF2ODHq9ltGpS6X-DKbXGMBiuHY-lcjxRYb2Mbyz2hiNA?e=kAXZmw )

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Solution 1: LSTM-Attention Classifier](#solution-1-lstm-attention-classifier)
  - [Architecture](#architecture)
  - [Solution 1 Dataset Class](#solution-1-dataset-class)
  - [Solution 1 Model Training](#solution-1-model-training)
- [Solution 2: DeBERTa-LSTM-Attention Classifier](#solution-2-deberta-lstm-attention-classifier)
  - [Architecture](#architecture-1)
  - [Solution 2 Dataset Class](#claimevidencetransformerdataset)
  - [Solution 2 Model Training](#solution-2-model-training)
- [eval.py – Evaluation Utilities and Interfaces](#evalpy--evaluation-utilities-and-interfaces)
- [Use the Model](#use-the-model)
- [Results](#results)
- [References](#references)


## Project Overview

This project implements two different approaches to the evidence detection task:

1. **LSTM-Attention Classifier**: A BiLSTM-based model inspired by the [ESIM architecture](https://arxiv.org/pdf/1609.06038), using attention mechanisms and no pre-trained transformers.
2. **DeBERTa-LSTM-Attention Classifier**: A hybrid model that based on [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) Transformer and combined LSTM and attention layers.

Both models achieve strong performance on the task, with the DeBERTa-based model achieving superior results.

## Repository Structure
```
.
├── Solution1/                      # LSTM-Attention solution
│   ├── LstmAttentionClassifier.py  # Solution 1 model implementation
│   ├── data/
│   │   ├── ClaimEvidenceDataset.py # Solution 1 Dataset class, with data-preprocessing implemented
│   │   ├── POStokenizer.py         # Custom tokenizer with POS tagging
│   │   └── SimpleTokenizer.py      # Alternative simple tokenizer
│   ├── gird_search.py              # Hyperparameter optimization
│   ├── train.py                    # Solution 1 model training script
│   └── train_config.py             # Configuration file for model training
│
├── Solution2/                      # DeBERTa-LSTM-Attention solution
│   ├── TransformerLstmClassifier.py      # Solution 2 model implementation
│   ├── data/
│   │   └── ClaimEvidenceTransformerDataset.py  # Solution 2 Dataset class, with data-preprocessing implemented
│   ├── gird_search.py              # Hyperparameter optimization
│   ├── train.py                    # Solution 2 model training script
│   └── train_config.py             # Configuration file for model training
│
├── eval.py                         # Evaluation utilities
├── s1_demo.ipynb                   # Demo notebook for Solution 1
└── s2_demo.ipynb                   # Demo notebook for Solution 2
```

## Solution 1: LSTM-Attention Classifier

### Architecture

This solution implements a variant of the Enhanced Sequential Inference Model (ESIM) architecture, originally designed for natural language inference, adapted for evidence detection.

Key components:
1. **Word Embeddings**: Defaultly using word2vec-google-news-300 embeddings
2. **Encoder**: BiLSTM to encode both claim and evidence sequences 
3. **Attention Mechanism**: Cross-attention between (encoded) claim and evidence
4. **Enhancement Layer**: Element-wise difference and multiplication
5. **Composition**: Second BiLSTM layer for compositional inference
6. **Pooling**: Max and average pooling operations
7. **Classification**: Full-connection layers for classification with dropout

```python
# Simplified architecture flow
claim_encoded = self._encode(claim_embed, claim_lens)
evidence_encoded = self._encode(evidence_embed, evidence_lens)

# Cross-attention
attended_claim, attended_evidence = self._softmax_attention(claim_mask, 
                                                           claim_encoded, 
                                                           evidence_mask, 
                                                           evidence_encoded)

# Difference enhancement
claim_diff = claim_encoded - attended_claim
claim_mul = claim_encoded * attended_claim
claim_combined = torch.cat([claim_encoded, attended_claim, claim_diff, claim_mul], dim=-1)

```

### **Solution 1 Dataset Class**

**`ClaimEvidenceDataset.ClaimEvidenceLstmDataset`** is the custom dataset class used in **Solution 1** (the LSTM-Attention approach) to handle claim-evidence pairs. It manages:

- **Vocabulary building** (for the training set)  
- **Tokenization** of input claim and evidence texts  
- **Loading** pretrained word embeddings (Google News vectors in this example)  
- Converting texts into **padded sequences** of indices (for LSTM/ESIM-style models)

Below is a breakdown of the main parameters, methods, and how the dataset is used in the training script.

#### Main Parameters

```python
ClaimEvidenceLstmDataset(
    dataframe=None, 
    dataset_type="train",
    vocab=None, 
    tokenizer=POS_tokenizer(),
    pre_trained_embedding=None,
    load_from=None
)
```

1. **`dataframe`**  
   - A path to a CSV file containing columns (e.g. "Claim", "Evidence", "label") – or a pandas DataFrame object loaded manually.  
   - If you set `dataset_type="train"`, this CSV is used to build your vocabulary and word embedding matrix from scratch.

2. **`dataset_type`**  
   - One of: **"train"**, **"validation"**, or **"test"**.  
   - In **"train"** mode, the class will automatically build a vocabulary from the text data and store it, as well as gather label data if present.  
   - In **"validation"/"test"** mode, it requires an existing `vocab` to be provided, so it does not attempt to modify or expand the vocabulary, as well as knows how to encode texts into input sequences.

3. **`vocab`**  
   - If you already have a vocabulary dictionary (token -> index), pass it in here.  
   - This parameter is **required** if `dataset_type` is **not** "train".  
   - In "train" mode, if you do not provide one, it starts a new vocabulary starting with `{"<unk>": 1, "<pad>": 0}`.

4. **`tokenizer`**  
   - Custo tokenizer instance (by default, `POS_tokenizer()`), which splits the claim and evidence texts into tokens.  
   - You can replace it with any other custom tokenizer if desired.

5. **`pre_trained_embedding`**  
   - A dictionary or map of (word -> vector) pretrained embeddings used for building embeddings in in training.  
   - By default, if this is not given **and** `dataset_type="train"`, it will load a default pretrained Google News embedding locally.

6. **`load_from`**  
   - If provided, the class will **load** a pre-serialized (pre-processed) dataset from this path (via `torch.load`), bypassing the normal CSV reading, tokenization, or vocabulary building logic.

---

#### Key Internal Methods

1. **`tokenize(text)`**  
   - Uses the provided `tokenizer` to split the raw text into tokens.

2. **`expand_vocab(tokens)`**  
   - For each token, if the token exists in the pretrained embedding but is not yet in `self.vocab`, it adds it to the vocabulary.  
   - Only called in **"train"** mode.

3. **`build_embedding_mat()`**  
   - Builds a torch Tensor (size `[vocab_size, embedding_dim]`) where each row corresponds to a token in the vocabulary.  
   - If a token is unknown (`<unk>` or `<pad>`), it is given a random vector; otherwise, it uses the pretrained embedding vector.

4. **`text_to_seq_tensor(text)`**  
   - Converts a tokenized text into a sequence of integers based on `self.vocab`.

5. **`pad_sequences(seqs)`**  
   - Calls PyTorch’s `pad_sequence` to pad variable-length sequences to the same length.  
   - Returns `(padded_seq, lengths)` – where `padded_seq` is a single tensor of shape `[batch_size, max_length]`.

6. **`pad_seq_with_zero_length()`**  
   - If any sequence ends up with length zero after tokenize, it replaces the first token with `<unk>` and ensures the length is recorded as 1.  
   - Helps avoid empty-sequence errors in the LSTM.

7. **`__getitem__` / `__len__`**  
   - Standard dataset indexing: returns a dictionary containing:
     ```python
     {
       "id": index,
       "claim": padded_claim_sequence, 
       "claim_length": length_of_claim_seq, 
       "evidence": padded_evidence_sequence,
       "evidence_length": length_of_evidence_seq,
       "label": label_if_available
     }
     ```

---

#### Usage in the Training Script

In **`Solution1/train.py`**, this dataset is initialised as follows:

1. **Initialization**:
   ```python
   train_dataset = ClaimEvidenceLstmDataset("./data/train.csv")
   val_dataset = ClaimEvidenceLstmDataset("./data/dev.csv", dataset_type="validation", vocab=train_dataset.vocab)
   ```
   - The **training** dataset is built from `"./data/train.csv"`, automatically creating a vocabulary and embedding matrix.  
   - The **validation** dataset uses the same vocabulary (`vocab=train_dataset.vocab`) but doesn’t modify or expand it.

2. **Embedding & Vocab**:
   - During training dataset initialization, the class also loads or builds the `embedding_mat`.  
   - This matrix is then passed to the LSTM-based model constructor to initialize the embedding layer.

3. **DataLoader**:
   ```python
   train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   ```
   - The dataset is given to a PyTorch `DataLoader` for batching and shuffling.

4. **Model Training**:
   - Inside the training loop, each batch from `train_dataloader` provides a dictionary with padded claim/evidence sequences, lengths, and labels.  
   - These are then fed into the **LstmAttentionClassifier** (or any LSTM model) for forward/backward passes.


### Solution 1 Model Training

In **Solution1**, the core training logic is in `train.py`. It orchestrates:

1. **Loading training and validation datasets**  
2. **Instantiating** the LSTM-Attention model class 
3. **Reading hyperparameters** from `train_config.py`  
4. **Running** an epoch-based training loop  
5. **Saving** the best model at each epoch (evaluated by development dataset)

---

#### `train_config.py`

```python
train_configs = {
    "epochs": 30,
    "batch_size": 5,
    "start_lr": 0.0004,
    "hidden_size": 128,
    "dropout_rate": 0.3,
    "use_focus_loss": True,
    "save_path": "./Solution1/models/",
    "device": "cuda",
}
```

1. **`epochs`**  
   - Integer. Number of training epochs.

2. **`batch_size`**  
   - Integer. Number of samples per mini-batch in DataLoader.

3. **`start_lr`**  
   - Float. Initial learning rate for the optimizer.

4. **`hidden_size`**  
   - Integer. The hidden dimension size for the BiLSTM layers in `LstmAttentionClassifier`.

5. **`dropout_rate`**  
   - Float in [0,1]. Dropout probability used to mitigate overfitting.

6. **`use_focus_loss`**  
   - Boolean. If **True**, uses a custom **FocusLoss** instead of standard CrossEntropy. FocusLoss emphasizes hard-to-classify examples.

7. **`save_path`**  
   - String. Directory where trained models and logs are saved.

8. **`device`**  
   - String: either `"cuda"` or `"cpu"`. Specifies computation device.

---

#### `train.py` – Usage

**`train.py`** reads the configs from **`train_config.py`**, sets up the model, loads the dataset, and starts training. Below is an outline of how to run it:

1. **Check or modify** hyperparameters in `train_config.py`.
2. **Run**:  
   ```bash
   python ./Solution1/train.py
   ```
3. **Process**:
   - It instantiates `ClaimEvidenceLstmDataset("./data/train.csv")` for training, and `ClaimEvidenceLstmDataset("./data/dev.csv", dataset_type="validation", vocab=...)` for validation.  
   - Creates a `DataLoader` for each dataset (batch size is from `train_configs["batch_size"]`).  
   - Builds an **LstmAttentionClassifier** using `hidden_size`, `dropout_rate`, etc. from `train_config.py`.  
   - Trains for `epochs` iterations.  
   - If a new best F1 score is achieved, the script saves the model to `<save_path>/best.pt`.

---

#### `gird_search.py`

**`gird_search.py`** automates hyperparameter tuning. It:

1. Defines **`param_gird`**:
   ```python
   param_gird = {
       "epochs": [20],
       "batch_size": [5,10,20,50,100],
       "start_lr": [0.0004],
       "hidden_size": [128, 256],
       "dropout_rate": [0.1, 0.3, 0.5],
       "use_focus_loss": [False],
       "save_path": ["./Solution1/models/girdsearch/"],
       "device": ["cuda"]
   }
   ```
   - Each key is a hyperparameter; each value is a list of possible settings.

2. **Generates** all combinations of these settings via `itertools.product`.

3. For each combination:
   - **Writes** the values into `train_config.py` using `write_train_config(params_dict)`.
   - **Calls** `python ./Solution1/train.py` to launch a training run with those hyperparams.

This results in multiple training runs, each using a unique set of hyperparameters. You can check the logs and final performance metrics in the specified `save_path` (`"./Solution1/models/girdsearch/"` by default). This approach allows an easy way to compare different configurations and find the best solution.


## Solution 2: DeBERTa-LSTM-Attention Classifier

### Architecture

This solution combines a pre-trained DeBERTa transformer with LSTM and attention layers for improved performance.

Key components:
1. **Transformer Encoder**: DeBERTa-v3-base for contextualized token representations
2. **LSTM Layer**: Bidirectional LSTM to capture sequential dependencies
3. **Attention Layer**: Self-attention mechanism with learnable attention vector
4. **Classification Head**: Multi-layer classifier with dropout

```python
# Simplified architecture flow
sequence_output = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
lstm_out, (_, _) = self.lstm(sequence_output)

# Attention mechanism
scores = torch.matmul(lstm_out, self.attention_vector).squeeze(-1)
scores = scores.masked_fill(~(attention_mask.bool()), float('-inf'))
alpha = F.softmax(scores, dim=-1)
context = torch.sum(lstm_out * alpha.unsqueeze(-1), dim=1)

# Classification
logits = self.classifier(context)
```

### ClaimEvidenceTransformerDataset

**`ClaimEvidenceTransformerDataset`** is the custom dataset class used in **Solution 2** (the DeBERTa + LSTM-Attention approach) to handle claim-evidence pairs for a **Transformer-based** model. It leverages a **Hugging Face tokenizer** (e.g., DeBERTa, BERT, etc.) to tokenize and encode both claim and evidence into a format suitable for a Transformer.

---

#### Constructor Parameters

```python
ClaimEvidenceTransformerDataset(
    dataframe, 
    tokenizer, 
    load_from=None, 
    max_length=256, 
    dataset_type="train"
)
```

1. **`dataframe`**  
   - A path to a CSV file or a `pandas.DataFrame` object that includes columns like `"Claim"`, `"Evidence"`, and possibly `"label"`.  
   - If you need labeled data, ensure `"label"` exists in the CSV or DataFrame.

2. **`tokenizer`**  
   - A Hugging Face tokenizer object (e.g., `AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")`).  
   - This tokenizer is responsible for turning the `Claim` and `Evidence` text into input IDs, attention masks, etc.

3. **`load_from`**  
   - Optional. If provided, the dataset can be **loaded directly** from a saved `.pt` file instead of parsing CSV.  
   - This bypasses the normal tokenization and encoding steps.

4. **`max_length`**  
   - An integer specifying the maximum sequence length for each tokenized pair (claim + evidence).  
   - Text exceeding `max_length` will be truncated; shorter ones will be padded.

5. **`dataset_type`**  
   - One of `"train"`, `"validation"`, or `"test"`.  
   - Determines whether the dataset includes labels (e.g., `"test"` might exclude label data if not present).

---

#### Key Internal Logic

1. **Loading & Tokenizing**  
   - If `load_from` is **not** provided, each row in `dataframe` is read, extracting the `Claim` and `Evidence` strings.  
   - The `tokenizer(..., text_pair=..., truncation=True, padding='max_length')` call produces:
     ```python
     {
       "input_ids": tensor(...),
       "attention_mask": tensor(...),
       "token_type_ids": tensor(...),
       ...
     }
     ```
   - These are stored along with the `labels` if `dataset_type != 'test'`.

2. **`with_label`**  
   - Internally, the class checks if `dataset_type` is **not** `"test"`.  
   - If true, it expects a `"label"` column in the DataFrame and stores the label under `"labels"` in each encoded batch.

3. **Saving/Loading**  
   - If `load_from` is provided, `torch.load()` is used to reload the entire dataset object from disk.  
   - You can also call `save_dataset(...)` to serialize the dataset for quick reuse later.

4. **Indexing**  
   - `__getitem__` returns a single dictionary batch for the sample, containing:
     ```python
     {
       "input_ids": ...,
       "attention_mask": ...,
       "token_type_ids": ...,
       "labels": ... (if present)
     }
     ```
   - This is then collated automatically by the PyTorch `DataLoader` for mini-batch operations.

---

#### Usage in the Training Script

Within **`Solution2/train.py`**, you’ll typically see:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = ClaimEvidenceTransformerDataset("./data/train.csv", tokenizer)
val_dataset = ClaimEvidenceTransformerDataset("./data/dev.csv", tokenizer, dataset_type="validation")
```

1. **Tokenization**  
   - The Hugging Face `tokenizer` is applied to each Claim-Evidence pair, producing `input_ids`, `attention_mask`, etc.

2. **DataLoader**  
   ```python
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   ```
   - The dataset is wrapped by `DataLoader`, enabling standard mini-batch iteration in the training loop.

3. **Training**  
   - Inside the training loop, the model expects `(input_ids, attention_mask, token_type_ids, labels)` as input.  
   - The script retrieves these from each batch in `train_loader`, runs the forward pass, computes loss, and updates model parameters.

In short, **`ClaimEvidenceTransformerDataset`** is the key to bridging your raw CSV or DataFrame (with Claim/Evidence) to a **Transformer-friendly format** used by the DeBERTa-based model in Solution 2.

### Solution 2 Model Training

Similar to **Solution1**, **Solution2** also uses a dedicated `train_config.py` for hyperparameters, and a `train.py` script to run the model training process. This time, the model is based on **DeBERTa** (or other Transformer families) plus an LSTM+Attention layer.

---

#### `train_config.py`

```python
train_configs = {
    'bert_model': "microsoft/deberta-v3-base", # can be bert-base-uncased, roberta-base, etc.
    "epochs": 15,
    'start_lr': 1e-5,
    'batch_size': 5,
    'dropout_rate': 0.5,
    'use_focus_loss': True,
    "save_path": "./Solution2/models/",
    "device": "cuda",
}
```

1. **`bert_model`**  
   - String specifying which Hugging Face model to load as the Transformer backbone. For example: `"microsoft/deberta-v3-base"`, `"bert-base-uncased"`, etc.

2. **`epochs`**  
   - Integer. Number of epochs to train.

3. **`start_lr`**  
   - Float. Initial learning rate for the AdamW optimizer.

4. **`batch_size`**  
   - Integer. Number of samples per batch.

5. **`dropout_rate`**  
   - Float in [0,1]. Dropout probability used in the classification layers and possibly in the LSTM layer, to reduce overfitting.

6. **`use_focus_loss`**  
   - List or Boolean. If set to `True`, the training script uses a custom **FocusLoss** to emphasize difficult examples. Otherwise, it defaults to standard CrossEntropy.

7. **`save_path`**  
   - String. The output directory where trained models, checkpoints, and logs will be saved.

8. **`device`**  
   - String. Typically `"cuda"` or `"cpu"`. Determines whether to run training on a GPU or CPU.

---

#### `train.py`

Within **Solution2**, `train.py` reads the configuration from **`train_config.py`**, loads the `ClaimEvidenceTransformerDataset` for training and validation, and orchestrates the training loop. Here’s how you can run and customize it:

1. **Set Parameters in `train_config.py`**  
   - Modify `'bert_model'`, `'epochs'`, etc. as needed.

2. **Run**:
   ```bash
   python ./Solution2/train.py
   ```
   - The script will import `train_configs` from `train_config.py` and use them to set up the **BertLstmAttentionClassifier** (or other suitable models).

3. **Process**:
   - **Data Loading**: Creates `train_dataset` and `val_dataset` from `ClaimEvidenceTransformerDataset(...)` pointing to your CSV (e.g. `./data/train.csv` / `./data/dev.csv`).  
   - **Model Initialization**: Loads the specified Hugging Face model (e.g., DeBERTa V3) plus an LSTM+Attention classifier head, applying the chosen `dropout_rate`.  
   - **FocusLoss** or **CrossEntropy**: Based on `use_focus_loss` in `train_configs`.  
   - **Training Loop**: Trains for `epochs` epochs, evaluating on the development set each epoch, and saving the best-performing model to `<save_path>/best.pt`.

#### `eval.py` – Evaluation Utilities and Interfaces

The **`eval.py`** script provides shared functionality for evaluating both **Solution1** (LSTM-Attention) and **Solution2** (DeBERTa+LSTM) models, as well as utilities for prediction, plotting metrics, and saving outputs.

---

#### 1. `FocusLoss(nn.Module)`

```python
class FocusLoss(nn.Module):
    ...
```
- A custom **focal loss** variant that emphasizes hard-to-classify samples by adjusting loss weights based on the confidence of correct classification.  
- It can incorporate **class weights** (`self.weight`) and a **gamma** parameter controlling how sharply to penalize confident but incorrect predictions.  
- **Usage**: Often substituted in place of standard `CrossEntropyLoss` when you set `use_focus_loss=True` in your training config.

---

#### 2. `evaluate_model(model, dataloader, loss_fn, device)`

```python
def evaluate_model(model, dataloader, loss_fn, device):
    ...
```
- **Evaluates** a model on a given `dataloader` (validation or test set).  
- Returns **average loss**, **accuracy**, and **F1-score**.  
- **Model type detection**: Checks if `model` is a `LstmAttentionClassifier` or a `BertLstmAttentionClassifier` and fetches batch data accordingly.  
- **Integration**: Called in both `train.py` scripts to compute validation metrics after each epoch and track performance.

---

#### 3. `model_predict(model, dataset, batch_size=150, device='cpu')`

```python
def model_predict(model, dataset, batch_size=150, device='cpu'):
    ...
```
- Generates **predictions** (class indices) for an entire dataset without returning intermediate probabilities or logits.  
- Respects **model type** (LSTM-based vs. Transformer-based) by extracting the correct fields from each batch.  
- **Used** to perform inference on a separate test set or real-world examples once the model is trained.

---

#### 4. `model_predict_text(model, claim, evidence, preprocessor)`

```python
def model_predict_text(model, claim, evidence, preprocessor):
    ...
```
- A **single-inference** method:  
  1. Wraps a single `(claim, evidence)` pair into a minimal **DataFrame**.  
  2. Instantiates the appropriate dataset class (LSTM or Transformer) by checking `model` type.  
  3. Calls `model_predict(...)` to get a **single** prediction (boolean).  
- Useful for quick testing or interactive demos.

---

#### 5. `predict_to_csv(labels, save_path)`

```python
def predict_to_csv(labels, save_path):
    ...
```
- Writes a **list of predicted labels** to a CSV file.  
- Typically used for test-set outputs, so you can easily submit predictions or analyze them externally.

---

#### 6. `compute_metrics(preds, labels)`

```python
def compute_metrics(preds, labels):
    ...
```
- Computes **accuracy** and **weighted F1-score** using **scikit-learn**.  
- A universal helper for evaluating classification results in both solutions.

---

#### 7. `plot_training_graph(train_info_recorder, save_path)`

```python
def plot_training_graph(train_info_recorder, save_path):
    ...
```
- Generates **line plots** of training vs. validation **Loss** and **Accuracy** across epochs.  
- Saves `.png` figures to the specified directory:
  - `train_val_loss.png`  
  - `train_val_acc.png`
- Typically called once training finishes, using a train info recorder wit the structure like:
  ```python
  train_info_recorder = [
      { "epoch": 0, "train_loss": ..., "val_loss": ..., "train_acc": ..., "val_acc": ... },
      ...
  ]
  ```


### Use the Model

Interactive demo notebooks are provided for both solutions:
  - s1_demo.ipynb: Demonstrates the LSTM-Attention model
  - s2_demo.ipynb: Demonstrates the DeBERTa-LSTM-Attention model

Both demo shows how to load models and necessary resources (tokenizer, vocab, embeddings). We also showed two approaches for model prediction: 

1. A test file contained many claim-evidence pairs without labels 

2. A given claim-evidence text pair.

## Results

### Solution 1 (LSTM-Attention)

Best performance with hyperparameters:
- batch_size: 5
- learning_rate: 0.0004
- hidden_size: 128
- dropout_rate: 0.3
- use_focus_loss: True

Metrics:
- Accuracy: 0.8331
- F1 Score: 0.8296
- Matthews Correlation: 0.5691

### Solution 2 (DeBERTa-LSTM-Attention)

Best performance with hyperparameters:
- learning_rate: 1e-05
- batch_size: 5
- dropout_rate: 0.1
- epochs: 15

Metrics:
- Accuracy: 0.8913
- F1 Score: 0.8922
- Matthews Correlation: 0.7338

## References

1. Chen, Q., Zhu, X., Ling, Z., Wei, S., Jiang, H., & Inkpen, D. (2016). Enhanced LSTM for Natural Language Inference. arXiv preprint arXiv:1609.06038. [Link](https://arxiv.org/pdf/1609.06038)

2. He, P., Liu, X., Gao, J., & Chen, W. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. arXiv preprint arXiv:2006.03654. [Link](https://arxiv.org/pdf/2006.03654)

3. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 2980-2988. [Link](https://arxiv.org/pdf/1708.02002v2)

4. Zhou, P., Shi, W., Tian, J., Qi, Z., Li, B., Hao, H., & Xu, B. (2016). Attention-based Bidirectional LSTM Networks for Relation Classification. In Proceedings of ACL. [Link](https://aclanthology.org/P16-2034.pdf)
