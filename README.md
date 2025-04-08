# Evidence Detection for Claim-Evidence Pairs

This repository contains two different solutions for a claim-evidence classification task, which determines whether a piece of evidence is relevant to a given claim. Both approaches implement different neural network architectures with state-of-the-art techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Solution 1: LSTM-Attention Classifier](#solution-1-lstm-attention-classifier)
- [Solution 2: DeBERTa-LSTM-Attention Classifier](#solution-2-deberta-lstm-attention-classifier)
- [Data Processing](#data-processing)
- [Training and Evaluation](#training-and-evaluation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Project Overview

This project implements two different approaches to the evidence detection task:

1. **LSTM-Attention Classifier**: A BiLSTM-based model inspired by the ESIM architecture, using attention mechanisms and no pre-trained transformers.
2. **DeBERTa-LSTM-Attention Classifier**: A hybrid model that combines the DeBERTa transformer with LSTM and attention layers.

Both models achieve strong performance on the task, with the DeBERTa-based model achieving superior results.

## Repository Structure
```
.
├── Solution1/                      # LSTM-Attention solution
│   ├── LstmAttentionClassifier.py  # Main model implementation
│   ├── data/
│   │   ├── ClaimEvidenceDataset.py # Dataset class
│   │   ├── POStokenizer.py         # Custom tokenizer with POS tagging
│   │   └── SimpleTokenizer.py      # Alternative simple tokenizer
│   ├── gird_search.py              # Hyperparameter optimization
│   ├── train.py                    # Training script
│   └── train_config.py             # Configuration file
│
├── Solution2/                      # DeBERTa-LSTM-Attention solution
│   ├── TransformerLstmClassifier.py      # Main model implementation
│   ├── data/
│   │   └── ClaimEvidenceTransformerDataset.py  # Dataset class for transformer
│   ├── gird_search.py              # Hyperparameter optimization
│   ├── train.py                    # Training script
│   └── train_config.py             # Configuration file
│
├── eval.py                         # Evaluation utilities
├── s1_demo.ipynb                   # Demo notebook for Solution 1
└── s2_demo.ipynb                   # Demo notebook for Solution 2
```

## Solution 1: LSTM-Attention Classifier

### Architecture

This solution implements a variant of the Enhanced Sequential Inference Model (ESIM) architecture, originally designed for natural language inference, adapted for evidence detection.

Key components:
1. **Word Embeddings**: Pre-trained word2vec-google-news-300 embeddings
2. **Encoder**: BiLSTM to encode both claim and evidence sequences 
3. **Attention Mechanism**: Cross-attention between claim and evidence
4. **Enhancement Layer**: Element-wise difference and multiplication
5. **Composition**: Second BiLSTM layer for compositional inference
6. **Pooling**: Max and average pooling operations
7. **Classification**: Multi-layer classifier with dropout

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

### **Data Structures**:
  - **ClaimEvidenceLstmDataset**: Custom PyTorch dataset for processing claim-evidence pairs
  - Uses POS tokenizer for text preprocessing
  - Vocabulary mapping and embedding matrix construction
  - Padding and sequence length tracking for dynamic batching

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

### **Data Structures**:
  - ClaimEvidenceTransformerDataset: Custom PyTorch dataset for transformer-based models
  - Uses DeBERTa tokenizer for text preprocessing
  - Special token handling for sequence pairs
  - Maximum sequence length handling with truncation

## Data Processing

Both solutions implement custom data processing pipelines:

1. **Solution 1**:
  - Uses POS tagging for tokenization
  - Builds vocabulary from training data
  - Loads pre-trained embeddings for known tokens
  - Handles padding and sequence length tracking

2. **Solution 2**:
  - Uses DeBERTa tokenizer
  - Handles claim-evidence pairs as single sequences with special tokens
  - Implements truncation and padding to fixed length
  - Stores tokenized sequences as tensors for efficiency

## Training and Evaluation

### Training Techniques

Both solutions employ advanced training techniques:

1. **Grid Search**: Systematic hyperparameter optimization
   ```python
   param_grid = {
       'lr': [1e-5, 5e-6],
       'batch_size': [5, 10],
       'dropout': [0.5],
       'use_focus_loss': [True, False]
   }
   ```

2. **Learning Rate Scheduling**:
  - Solution 1: ReduceLROnPlateau (reduces LR when metrics plateau)
  - Solution 2: Linear warmup with decay

3. **Loss Functions**:
  - Weighted Cross-Entropy Loss (default)
  - Focal Loss (optional) - Downweights well-classified examples to focus on difficult ones
    ```python
    class FocusLoss(nn.Module):
        def forward(self, logits, targets):
            ce_loss = torch.nn.functional.cross_entropy(logits, targets)
            pt = torch.exp(-ce_loss)
            at = self.weight.gather(0, targets)
            loss = at * (1 - pt) ** self.gamma * ce_loss
            return loss.mean()
    ```

4. **Model Checkpointing**: Saves the best model based on validation F1 score

### Evaluation Metrics:
  - Accuracy
  - F1 Score (weighted)
  - Precision and Recall
  - Matthews Correlation Coefficient

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/xxxbf0222/COMP34812.git
cd COMP34812

# Install dependencies
pip install torch pandas numpy matplotlib scikit-learn transformers nltk
```

### Running the models

1. Solution 1 (LSTM-Attention):
  ```bash
  # Train model
  python Solution1/train.py

  # Grid search for hyperparameters
  python Solution1/gird_search.py
  ```

2. Solution 2 (DeBERTa-LSTM-Attention):
  ```bash
  # Train model
  python Solution2/train.py
  
  # Grid search for hyperparameters
  python Solution2/gird_search.py
  ```

### Demo Notebooks

Interactive demo notebooks are provided for both solutions:
  - s1_demo.ipynb: Demonstrates the LSTM-Attention model
  - s2_demo.ipynb: Demonstrates the DeBERTa-LSTM-Attention model

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
