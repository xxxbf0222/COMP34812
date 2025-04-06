---
{}
---
language: en
license: cc-by-4.0
tags:
- pairwise-sequence-classification
repo: https://github.com/xxxbf0222/COMP34812

---

# Model Card for z71565fm-g03302zn-evidence_detection

<!-- Provide a quick summary of what the model is/does. -->

This is a fine-tuned DeBERTa model with LSTM and attention mechanism for claim-evidence classification, determining whether a piece of evidence is relevant to a given claim.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model fine-tunes DeBERTa-v3-base and adds a bidirectional LSTM followed by a self-attention layer and classification head. The architecture processes claim-evidence pairs through the transformer encoder, captures sequential dependencies with the LSTM, and uses attention to focus on the most relevant parts of the sequence before final classification. The attention mechanism employs a learnable attention vector to weight token representations, particularly effective for longer sequences.

- **Developed by:** Fan Mo and Zixiao Nong
- **Language(s):** English
- **Model type:** Supervised-learning using Transformer architecture
- **Model architecture:** Transformer-LSTM with Attention
- **Finetuned from model [optional]:** microsoft/deberta-v3-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/microsoft/deberta-v3-base
- **Paper or documentation:** https://arxiv.org/abs/2006.03654

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The model was trained on more than 21K claim-evidence pairs with binary labels indicating relevance/irrelevance (train.csv). The DeBERTa tokenizer was used to encode the claim-evidence pairs as sequence pairs with special tokens, truncated to a maximum length of 256 tokens. Claim and evidence texts are processed together as a text pair, leveraging the transformer's ability to model the relationship between two text segments.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 0.00001
      - batch_size: 5
      - num_epochs: 15
      - dropout_rate: 0.1
      - optimizer: AdamW

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

TODO
      - overall training time:  hours
      - duration per training epoch:  minutes
      - model size: 

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

The development set (dev.csv) is used for indecating model performances, which does not appear in the training set.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

Best model performance:
      - Epoch: 14/15
      - Training Loss: 0.0560
      - Training Accuracy: 0.9829
      - Development Loss: 0.0993
      - Development Accuracy: 0.8930
      - Development F1-score: 0.87

## Technical Specifications

### Hardware


      - RAM: at least 4 GB
      - Storage: at least 1GB,
      - GPU: RTX4080 LAPTOP (12G)

### Software


      - PyTorch
      - Transformers
      - Pandas
      - Numpy

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

We only use the development set to test the model performance, which could leads the model overfit to the development set and reduce some general ability. Limited by the maximum sequence length (256 tokens). Model performance may decrease for very long claim-evidence pairs due to truncation. May inherit biases present in the pre-trained DeBERTa model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The model combines the strengths of pre-trained transformers with recurrent neural networks. The transformer component (DeBERTa) provides strong contextual embeddings, while the LSTM captures sequential dependencies and the attention mechanism highlights the most relevant parts of the input. 
    Scripts for loading/preprocessing dataset, training/evaluating model are all provided in the [Github code repository](https://github.com/xxxbf0222/COMP34812).
