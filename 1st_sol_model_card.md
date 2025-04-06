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

This is a classification model based on the ESIM architecture that was trained to detect whether a piece of evidence supports or contradicts a given claim.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model implements a variant of the Enhanced Sequential Inference Model (ESIM, Chen et al., 2016) with bidirectional LSTM and attention mechanisms for claim-evidence classification. The model processes claim and evidence text separately through embedding and BiLSTM layers, applies cross-attention between them, and enhances the representation with element-wise difference and multiplication operations. A second BiLSTM composition layer and pooling operations prepare the final representation for classification.

- **Developed by:** Fan Mo and Zixiao Nong
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** ESIM-inspired BiLSTM with Attention
- **Finetuned from model [optional]:** None (Uses pretrained word2vec-google-news-300 embeddings with POS tokenizer)

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://github.com/coetaur0/ESIM
- **Paper or documentation:** https://arxiv.org/pdf/1609.06038

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

More than 21K claim-evidence pairs with binary labels indicating support/contradiction

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - batch_size: 10
      - num_epochs: 30
      - learning_rate: 0.0001
      - hidden_size: 256
      - dropout_rate: 0.5
      - use_focus_loss: False
      - optimizer: Adam
      - scheduler: ReduceLROnPlateau

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time:  hours
      - duration per training epoch:  minutes
      - model size: 

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A development(validation) set provided, amounting to 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

Best model performance:
      - Epoch: 20/30
      - Training Loss: 0.4609
      - Training Accuracy: 0.8467
      - Development Loss: 0.4965
      - Development Accuracy: 0.8071
      - Development F1-score: 0.8071

## Technical Specifications

### Hardware


      - RAM: --------------------------at least 16 GB
      - Storage: ------------------------------at least 2GB,
      - GPU: -----------------------------V100

### Software


      - Pytorch
      - Pandas
      - Numpy

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Limited by the quality and coverage of the word embeddings.
      Out-of-vocabulary words are mapped to <unk> token.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The model is inspired by the ESIM architecture but with custom implementation. It uses POS tokenizer and supports optional Focus Loss. Parameter selection and learning rate scheduler were inspired by the ESIM repository. Grid search was used for hyperparameter tuning to reach the optimal configuration. The model uses learning rate reduction on plateau based on F1 scores. The best model is saved during training based on development set F1 score. Word embeddings are frozen during training.
