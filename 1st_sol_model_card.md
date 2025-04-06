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

This is a classification model that was trained to detect whether 
      a piece of evidence supports or contradicts a given claim.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model uses a bidirectional LSTM with attention mechanism to classify claim-evidence pairs. It processes both claim and evidence text separately, applies attention between them, and uses difference enhancement for better feature extraction. The model architecture includes embedding layer, BiLSTM encoding, softmax attention, difference enhancement, composition BiLSTM, and max/avg pooling before classification.

- **Developed by:** Fan Mo and Zixiao Nong
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** BiLSTM with Attention
- **Finetuned from model [optional]:** None (Uses pretrained word2vec-google-news-300 embeddings)

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** None
- **Paper or documentation:** None

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
      - learning_rate: 1e-04
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

The model obtained an F1-score of 80% and an accuracy of 80%.

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

he model uses learning rate reduction on plateau based on F1 scores.
      The best model is saved during training based on development set F1 score.
      Word embeddings are frozen during training.
