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

This is a claim-evidence classification model based on the ESIM architecture that was trained to detect whether a piece of evidence is relevant to a given claim.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model implements a variant of the [Enhanced Sequential Inference Model ESIM, (Chen et al., 2016)](https://arxiv.org/pdf/1609.06038), which extend the ESIM model from solving Natural Language Inference (NLI) task to the Evidence Detection (ED) task. It combined bidirectional LSTM and attention mechanisms for claim-evidence classification. The model processes claim and evidence text separately through embedding and BiLSTM encoding layers, applies cross-attention between them, and enhances the representation with element-wise difference and multiplication operations. A second BiLSTM composition layer and pooling operations prepare the final representation for classification. 



- **Developed by:** Fan Mo and Zixiao Nong
- **Language(s):** English
- **Model type:** Supervised-learning without using Transformer architecture
- **Model architecture:** ESIM-inspired BiLSTM with Attention Mechanism
- **Finetuned from model [optional]:** None (Uses pretrained word2vec-google-news-300 embeddings with POS tokenizer)

### Model Resources

This model implements a variant of the [Enhanced Sequential Inference Model ESIM, (Chen et al., 2016)](https://arxiv.org/pdf/1609.06038), since the paper was published very early (2016), the original code base used python 2, so we also checked [@coetaur0 's ESIM code base](https://arxiv.org/pdf/1609.06038), which rewrote the ESIM architecture using PyTorch, and we referred to this code base to determine the learning rate scheduler and the initial training parameter range. We also used [Focus Loss](https://arxiv.org/pdf/1708.02002v2) as an optional loss function as a comparasion.
<!-- Provide links where applicable. -->

- **Repository:** https://github.com/coetaur0/ESIM
- **Paper or documentation:** https://arxiv.org/pdf/1609.06038
- **Focus Loss:** https://arxiv.org/pdf/1708.02002v2

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The model was training on more than 21K claim-evidence pairs with binary labels indicating relevance/irrelevance (train.csv). A Part-of-Speech (POS) tokenizer is implemented and used to dealing with input texts. 

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

The model was trained totally on the training set (train.csv), and trained for 30 epochs. Word embeddings are frozen during training. After each epoch, the model runs an evaluation on the development set (dev.csv), and record its accuracy and f1-score. Due to the unbalance in training set, weighted f1-score is used as the performace indicator. If the current f1-score is higher than previous epochs, the model will be considered as the current best model, and saved by covering a previous best model. 

Cross Entropy Loss is used by default, but the Focus Loss is also implemented and supported. Since the training data is unbalance, class weights is applied in loss function.

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

A Gird Search is applied to find the best parameters. Parameter combinations are generated from:

      - epochs: 30
      - batch_size: [5,10,20,50,100]
      - start_learning_rate: [0.0004, 0.0002, 0.0001]
      - hidden_size: [128, 256]
      - dropout_rate: [0.1, 0.3, 0.5]
      - use_focus_loss: [True, False]
      - optimizer: Adam
      - scheduler: ReduceLROnPlateau

For each param-combination, we trained on the entire training dataset for 30 epochs, and save the best model indecated by f1-score on development set. Finally, the best param-combination that trained the best model are:

      - batch_size: 10
      - num_epochs: 30
      - start_learning_rate: 0.0001
      - hidden_size: 256
      - dropout_rate: 0.5
      - use_focus_loss: False
      - optimizer: Adam
      - scheduler: ReduceLROnPlateau

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->
TODO:

Time for training model using best parameters:

      - overall training time: hours
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
      - accuracy_score
      - macro_precision
      - macro_recall
      - macro_f1
      - weighted_macro_precision
      - weighted_macro_recall
      - **weighted_mmacro_f1** 
      (performance indecator in training process)
      - matthews_corrcoef

### Results
TODO:

Best model performance:
      - accuracy_score
      - macro_precision
      - macro_recall
      - macro_f1
      - weighted_macro_precision
      - weighted_macro_recall
      - **weighted_mmacro_f1**
      - matthews_corrcoef

## Technical Specifications

### Hardware


      - RAM: at least 4 GB
      - Storage: at least 1GB,
      - GPU: RTX4080 LAPTOP (12G)

### Software


      - PyTorch
      - Pandas
      - Numpy

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

* We only use the development set to test the model performance, which could leads the model overfit to the development set and reduce some general ability.

* The (ESIM architecture)[https://arxiv.org/pdf/1609.06038] is first proposed for Natural Languange Inference (NLI) task, not Evidence Detection. Therefore, there may be some performance loss when migrating the ESIM architecture to the dataset of this task.

* We only tried one pre-trained word embeddings (word2vec-google-news-300). So the solution might limited by the quality and coverage of the pre-trained word embeddings. Other pre-trained or self-trained embeddings might have potential in improving model performance.



## Additional Information
<!-- Any other information that would be useful for other people to know. -->
Scripts for loading/preprocessing dataset, training/evaluating model are all provided in the [Github code repository](https://github.com/xxxbf0222/COMP34812).