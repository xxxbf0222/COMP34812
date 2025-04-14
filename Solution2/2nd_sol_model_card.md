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

This is a fine-tuned model from [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) with LSTM and attention mechanism for claim-evidence classification, determining whether a piece of evidence is relevant to a given claim.

## Download Checkpoint

[Access model file here](https://livemanchesterac-my.sharepoint.com/:f:/g/personal/fan_mo-4_student_manchester_ac_uk/ElF2ODHq9ltGpS6X-DKbXGMBiuHY-lcjxRYb2Mbyz2hiNA?e=kAXZmw )

The model checkpoint is `./Solution2/models/best.pt`, please also download the `deberta-v3-tokenizer.pt` as well for data preprocessing. Tutorials for how to use these and run the model, visit [our Github code repository](https://github.com/xxxbf0222/COMP34812).

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model fine-tunes [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) and adds a bidirectional LSTM followed by a self-attention layer and classification head. The architecture processes claim-evidence pairs through the transformer encoder, captures sequential dependencies with the LSTM, and uses attention to focus on the most relevant parts of the sequence before final classification. The attention mechanism employs a learnable attention vector to weight token representations to discover more hidden information from DeBERTa outputs.

- **Developed by:** [Fan Mo](https://github.com/xxxbf0222) and [Zixiao Nong](https://github.com/zix1ao)
- **Language(s):** English
- **Model type:** Supervised-learning using Transformer architecture
- **Model architecture:** Transformer-LSTM with Attention
- **Finetuned from model:** [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)

**Model Pipeline** :
1) DeBERTa-v3-base
2) Bi-LSTM
3) Self-Attention Layer
5) Final classification layer


### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/microsoft/deberta-v3-base
- **Paper or documentation:** https://arxiv.org/abs/2006.03654
- **Focal Loss:** https://arxiv.org/pdf/1708.02002v2
- **bi-LSTM Attention:** https://aclanthology.org/P16-2034.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The model was trained on more than 21K claim-evidence pairs with binary labels indicating relevance/irrelevance [train.csv](./data/train.csv). The DeBERTa-v3-base tokenizer was used to encode the claim-evidence pairs as sequence pairs with special tokens, truncated to a maximum length of 256 tokens. Claim and evidence texts are processed together as a text pair, with a special token to separate between them, leveraging the transformer's ability to model the relationship between two text segments.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

The model was trained totally on the training set [train.csv](../data/train.csv), and trained for 5 (out of 15) epochs. After each epoch, the model runs an evaluation on the development set [dev.csv](../data/dev.csv), and record its accuracy and f1-score. Due to the unbalance in training set, **weighted f1-score** is used as the performace indicator. If the current f1-score is higher than previous epochs, the model will be considered as the current best model, and saved by covering a previous best model. 

Cross Entropy Loss is used by default, but the Focal Loss is also implemented and supported. Since the training data is unbalance, class weights is applied in loss function.

Due to limited computing resources, we can only use a small batch size. Therefore, to prevent overfitting due to too many steps in the early stage of model training, we use a warm-up strategy in the first 10% of the steps.

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

A Gird Search is applied to find the best parameters. Parameter combinations are generated from:

      - epochs: 15
      - batch_size: [5,10,15]
      - start_learning_rate: [1e-6, 1e-5, 5e-5]
      - dropout_rate: [0.1, 0.3, 0.5]
      - use_focus_loss: [True, False]
      - optimizer: AdamW
      - scheduler: linear with warm-up steps

For each param-combination, we trained on the entire training dataset for 15 epochs, and save the best model indecated by f1-score on development set. Finally, the best param-combination that trained the best model are:

      - batch_size: 5
      - start_learning_rate: 5e-6
      - dropout_rate: 0.1
      - use_focus_loss: True
      - optimizer: AdamW
      - scheduler: linear with warm-up steps

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

      - overall training time: 2.8 hours
      - duration per training epoch:  10.5 minutes
      - model size: 744.4 MB

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
      - **weighted_mmacro_f1**  (performance indecator in training process)
      - matthews_corrcoef

### Results


Best model performance:

      - accuracy_score: 0.89132635842052
      - macro_precision: 0.86079804339787
      - macro_recall: 0.87311140639405
      - macro_f1: 0.86658365953765
      - weighted_macro_precision: 0.89366131498043
      - weighted_macro_recall: 0.89132635842052
      - weighted_mmacro_f1: 0.89223772742326
      - matthews_corrcoef: 0.73380614714351


## Technical Specifications

### Hardware


      - RAM: at least 4 GB
      - Storage: at least 1GB,
      - GPU: RTX4080 LAPTOP (12G)

### Software

      - python
      - matplotlib==3.8.0
      - nltk==3.8.1
      - numpy==1.26.4
      - pandas==1.5.3
      - scikit_learn==1.2.2
      - torch==2.6.0
      - transformers==4.45.2


## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

* We only use the development set to test the model performance, which could leads the model overfit to the development set and reduce some general ability.

* Limited by the maximum sequence length (256 tokens). Model performance may decrease for very long claim-evidence pairs due to truncation. May inherit biases present in the pre-trained DeBERTa model.

* The forward Lstm-attention did not significantly improve the model performance, but increased the model training time and required computing resources. Therefore, the effectiveness of this architecture deserves further study on other tasks.


## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Scripts for loading/preprocessing dataset, training/evaluating model are all provided in the [Github code repository](https://github.com/xxxbf0222/COMP34812).
