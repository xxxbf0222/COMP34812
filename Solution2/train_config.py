train_configs = {
    'bert_model': "microsoft/deberta-v3-base", # support different BERT model, e.g. bert-base-uncased/bert-base-cased/Roberta/Deberta
    "epochs": 15,
    'start_lr': 1e-5,
    'batch_size': 5,
    'dropout_rate': 0.5,
    'use_focus_loss': [True, False],
    "save_path": "./Solution2/models/", # please keep a '/' in the end
    "device": "cuda",
}