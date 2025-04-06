# COMP34812

## Solution 1

数据集类 ClaimEvidenceDataset -> ClaimEvidenceLstmDataset

模型类  LstmAttentionClassifier -> LstmAttentionClassifier

训练脚本 train_attn_lstm.py

模仿的ESIM结构

论文：https://arxiv.org/pdf/1609.06038

参数选择和调度器参考了这个代码库 https://github.com/coetaur0/ESIM

但模型代码部分是自己实现的

使用了pos tokenizer，focus loss（支持选择不同tokenizer和loss fn进行训练）并用gird search做超参数比较，目前最优：

```
Hyper-params:

{'batch_size': 10, 'epochs': 30, 'start_lr': 0.0001, 'hidden_size': 256, 'dropout_rate': 0.5, 'use_focus_loss': False}

Performance:

Epoch [20/30], Train Loss=0.4609, Acc=0.8467, Dev Loss=0.4965, Acc=0.8071, F1=0.8071
```




## Solution 2 (未整理好）
数据集类 finetune_deberta -> ClaimEvidenceTransformerDataset

模型类  TransformerLstmClassifier -> BertLstmAttentionClassifier

训练脚本 finetune_deberta.py

基于deberta-v3-base微调，输出层接入lstm -> attention layer -> full connect layer for classification （详见代码）

参考论文 https://arxiv.org/abs/2006.03654

模型原地址 https://huggingface.co/microsoft/deberta-v3-base

**再找1-2篇基于bert+lstm文本分类的**

支持载入不同的bert模型（bert-case/uncase、roberta、deberta），以及选择loss fn

使用gird search比较超参数

目前最优：
```
Hyper-params:

lr=>1e-05 batch_size=>5 dropout_rate=>0.1

Performance:

Epoch [14/15], Train Loss=0.0560, Acc=0.9829, Dev Loss=0.0993, Acc=0.8930, F1=0.87
```
