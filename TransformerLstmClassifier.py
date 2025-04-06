import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class BertLstmAttentionClassifier(nn.Module):
    def __init__(self,
                 bert_model_name="microsoft/deberta-v3-base",
                 lstm_hidden_size=256,
                 num_labels=2,
                 bidirectional=True,
                 dropout=0.1):
        super().__init__()
        
        # 1. 加载BERT的主体(不包含分类头)
        #    如果想用DeBERTa,可以替换成AutoModel: "microsoft/deberta-v3-base"
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # BERT隐藏维度大小
        hidden_size = self.bert.config.hidden_size  # e.g. 768 for bert-base
        
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        
        # 2. 构建LSTM层
        #    输入维度=bert hidden_size, 输出维度=rnn_hidden_size
        #    如果双向,输出实际维度= 2*rnn_hidden_size
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=lstm_hidden_size,
                            batch_first=True,
                            bidirectional=bidirectional)
        
        # 3. Attention向量 (可学习)
        #    用于对LSTM输出序列做attention加权
        #    形状 [2*rnn_hidden_size, 1] (若是双向)
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.attention_vector = nn.Parameter(torch.randn(lstm_output_size, 1))
        
        # 4. Dropout + 最后分类层
        self.dropout = nn.Dropout(dropout)
                # classify head
        self.classifier = nn.Sequential(self.dropout,
                                        nn.Linear(lstm_output_size,
                                                       lstm_output_size//2),
                                        nn.Tanh(),
                                        self.dropout,
                                        nn.Linear(lstm_output_size//2,
                                                       num_labels),
                                        nn.Softmax(dim=-1))
        
        # # 初始化
        # nn.init.xavier_uniform_(self.classifier.weight)
        # nn.init.normal_(self.attention_vector, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # ====== 1. BERT编码 ======
        # BERT输出: last_hidden_state = [B, L, hidden_size], pooler_output, ...
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)
        sequence_output = outputs.last_hidden_state  # [B, L, hidden_size]
        
        # ====== 2. LSTM ======
        # 输入 shape: [B, L, hidden_size], 输出 shape: [B, L, 2*rnn_hidden_size] (双向)
        lstm_out, (_, _) = self.lstm(sequence_output)
        # lstm_out: [B, L, rnn_output_size]

        # ====== 3. Attention加权 ======
        # 3.1 计算每个token的注意力分数
        #     attention_vector: [rnn_output_size, 1]
        #     lstm_out: [B, L, rnn_output_size]
        # => scores: [B, L]
        scores = torch.matmul(lstm_out, self.attention_vector).squeeze(-1)  # [B, L, 1] -> [B, L]
        
        # 对padding位置进行mask(若需要)
        # 如果 attention_mask shape=[B, L],可以在scores中对mask=0的地方加 -inf
        if attention_mask is not None:
            # attention_mask: [B, L], 1表示有效token, 0表示padding
            scores = scores.masked_fill(~(attention_mask.bool()), float('-inf'))
        
        # 3.2 归一化(softmax)
        alpha = F.softmax(scores, dim=-1)  # [B, L], 每个token的权重
        
        # 3.3 计算加权和: context = \sum alpha_i * h_i
        #     lstm_out: [B, L, rnn_output_size], alpha: [B, L]
        # => context: [B, rnn_output_size]
        alpha = alpha.unsqueeze(-1)  # [B, L, 1]
        context = torch.sum(lstm_out * alpha, dim=1)  # [B, rnn_output_size]
        
        # ====== 4. 分类 ======
        context = self.dropout(context)
        logits = self.classifier(context)  # [B, num_labels]
        
        return logits
