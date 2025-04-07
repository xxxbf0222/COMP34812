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
        """
        This class implements a Transformer-based backbone (BERT/DeBERTa)
        followed by an LSTM layer and a final attention mechanism for classification.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        
        # 1. load BERT model (without classifier head) 
        # support different BERT model, e.g. bert-base-uncased/bert-base-cased/Roberta/Deberta
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # BERT hidden dimension size
        hidden_size = self.bert.config.hidden_size
        
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        
        # 2. Build LSTM layer. 
        #     Input dimension = BERT hidden_size, 
        #     output dimension = lstm_hidden_size. If bidirectional=True, 
        #     output dimension = 2 * lstm_hidden_size.")
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 3. Attention vector (learnable). 
        #     Used to compute attention scores on LSTM outputs. 
        #     If bidirectional, shape is [2 * lstm_hidden_size, 1].")
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.attention_vector = nn.Parameter(torch.randn(lstm_output_size, 1))
        
        # 4. Final classification layer.
        #     Using same classification header from solution 1
        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            self.dropout,
            nn.Linear(lstm_output_size // 2, num_labels),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward method:
        1) Encode input with BERT/DeBERTa,
        2) Pass through LSTM,
        3) Apply attention weighting,
        4) Classify via a feed-forward head.
        """
        # ====== 1. BERT encoding ======
        # Output is last_hidden_state of shape [B, L, hidden_size]
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state  # [B, L, hidden_size]
        
        # ====== 2. LSTM ======
        # Input shape: [B, L, hidden_size], 
        # Output shape: [B, L, 2*rnn_hidden_size]
        lstm_out, (_, _) = self.lstm(sequence_output)
        # lstm_out: [B, L, rnn_output_size]

        # ====== 3. Attention Layer ======
        # 3.1 Compute attention score for each token using the attention vector.
        scores = torch.matmul(lstm_out, self.attention_vector).squeeze(-1)  # [B, L, 1] -> [B, L]
        
        # "Mask padding positions if attention_mask is provided.
        if attention_mask is not None:
            # attention_mask: [B, L], 1 for alid token, 0 for padding
            scores = scores.masked_fill(~(attention_mask.bool()), float('-inf')) # [B, L]
        
        # 3.2 Normalize scores via softmax.
        alpha = F.softmax(scores, dim=-1)  # [B, L]

        # 3.3 context = sum_i(alpha_i * h_i)
        # Weighted sum of LSTM outputs using attention coefficients alpha.
        alpha = alpha.unsqueeze(-1)  # [B, L, 1]
        context = torch.sum(lstm_out * alpha, dim=1)  # [B, rnn_output_size]
        
        # ====== 4. Classification Header ======
        context = self.dropout(context)
        logits = self.classifier(context)  # [B, num_labels]
        
        return logits
