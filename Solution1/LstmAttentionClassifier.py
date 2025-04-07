import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SimpleLstmClassifier(nn.Module):
    '''
    Implemented as a LSTM baseline model for performance comparison.
    '''
    def __init__(self, vocab_size, 
                 embedding_dim, 
                 num_layers, 
                 hidden_dim, 
                 embedding_mat, 
                 dropout_rate, 
                 num_class):
        """
        A simple LSTM-based baseline model:
          - Uses a frozen embedding layer loaded from pre-trained vectors
          - Encodes both claim and evidence via an same LSTM
          - Concatenates the final hidden states from claim and evidence
          - Passes them through a couple of linear layers for classification
        """
        super(SimpleLstmClassifier, self).__init__()
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Embedding layer with pre-trained weights
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_mat)
        # Freeze embedding layer so that weights are not updated
        self.embedding.weight.requires_grad = False

        # LSTM encoder (bidirectional)
        self.lstm_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        
        # Two fully-connected layers for classification
        # The first reduces dimensionality, the second outputs num_class
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, num_class)
        
        # Final activation to produce probabilities
        self.activation = nn.Softmax()

    def _encode(self, embedded_seq, lengths):
        """
        Encodes an embedded sequence of tokens using bidirectional LSTM.
        Returns the concatenation of the final forward and backward hidden states.
        """
        packed_seq = pack_padded_sequence(
            embedded_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm_encoder(packed_seq)
        
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        # So we take hidden[-2] (last forward state) and hidden[-1] (last backward state)
        return torch.cat((hidden[-2], hidden[-1]), dim=1)

    def forward(self, claim_seq, claim_lens, evidence_seq, evicence_lens):
        """
        Forward pass:
         1. Embed the claim and evidence sequences
         2. Encode with LSTM to get final hidden states
         3. Concatenate claim and evidence features
         4. Pass through fc1 -> fc2
         5. Output logits and probabilities
        """
        embedded_claim = self.embedding(claim_seq)
        embedded_evidence = self.embedding(evidence_seq)

        encoded_claim = self._encode(embedded_claim, claim_lens)
        encoded_evidence = self._encode(embedded_evidence, evicence_lens)
        
        # Concatenate final claim and evidence vectors
        combined = torch.cat((encoded_claim, encoded_evidence), dim=1)
        
        # Feed-forward through two linear layers with dropout
        output = self.dropout(self.fc2(self.fc1(combined)))

        # Return raw logits and the softmax probabilities
        return output, self.activation(output)

class LstmAttentionClassifier(nn.Module):
    """
    LstmAttentionClassifier implements an ESIM-like architecture for Claim-Evidence matching.
    
    The pipeline:
      1) Embedding & BiLSTM encoding for claim and evidence.
      2) Cross-attention between claim and evidence sequences.
      3) Composition through another BiLSTM.
      4) Pooling (max & average) of the resulting sequences.
      5) Final classification layer generating output logits.
    """

    def __init__(self,
                 vocab_size,
                 pad_id=0,
                 embed_dim=300,
                 hidden_size=256,
                 num_classes=2,
                 embedding_mat=None,
                 train_embedding=False,
                 dropout_rate=0.1):
        
        super(LstmAttentionClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.pad_id = pad_id

        self.dropout = nn.Dropout(dropout_rate)
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        if embedding_mat is not None:
            self.embedding.weight.data.copy_(embedding_mat)
            # If train the word embeddings
            self.embedding.weight.requires_grad = train_embedding
        
        # biLSTM encoder
        self.bilstm_encoder = nn.LSTM(input_size=embed_dim,
                                       hidden_size=hidden_size,
                                       batch_first=True,
                                       bidirectional=True)
        # F projection
        self.F = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                        self.hidden_size),
                                        nn.ReLU(),
                                        self.dropout)
        
        # composition
        self.bilstm_composition = nn.LSTM(input_size=hidden_size,
                                          hidden_size=hidden_size,
                                          batch_first=True,
                                          bidirectional=True)
        
        # classify head
        self.classifier = nn.Sequential(self.dropout,
                                        nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                        nn.Tanh(),
                                        self.dropout,
                                        nn.Linear(self.hidden_size,
                                                       self.num_classes),
                                        nn.Softmax(dim=-1))
        
        

    def _encode(self, x, lengths):
        # pack
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # LSTM
        packed_out, _ = self.bilstm_encoder(packed_x)
        # pad
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out
    
    def _compose(self, x, lengths):
        """
        x: [B, seq_len, 8*hidden_size]
        lengths: [B], each is the valid length
        return: [B, seq_len, 2*hidden_size]
        """
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm_composition(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out
    
    def _masked_softmax(self, logits, mask, dim=-1, eps=1e-13):
        """
        logits: [B, Lc, Le] or [B, Le], ...
        mask: [B, Le] or [B, Lc, Le]
        
        return softmax applied tensor with masked position nearly =0
        """

        # print(logits.shape, mask.shape)

        mask = mask.to(logits.dtype)
        # set a extreme small value to masked position that gives 0 after softmax
        masked_logits = logits + (1 - mask) * eps 
        return nn.functional.softmax(masked_logits, dim=dim)
    
    def _softmax_attention(self, claim_mask, claim_encoded, evidence_mask, evidence_encoded):

        # claim_encoded: [B, Lc, 2H], evidence_encoded: [B, Le, 2H]
        # Calculate similarity Matrix [B, Lc, Le]
        similarity_matrix = torch.matmul(claim_encoded, evidence_encoded.transpose(1, 2).contiguous())  
        
        # calculate attented claim
        # => mask shape [B, 1, Le], using broadcast to multiply [B, Lc, Le]
        evidence_mask_for_claim = evidence_mask.unsqueeze(1)  # [B, 1, Le]
        claim2evidence = self._masked_softmax(similarity_matrix, evidence_mask_for_claim, dim=2)  # [B, Lc, Le]
        attended_claim = claim2evidence.bmm(evidence_encoded) # [B, Lc, 2H]
        
        # calculate attented evidence
        # same as claim
        claim_mask_for_evidence = claim_mask.unsqueeze(1)  # [B, 1, Lc]
        evidence2claim = self._masked_softmax(similarity_matrix.transpose(1,2).contiguous(),
                              claim_mask_for_evidence, dim=2)  # [B, Le, Lc]
        attended_evidence = evidence2claim.bmm(claim_encoded) # [B, Le, 2H]

        return attended_claim, attended_evidence

    
    def forward(self, claim_ids, claim_lens, evidence_ids, evidence_lens):
        """
        Perform a forward pass through the model architecture:
         - 1. Embed claim and evidence
         - 2. Encode them via a BiLSTM
         - 3. Compute cross-attention between claim and evidence
         - 4. Construct enhanced representations
         - 5. Apply another BiLSTM (composition)
         - 6. Pool (max & avg) the outputs
         - 7. Pass through the final classification head
         
        :param claim_ids: [B, Lc] Claim token IDs
        :param claim_lens: [B] Valid lengths for claim sequences
        :param evidence_ids: [B, Le] Evidence token IDs
        :param evidence_lens: [B] Valid lengths for evidence sequences
        
        :return: logits -> [B, num_classes]
        """

        # ========== 1. Embedding Layer ==========
        claim_embed = self.embedding(claim_ids)        # [B, Lc, embed_dim]
        evidence_embed = self.embedding(evidence_ids)  # [B, Le, embed_dim]
        
        claim_embed = self.dropout(claim_embed)
        evidence_embed = self.dropout(evidence_embed)
        
        # ========== 2. BiLSTM Encoding Layer ==========
        # output: [B, Lc, 2H] / [B, Le, 2H]
        claim_encoded = self._encode(claim_embed, claim_lens)
        evidence_encoded = self._encode(evidence_embed, evidence_lens)
        
        # ========== 3. Softmax Attention ==========

        # build attention mask based on whether id==pad_id
        batch_size, max_Lc, _ = claim_encoded.size()
        _, max_Le, _ = evidence_encoded.size()

        claim_mask = (claim_ids != self.pad_id)[:batch_size, :max_Lc].float()  # [B, Lc]
        evidence_mask = (evidence_ids != self.pad_id)[:batch_size, :max_Le].float()  # [B, Le]

        attended_claim, attended_evidence = self._softmax_attention(claim_mask, 
                                                                    claim_encoded, 
                                                                    evidence_mask, 
                                                                    evidence_encoded)
        
        # ========== 4. Difference Enhancement ==========
        claim_diff = claim_encoded - attended_claim
        claim_mul  = claim_encoded * attended_claim
        claim_combined = torch.cat([claim_encoded, attended_claim, claim_diff, claim_mul], dim=-1)# [B, Lc, 8H]
        
        evidence_diff = evidence_encoded - attended_evidence
        evidence_mul  = evidence_encoded * attended_evidence
        evidence_combined = torch.cat([evidence_encoded, attended_evidence, evidence_diff, evidence_mul], dim=-1)# [B, Le, 8H]

        
        # ========== 5. Composition BiLSTM ==========

        # Dense layer for parameter reduction: 2H -> H
        claim_combined = self.F(claim_combined)
        evidence_combined = self.F(evidence_combined)

        claim_composed = self._compose(claim_combined, claim_lens)       # [B, Lc, 2H]
        evidence_composed = self._compose(evidence_combined, evidence_lens) # [B, Le, 2H]
        
        # ========== 6. Max & Avg Pooling ==========
        
        # build mask with aligned dims for computing max and avg
        claim_mask_3d = claim_mask.unsqueeze(-1)  # [B, Lc, 1]
        evidence_mask_3d = evidence_mask.unsqueeze(-1)  # [B, Le, 1]
        
        # Max & Avg for claim
        claim_len_float = claim_lens.unsqueeze(-1).float() # [B, 1]
        claim_composed_masked = claim_composed * claim_mask_3d  # [B, Lc, 2H]
        claim_avg = claim_composed_masked.sum(dim=1) / claim_len_float  # [B, 2H]
        claim_max, _ = (claim_composed_masked + (claim_mask_3d - 1)*1e-9).max(dim=1) # [B, 2H]
        
        # Max & Avg for evidence
        evidence_len_float = evidence_lens.unsqueeze(-1).float()
        evidence_composed_masked = evidence_composed * evidence_mask_3d # [B, Le, 2H]
        evidence_avg = evidence_composed_masked.sum(dim=1) / evidence_len_float # [B, 2H]
        evidence_max, _ = (evidence_composed_masked + (evidence_mask_3d - 1)*1e-9).max(dim=1) # [B, 2H]
        
        # Concat for Classify Layer
        claim_pooled = torch.cat([claim_avg, claim_max], dim=1)       # [B, 4H]
        evidence_pooled = torch.cat([evidence_avg, evidence_max], dim=1) # [B, 4H]
        
        final_vec = torch.cat([claim_pooled, evidence_pooled], dim=1)  # [B, 8H]
        final_vec = self.dropout(final_vec)
        
        # ========== 7. Classification Layer ==========
        logits = self.classifier(final_vec)  # [B, num_classes]
        # probabilities = nn.functional.softmax(logits, dim=-1)
        return logits

