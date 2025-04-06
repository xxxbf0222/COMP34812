from POStokenizer import POS_tokenizer
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np

class ClaimEvidenceLstmDataset():
    def __init__(self, file_path=None, 
                 type="train",
                 vocab=None, 
                 tokenizer=POS_tokenizer(),
                 pre_trained_embedding=None,
                 load_from=None):
        
        
        assert type in ["train","validation", "test"]
        assert (load_from!=None) or (file_path!=None)
        if type!="train":
            assert vocab != None

        if load_from!=None:
            print("Load dataset from:", load_from)
            loaded_dataset = torch.load(load_from, weights_only=False)
            self.__dict__.update(loaded_dataset.__dict__)
            print("Done!")
            return

        self.type = type
        self.padded_claim_seqs = []
        self.claim_lengths = []
        self.padded_evidence_seqs = []
        self.evidence_lengths = []
        self.labels = []
        self.with_label = (type != "test")
        
        self.tokenizer = tokenizer
        
        if pre_trained_embedding:
            self.pre_trained_embedding = pre_trained_embedding 
        elif type == "train":
            print("Loading 'word2vec-google-news-300' as pretrained word embeddings...")
            self.pre_trained_embedding = torch.load("./data/word2vec-google-news-300.pt", weights_only=False)
            
        self.vocab = vocab if type!="train" else {"<unk>":1, "<pad>":0}
        self.embedding_mat = None
        

        
        self.padding_value = 0
        
        file = pd.read_csv(file_path)
        
        claim_seqs = []
        evidence_seqs = []
        if self.type == "train":
            all_tokenized_c = []
            all_tokenized_e = []
            for row in file.iterrows():
                claim, evidence = row[1]["Claim"], row[1]["Evidence"]
                if self.with_label:
                    label = row[1]["label"]
                    self.labels.append(label)
                    
                tokenized_claim, tokenized_evidence = self.tokenize(claim), self.tokenize(evidence)
                all_tokenized_c.append(tokenized_claim)
                all_tokenized_e.append(tokenized_evidence)
    
                self.expand_vocab(tokenized_claim)
                self.expand_vocab(tokenized_evidence)
                
                print(f"Loading training dataset and building vocab...  row ({row[0]}/{file.shape[0]})",end="\r")

            print("\nBuilding Word Embeddings...")
            self.build_embedding_mat()
            print("Converting Text to Sequences...")
            claim_seqs = [self.text_to_seq_tensor(tokenized_claim) for tokenized_claim in all_tokenized_c]
            evidence_seqs = [self.text_to_seq_tensor(tokenized_evidence) for tokenized_evidence in all_tokenized_e]
            
        
        else:
            for row in file.iterrows():
                claim, evidence = row[1]["Claim"], row[1]["Evidence"]
                if self.with_label:
                    label = row[1]["label"]
                    self.labels.append(label)
                    
                tokenized_claim, tokenized_evidence = self.tokenize(claim), self.tokenize(evidence)
                claim_seqs.append(self.text_to_seq_tensor(tokenized_claim))
                evidence_seqs.append(self.text_to_seq_tensor(tokenized_evidence))
                print(f"Loading {self.type} dataset...   row ({row[0]}/{file.shape[0]})",end="\r")
            
            print("\nConverting Text to Sequences...")
            print("Using exist vocab...")
        
        self.padded_claim_seqs, self.claim_lengths = self.pad_sequences(claim_seqs)
        self.padded_evidence_seqs, self.evidence_lengths = self.pad_sequences(evidence_seqs)
        self.pad_seq_with_zero_length()
        
        print("Done!")
        
    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
        
    def expand_vocab(self, tokens):
        for token in tokens:
            if (token not in self.vocab) and (token in self.pre_trained_embedding):
                idx = len(self.vocab)
                self.vocab[token] = idx # word index
                
    def build_embedding_mat(self):
        embedding_dim = len(self.pre_trained_embedding[0])
        embedding_mat = np.zeros((len(self.vocab), embedding_dim),dtype=np.float32)
        
        for word,i in self.vocab.items():
            if i >= 2:
                embedding_mat[i] = self.pre_trained_embedding[word]
            else:
                # <unk> and <pad> will not use in training and is initialised randomly
                embedding_mat[i] = np.random.uniform(-0.01, 0.01, embedding_dim) 
        
        embedding_mat = torch.tensor(embedding_mat)
        self.embedding_mat = embedding_mat
    
    def text_to_seq_tensor(self,text):
        seq = []
        for token in text:
            if token in self.vocab:
                word_idx = self.vocab[token]
                seq.append(word_idx)
        return torch.tensor(seq)

    def pad_sequences(self, seqs):
        lengths = [ len(seq) for seq in seqs ]
        padded_seq = pad_sequence(seqs, batch_first = True, padding_value = self.padding_value)
        
        return padded_seq, lengths

    def pad_seq_with_zero_length(self):
        for idx in range(len(self.padded_claim_seqs)):
            if self.claim_lengths[idx] == 0:
                self.claim_lengths[idx] = 1
                self.padded_claim_seqs[idx, 0] = self.vocab["<unk>"]
            elif self.evidence_lengths[idx] == 0:
                self.evidence_lengths[idx] = 1
                self.padded_evidence_seqs[idx, 0] = self.vocab["<unk>"]
    
    def save_dataset(self, save_path):
        torch.save(self, save_path, pickle_protocol=5)
        print("Dataset saved to:",save_path)
    
    def save_embedding_mat(self, save_path):
        assert self.embedding_mat != None
        torch.save(self.embedding_mat, save_path, pickle_protocol=5)
        print("Word Embedding Matrix saved to:",save_path)
    
    def save_vocab(self, save_path):
        torch.save(self.vocab, save_path, pickle_protocol=5)
        print("Vocab saved to:",save_path)
    
    def __len__(self):
        return len(self.padded_claim_seqs)

    def __getitem__(self, index):
        if self.with_label:
            return {"id": index,
                    "claim": self.padded_claim_seqs[index],
                    "claim_length": self.claim_lengths[index],
                    "evidence": self.padded_evidence_seqs[index],
                    "evidence_length": self.evidence_lengths[index],
                    "label": self.labels[index]}
        else:
            return {"id": index,
                    "claim": self.padded_claim_seqs[index],
                    "claim_length": self.claim_lengths[index],
                    "evidence": self.padded_evidence_seqs[index],
                    "evidence_length": self.evidence_lengths[index]}
        


if __name__ == "__main__":

    print("==> Loading Training Set from file:")
    train_dataset = ClaimEvidenceLstmDataset("./data/train.csv", type="train")
    train_dataset.save_dataset("./data/train_dataset.pt")
    train_dataset.save_embedding_mat("./data/default_embedding_mat.pt")
    train_dataset.save_vocab("./data/default_vocab.pt")

    print("\n==> Loading Development Set from file:")
    val_dataset = ClaimEvidenceLstmDataset("./data/dev.csv",
                                   vocab=train_dataset.vocab,
                                   type="validation")
    val_dataset.save_dataset("./data/dev_dataset.pt")