import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from Solution1.data.POStokenizer import POS_tokenizer

class ClaimEvidenceLstmDataset():
    """
    This dataset class handles the claim-evidence pairs for the LSTM-attetion model.
    It supports building vocabulary for training, or loading vocabularies for val/test dataset
    Also supports different pretrained embeddings, tokenizing text, and converting them into padded sequences 
    for PyTorch DataLoader.
    """
    def __init__(self, dataframe=None, 
                 dataset_type="train",
                 vocab=None, 
                 tokenizer=POS_tokenizer(),
                 pre_trained_embedding=None,
                 load_from=None):
        """
        :param dataframe: Could be a file path (CSV) or a pandas DataFrame object.
        :param dataset_type: 'train', 'validation', or 'test'. Affects whether labels exist and if vocab can be expanded.
        :param vocab: The existing vocabulary if not in 'train' mode.
        :param tokenizer: A tokenizer object (default = POS_tokenizer).
        :param pre_trained_embedding: Preloaded word embedding dictionary (word -> vector).
        :param load_from: If specified, loads a pre-processed dataset from the given file path.
        """

        # Check that dataset_type is one of the valid options
        assert dataset_type in ["train","validation", "test"]
        # Ensure dataset are loading from pandas or saved torch pre-processed file
        assert (load_from!=None) or (dataframe is not None)
        # If the dataset is not for training, the vocabulary must provided for text preprocessing
        if dataset_type!="train":
            assert vocab != None

        # If loading from an saved preprocessed dataset, skip building vocab/embeddings
        if load_from!=None:
            print("Load dataset from:", load_from)
            loaded_dataset = torch.load(load_from, weights_only=False)
            self.__dict__.update(loaded_dataset.__dict__)
            print("Done!")
            return

        self.dataset_type = dataset_type
        self.padded_claim_seqs = []
        self.claim_lengths = []
        self.padded_evidence_seqs = []
        self.evidence_lengths = []
        self.labels = []
        self.with_label = (dataset_type != "test")  # If 'test', no labels
        
        self.tokenizer = tokenizer
        
        # If load-from not provided, load pretrained embeddings for the 'train' set to build vocabulary and embedding mat
        if pre_trained_embedding:
            self.pre_trained_embedding = pre_trained_embedding 
        elif dataset_type == "train":
            print("Loading 'word2vec-google-news-300' as pretrained word embeddings...")
            self.pre_trained_embedding = torch.load("./Solution1/data/vocab&embeddings/word2vec-google-news-300.pt", weights_only=False)
            
        # For train mode, create or initialize a new vocab; for validation/test, must have an existing vocab
        self.vocab = vocab if dataset_type!="train" else {"<unk>":1, "<pad>":0}
        self.embedding_mat = None
        
        self.padding_value = 0
        
        # If 'dataframe' is a string, treat it as CSV path; otherwise assume it's a pandas DataFrame class
        file = pd.read_csv(dataframe) if type(dataframe) is str else dataframe
        
        claim_seqs = []
        evidence_seqs = []
        
        # ============= Building Vocab & Sequences in Train Mode =============
        if self.dataset_type == "train":
            all_tokenized_c = []
            all_tokenized_e = []
            for row in file.iterrows():
                claim, evidence = row[1]["Claim"], row[1]["Evidence"]
                if self.with_label:
                    label = row[1]["label"]
                    self.labels.append(label)
                
                # Tokenize the text for both claim and evidence
                tokenized_claim = self.tokenize(claim)
                tokenized_evidence = self.tokenize(evidence)
                all_tokenized_c.append(tokenized_claim)
                all_tokenized_e.append(tokenized_evidence)
    
                # Expand vocab with new tokens
                self.expand_vocab(tokenized_claim)
                self.expand_vocab(tokenized_evidence)
                
                print(f"Loading training dataset and building vocab...  row ({row[0]+1}/{file.shape[0]})", end="\r")

            print("\nBuilding Word Embeddings...")
            # Build embedding matrix (aligned with self.vocab)
            self.build_embedding_mat()
            
            print("Converting Text to Sequences...")
            # Convert tokenized text into numeric sequences
            claim_seqs = [self.text_to_seq_tensor(tokenized_claim) for tokenized_claim in all_tokenized_c]
            evidence_seqs = [self.text_to_seq_tensor(tokenized_evidence) for tokenized_evidence in all_tokenized_e]
            
        # ============= Using Existing Vocab in Validation/Test Mode =============
        else:
            for row in file.iterrows():
                claim, evidence = row[1]["Claim"], row[1]["Evidence"]
                if self.with_label:
                    label = row[1]["label"]
                    self.labels.append(label)
                    
                tokenized_claim = self.tokenize(claim)
                tokenized_evidence = self.tokenize(evidence)
                claim_seqs.append(self.text_to_seq_tensor(tokenized_claim))
                evidence_seqs.append(self.text_to_seq_tensor(tokenized_evidence))
                print(f"Loading {self.dataset_type} dataset...   row ({row[0]+1}/{file.shape[0]})", end="\r")
            
            print("\nConverting Text to Sequences...")
            print("Using exist vocab...")
        
        # Pad the sequences to build final Tensors for claim & evidence
        self.padded_claim_seqs, self.claim_lengths = self.pad_sequences(claim_seqs)
        self.padded_evidence_seqs, self.evidence_lengths = self.pad_sequences(evidence_seqs)
        
        # After tokenize and encoding, empty sequences are replaced with <unk>
        self.pad_seq_with_zero_length()
        
        print("Done!")
        
    def tokenize(self, text):
        """
        Uses self.tokenizer (POS_tokenizer by default) to split text into a list of tokens.
        """
        return self.tokenizer.tokenize(text)
        
    def expand_vocab(self, tokens):
        """
        Adds new tokens to the vocabulary if they exist in the pretrained embedding.
        """
        for token in tokens:
            if (token not in self.vocab) and (token in self.pre_trained_embedding):
                idx = len(self.vocab)
                self.vocab[token] = idx  # assign next available index
    
    def build_embedding_mat(self):
        """
        Builds a torch.Tensor [vocab_size, embedding_dim] 
        containing the pre-trained vectors for tokens in self.vocab.
        """
        embedding_dim = len(self.pre_trained_embedding[0])
        embedding_mat = np.zeros((len(self.vocab), embedding_dim), dtype=np.float32)
        
        for word, i in self.vocab.items():
            if i >= 2:
                embedding_mat[i] = self.pre_trained_embedding[word]
            else:
                # <unk> and <pad> are randomly initialized
                embedding_mat[i] = np.random.uniform(-0.01, 0.01, embedding_dim) 
        
        embedding_mat = torch.tensor(embedding_mat)
        self.embedding_mat = embedding_mat
    
    def text_to_seq_tensor(self, text):
        """
        Convert a list of tokens into a torch.Tensor of token indices 
        (mapping each token to self.vocab index).
        """
        seq = []
        for token in text:
            if token in self.vocab:
                word_idx = self.vocab[token]
                seq.append(word_idx)
        return torch.tensor(seq)

    def pad_sequences(self, seqs):
        """
        Pad a list of variable-length sequences to a uniform length 
        using torch.nn.utils.rnn.pad_sequence. 
        Returns (padded_sequences, lengths).
        """
        lengths = [len(seq) for seq in seqs]
        padded_seq = pad_sequence(seqs, batch_first=True, padding_value=self.padding_value)
        return padded_seq, lengths

    def pad_seq_with_zero_length(self):
        """
        Ensures that any sequence with zero length is assigned length=1 
        and the first token replaced with <unk>, preventing empty sequences.
        """
        for idx in range(len(self.padded_claim_seqs)):
            if self.claim_lengths[idx] == 0:
                self.claim_lengths[idx] = 1
                self.padded_claim_seqs[idx, 0] = self.vocab["<unk>"]
            elif self.evidence_lengths[idx] == 0:
                self.evidence_lengths[idx] = 1
                self.padded_evidence_seqs[idx, 0] = self.vocab["<unk>"]
    
    def save_dataset(self, save_path):
        """
        Saves the current dataset object (including processed vocab, sequences, etc.).
        """
        torch.save(self, save_path, pickle_protocol=5)
        print("Dataset saved to:", save_path)
    
    def save_embedding_mat(self, save_path):
        """
        Saves the built embedding matrix for training.
        """
        assert self.embedding_mat != None
        torch.save(self.embedding_mat, save_path, pickle_protocol=5)
        print("Word Embedding Matrix saved to:", save_path)
    
    def save_vocab(self, save_path):
        """
        Saves the vocabulary for future use.
        """
        torch.save(self.vocab, save_path, pickle_protocol=5)
        print("Vocab saved to:", save_path)
    
    def __len__(self):
        """
        Returns the total number of sequences in the dataset.
        """
        return len(self.padded_claim_seqs)

    def __getitem__(self, index):
        """
        Retrieve the claim/evidence pair (and label if present) by index.
        """
        if self.with_label:
            return {
                "id": index,
                "claim": self.padded_claim_seqs[index],
                "claim_length": self.claim_lengths[index],
                "evidence": self.padded_evidence_seqs[index],
                "evidence_length": self.evidence_lengths[index],
                "label": self.labels[index]
            }
        else:
            return {
                "id": index,
                "claim": self.padded_claim_seqs[index],
                "claim_length": self.claim_lengths[index],
                "evidence": self.padded_evidence_seqs[index],
                "evidence_length": self.evidence_lengths[index]
            }


if __name__ == "__main__":
    print("==> Loading Training Set from file:")
    train_dataset = ClaimEvidenceLstmDataset("./data/train.csv", dataset_type="train")
    train_dataset.save_embedding_mat("./Solution1/data/default_embedding_mat.pt")
    train_dataset.save_vocab("./Solution1/data/default_vocab.pt")
