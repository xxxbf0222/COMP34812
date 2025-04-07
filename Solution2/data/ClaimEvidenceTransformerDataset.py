import torch
import pandas as pd

class ClaimEvidenceTransformerDataset():
    def __init__(self, dataframe, tokenizer, load_from=None, max_length=256, dataset_type="train"):
        """
        A dataset class for handling Claim-Evidence pairs using a Transformer tokenizer.

        :param dataframe: CSV file path or a pandas DataFrame containing columns 'Claim', 'Evidence', and possibly 'label'.
        :param tokenizer: A BERT tokenizer load from transformer api (e.g., DeBERTa).
        :param load_from: If provided, load a preprocessed dataset from disk using torch.load(...).
        :param max_length: The maximum sequence length for tokenization (default 256).
        :param dataset_type: One of ['train', 'validation', 'test']. 
                            If it is 'test', 'with_label' will be False.
        """

        # Check that dataset_type is one of the valid options
        assert dataset_type in ["train","validation", "test"]
        # Ensure dataset are loading from pandas or saved torch pre-processed file
        assert (load_from!=None) or (dataframe is not None)

        self.data = []
        # 'with_label' = True if it's not a test set, meaning we have ground truth labels
        self.with_label = (dataset_type != "test")

        # If load_from is specified, skip preprocessing and directly load from a saved dataset
        if load_from!=None:
            print("Load dataset from:", load_from)
            loaded_dataset = torch.load(load_from, weights_only=False)
            # Copy over the loaded dataset's attributes into the current instance
            self.__dict__.update(loaded_dataset.__dict__)
            print("Done!")
            return
        
        # If dataframe is a file path (string), read it, otherwise assume it's already a DataFrame object
        file = pd.read_csv(dataframe) if type(dataframe) is str else dataframe
        
        # Iterate through each row to build the tokenized dataset
        for idx in range(len(file)):
            claim_text, evidence_text = file["Claim"][idx], file["Evidence"][idx]
            if self.with_label:
                label = file["label"][idx]
            
            # Use the tokenizer to encode the Claim-Evidence pair
            encoded_claim_evidence = tokenizer(
                text=claim_text,
                text_pair=evidence_text,
                truncation=True, 
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Attach label to the encoding if it's a train/validation set
            if self.with_label:
                encoded_claim_evidence["labels"] = torch.tensor(label, dtype=torch.float)
            
            self.data.append(encoded_claim_evidence)
            print(f"Loading dataset...   row ({idx+1}/{file.shape[0]})", end="\r")
        print("\nDone!")
        
    def save_dataset(self, save_path):
        """
        Save the current dataset object to disk using torch.save(...).
        """
        torch.save(self, save_path, pickle_protocol=5)
        print("Dataset saved to:", save_path)
    
    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieve a single item of tokenized data by index.
        """
        return self.data[idx]
