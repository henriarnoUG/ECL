import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.tensorboard import SummaryWriter

        
class CustomDataset(Dataset):
  
    
    def __init__(self, dataframe, tokenizer, path_corpus):
        
        """
        Args:
            dataframe (pd.DataFrame): ECL dataset as a dataframe (does not need to be matched with Compustat data)
            tokenizer (transformers.PreTrainedTokenizer): pre-trained tokenizer to tokenize documents
            path_corpus (str): path to corpus with text files
        """
        
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.path_corpus = path_corpus
        

    def __len__(self):
        
        """
        Returns:
            int: number of samples in the dataset
        """
        
        return len(self.data)
    
    

    def __getitem__(self, idx):
        
        """
        Args:
            idx (int): idx of the sample to retrieve

        Returns:
            dict: dictionary containing 'input_ids', 'attention_mask', and 'labels' of retrieved sample
        """
        
        # get text and label
        text_file = self.path_corpus + self.data['filename'].iloc[idx]
        text = open(text_file, 'r', encoding="utf8").read()
        labels = int(self.data['label'].iloc[idx])
        
        # encode text
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # return
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.long)
        }
