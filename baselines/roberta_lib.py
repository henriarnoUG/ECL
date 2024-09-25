import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.tensorboard import SummaryWriter

        
class CustomDataset(Dataset):
    """
    Custom dataset class for text classification tasks.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing 'filename' and 'label' columns.
        tokenizer (transformers.PreTrainedTokenizer): A pre-trained tokenizer for text encoding.
        text_folder (str): The path to the folder containing text files.

    Attributes:
        data (pd.DataFrame): The input DataFrame with 'filename' and 'label' columns.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for text encoding.
        text_folder (str): The path to the folder containing text files.
    """
    
    def __init__(self, dataframe, tokenizer, text_folder):
        """
        Initialize the CustomDataset.

        Args:
            dataframe (pd.DataFrame): A DataFrame containing 'filename' and 'label' columns.
            tokenizer (transformers.PreTrainedTokenizer): A pre-trained tokenizer for text encoding.
            text_folder (str): The path to the folder containing text files.
        """
        self.data = dataframe.reset_index(drop=True) # reset the dataframe indices!
        self.tokenizer = tokenizer
        self.text_folder = text_folder

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
        """
        text_file = self.text_folder + self.data['filename'].iloc[idx]
        text = open(text_file, 'r', encoding="utf8").read()
        labels = int(self.data['label'].iloc[idx])  # Convert boolean to integer
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.long)
        }
