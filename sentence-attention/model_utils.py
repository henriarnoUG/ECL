# general imports
import numpy as np
import pandas as pd
import math

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SentenceDataset(Dataset):
    
    def __init__(self, dataframe):
        
        """
        Args:
            dataframe (pd.DataFrame): ECL dataset as a dataframe (does need to be matched with Compustat data)
        """
        
        self.data = dataframe.reset_index(drop=False)
        self.predictors = ['actlct','apsale','cashat','chat','chlct','ebit_dp_at','ebitat','ebitsale','fat','invchinvt','invtsale','lct_ch_at','lctat','lctlt','lctsale','ltat', 'log_at','log_sale','niat','nisale','oiadpat','oiadpsale','qalct','reat','relct','saleat','seqat','wcapat']


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
            dict: dictionary containing 'sentence_embeddings', 'sentence_masks', 'structured_features', 'labels' and 'idx' 
            (original indices for post-processing) of retrieved sample
        """

        # store path to embeddings and masks on disk - see embedding_demo.ipynb
        # change path to appropriate location
        text_path = self.data.iloc[idx]['filename'].replace('.txt', '.npy')
        embedding_path = text_path.replace('/raw_corpus/', '/embeddings/')
        mask_path = text_path.replace('/raw_corpus/', '/masks/')

        # load
        embeddings = torch.tensor(np.load(embedding_path), dtype=torch.float)
        masks = torch.tensor(np.load(mask_path), dtype=torch.float)
        

        # get the structured features
        features = self.data.iloc[idx][self.predictors]
        features = features.values.astype(np.float64)
        features = torch.tensor(features, dtype=torch.float)

        # get the labels
        labels = torch.tensor(int(self.data.iloc[idx]['label']), dtype=torch.float)

        # get the original indices
        original_ids = self.data.iloc[idx]['index']
        
        return {
            'sentence_embeddings': embeddings,
            'sentence_masks': masks,
            'structured_features': features,
            'labels': labels,
            'idx': original_ids}



def scaled_dot_product_attention(query, key, value, masks) -> torch.Tensor:
    """
    adapted from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Args:
        query (torch.tensor): representation of structured features
        key (torch.tensor): representation of sentence embeddings to compare to query
        value (torch.tensor): representation of sentence embeddings to transform into document representation
        masks (torch.tensor): mask that indicates which sentences were padded and do not take part in attention operation
    Returns:
        weighted_value (torch.tensor): attention weighted value tensors
        attn_weight (torch.tensor): attention weights
    """

    # set scaling factor
    hidden_dim = query.size(-1)
    scale_factor = 1 / math.sqrt(hidden_dim)

    # dot product
    attn_weight = query @ key.transpose(-2, -1) * scale_factor    

    # set attention score to -inf for masked sentences
    masks = masks.bool().unsqueeze(1)
    bias = torch.zeros(masks.shape, device=masks.device)
    bias.masked_fill_(masks, float("-inf"))

    # add bias to attention scores and softmax
    attn_weight += bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # weigh value vectors with attention scores
    weighted_value = attn_weight @ value
    
    return weighted_value, attn_weight



class SentenceAttentionNetwork(nn.Module):  
    
    def __init__(self, embedding_dim=384, feature_dim=28, hidden_dim=32):
        super().__init__()
        
        """
        Args:
            embedding_dim (int): dimension of sentence embeddings
            feature_dim (int): dimension of structured features
            hidden_dim (int): dimension of query, key and value representations
        """

        # linear map for structured features
        self.linear_map = nn.Linear(feature_dim, hidden_dim)

        # trainable tensor
        self.trainable = nn.Parameter(torch.randn(1, embedding_dim))

        # linear map for embeddings to key and value matrices
        self.key_layer = nn.Linear(embedding_dim, hidden_dim)
        self.value_layer = nn.Linear(embedding_dim, hidden_dim)

        # classification layer
        self.classification = nn.Linear(hidden_dim*2, 1)


    def forward(self, embeddings, masks, features):
        
        """
        Args:
            embeddings (torch.tensor): sentence embeddings
            masks (torch.tensor): sentence masks
            features (torch.tensor): structured features
        Returns:
            logits (torch.tensor): raw logits for positive class
            attn_weight (torch.tensor): attention weights of sentences (including on trainable tensor)
        """

        # store dimensions
        batch_size, sentences, embedding_dim = embeddings.shape

        # map structured features to embedding dimension
        feature_map = self.linear_map(features)

        # ensure that norm of trainable tensor is one and add to embeddings
        normed = self.trainable / torch.norm(self.trainable)
        normed_batch = normed.unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = torch.cat((normed_batch, embeddings), dim=1)

        # allow attentending to trainable vector
        trainable_mask = torch.zeros((batch_size, 1), dtype=torch.float, device=masks.device)
        masks = torch.cat([trainable_mask, masks], dim=1)

        # map embeddings to key and value matrices
        keys = self.key_layer(embeddings)
        values = self.value_layer(embeddings)
        queries = feature_map.unsqueeze(1)

        # apply SDPA
        document_representation, attn_weight = scaled_dot_product_attention(key=keys, query=queries, value=values, masks=masks)

        # remove singleton dimensions
        document_representation = document_representation.squeeze(1)
        attn_weight = attn_weight.squeeze(1)

        # clasify
        clf_input = torch.cat((document_representation, feature_map), dim=1)
        logits = self.classification(clf_input)

        return logits, attn_weight