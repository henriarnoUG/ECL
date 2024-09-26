# general imports
import pandas as pd
import numpy as np
import string
from matplotlib import pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# sklearn imports
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    auc,
    precision_recall_curve,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_curve,
)



def compute_cap_recall (labels, predictions, k, verbose=False):
    """
    Args:
        labels (list): true labels
        predictions (list): predicted probabilities for positive class
        k (int): number of top predictions to consider
        verbose (bool): If True, plot CAP curve

    Returns:
        float: cap (cumulative accuracy profile) and recall@k metrics
    """

    # cast labels to appropriate datatypes
    labels = [bool(x) for x in labels]
    
    # count instances
    total = len(labels)
    positives = sum(labels)
    negatives = total - positives
        
    # sort on predicted probability
    sorted_predictions, sorted_labels = zip(*sorted(zip(predictions, labels), reverse=True))
    recall = sum(sorted_labels[:k]) / positives
    
    # compute proportion of data and recall for (1) evaluated model, (2) random model and (3) perfect model
    x = np.arange(0, total + 1) / total
    y = np.append([0], np.cumsum(sorted_labels)) / positives
    
    x_random = [0, 1.0]
    y_random = [0, 1.0]

    x_perf = [0, positives / total, 1.0]
    y_perf = [0, 1.0, 1.0]
    
    
    # compute cap
    r = auc(x_random, y_random)
    p = auc(x_perf, y_perf)
    c = auc(x, y)
    aP = p - r
    aR = c - r
    cap = (aR / aP)
        
        
    # optionally plot
    if verbose:
        fig, ax1 = plt.subplots()
        ax1.plot(x, y, c = 'b', label = 'clf', linewidth = 2)
        ax1.plot(x_random, y_random, c = 'b', linestyle = '--', label = 'random')
        ax1.plot(x_perf, y_perf, c = 'grey', linewidth = 2, label = 'perfect')
        ax1.legend()
        ax1.set_xlabel('proportion of data')
        ax1.set_ylabel('recall')
        ax1.set_title('cap curve')
        
    return cap, recall






def evaluate(labels, predictions):
    """
    Args:
        labels (list): true labels
        predictions (list): predicted probabilities for positive class

    Returns:
        prints AP, AUC, CAP en recall@100 metrics
    """
    
    # compute metrics
    ap = average_precision_score(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    cap, recall = compute_cap_recall(labels, predictions, k=100, verbose=False)

    print('-- RESULTS --')
    print(f'AUC: {auc:.4f}')
    print(f'AP: {ap:.4f}')
    print(f'recall@100: {recall:.4f}')
    print(f'CAP: {cap:.4f}')
          
        

        
        
def tokenize_lemmatize(text):
    """
    Args:
        text (string): text to process 
    Returns: 
        tokens (list): lemmatized tokens
    """ 
    
    # lowercase and tokenize
    text = str(text).lower()
    text = nltk.word_tokenize(text)
    
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    return tokens






def remove_stop_punct_num(tokenized_text):
    """
    Args:
        tokenized_text (list): tokens
    Returns:
        processed_text (list): tokens without punctuation, stopwords, and numerals
    """

    # init
    stop_words = stopwords.words('english')
    punct = list(string.punctuation)
    
    # filter
    processed_text = []

    if not isinstance(tokenized_text, list):
        return []

    for word in tokenized_text:
        if (word not in stop_words) and (word not in punct) and (not any(char.isdigit() for char in word)):
            processed_text.append(word)

    return processed_text

