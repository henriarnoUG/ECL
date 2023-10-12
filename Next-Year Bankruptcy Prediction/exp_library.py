import pandas as pd
import numpy as np
import string

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

from matplotlib import pyplot as plt

import xgboost as xgb

from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords


# Add custom function to compute CAP
# source 1: https://www.geeksforgeeks.org/python-cap-cumulative-accuracy-profile-analysis/
# source 2: https://towardsdatascience.com/machine-learning-classifier-evaluation-using-roc-and-cap-curves-7db60fe6b716

def CAP_recall (labels, predictions, k, verbose = False):
    
    """
    Compute the Cumulative Accuracy Profile (CAP) and Recall@K.

    Args:
        labels (list): True labels (binary).
        predictions (list): Predicted probabilities.
        k (int): Number of top predictions to consider.
        verbose (bool, optional): Whether to display the CAP curve plot.

    Returns:
        float: CAP AR (Cumulative Accuracy Profile Area Ratio).
        float: Recall@K.

    """

    # cast labels to boolean
    labels = [bool(x) for x in labels]
    
    # compute number of instances, positives and negatives
    total = len(labels)
    positives = sum(labels)
    negatives = total - positives
        
    # Sort the labels based on predicted probability
    sorted_predictions, sorted_labels = zip(*sorted(zip(predictions, labels), reverse=True))
    recall = sum(sorted_labels[:k]) / positives
    
    # Compute the proportion of the data at each datapoint
    x = np.arange(0, total + 1) / total
    # Compute the proportion of positive samples retained at each datapoint
    y = np.append([0], np.cumsum(sorted_labels)) / positives
    
    
    # Get the values to plot for a random and a perfect model
    x_random = [0, 1.0]
    y_random = [0, 1.0]

    x_perf = [0, positives / total, 1.0]
    y_perf = [0, 1.0, 1.0]
    
    
    # Compute metrics
    r = auc(x_random, y_random)
    p = auc(x_perf, y_perf)
    c = auc(x, y)
    # Area between Perfect and Random Model
    aP = p - r
    # Area between Trained and Random Model
    aR = c - r
    
    # CAP-ratio
    CAP_AR = (aR / aP)
        
    # Create the plot if verbose is enabled
    if verbose:
        fig, ax1 = plt.subplots()
        ax1.plot(x, y, c = 'b', label = 'Classifier', linewidth = 2)
        ax1.plot(x_random, y_random, c = 'b',
                 linestyle = '--', label = 'Random Model')
        ax1.plot(x_perf, y_perf,
                 c = 'grey', linewidth = 2, label = 'Perfect Model')
        ax1.legend()
        ax1.set_xlabel('% of data (ranked according to predicted probability)')
        ax1.set_ylabel('Cumulative % postive cases')
        ax1.set_title('CAP Curve - CAP AR = '  + str(np.round(CAP_AR, 4)))
        
    return CAP_AR, recall




def evaluate(labels, predictions):
    """
    Evaluate a classification model using various performance metrics.

    Args:
        labels (list): True labels (binary).
        predictions (list): Predicted probabilities.

    Prints:
        - Average Precision (AP)
        - Area Under the ROC Curve (AUC)
        - Cumulative Accuracy Profile (CAP)
        - Recall@100

    """
    # Compute Average Precision (AP)
    AP = average_precision_score(labels, predictions)
    
    # Compute Area Under the ROC Curve (AUC)
    AUC = roc_auc_score(labels, predictions)
    
    # Compute Cumulative Accuracy Profile (CAP) and Recall@100
    CAP, recall = CAP_recall(labels, predictions, k=100, verbose=False)

    print('-- RESULTS --')
    print('AUC: ' + str(np.round(AUC, 4)))
    print('AP: ' + str(np.round(AP, 4)))
    print('recall@100: ' + str(np.round(recall, 4)))
    print('CAP: ' + str(np.round(CAP, 4)))
    
          
          
          
          
def performance_curves (labels, predictions):
    """
    Generate and display performance curves including CAP, PR-curve, and ROC-curve.

    Args:
        labels (list): True labels (binary).
        predictions (list): Predicted probabilities.
    """
    
    # CAP curve
    CAP_AR, _ = CAP_recall(labels, predictions, 100, True)

    # PR-curve
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.show()

    # ROC-curve
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR (Sensitivity)')
    plt.show()

    return CAP_AR


def tokenize_lemmatize(text):
    """
    Tokenize and lemmatize a text.
    
    :param text: Input string to process.
    :return: List of lemmatized tokens.
    """
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Convert to lowercase
    text = str(text).lower()
    # Tokenize
    text = nltk.word_tokenize(text)
    # Lemmatize
    text = [lemmatizer.lemmatize(word) for word in text]
    # Return the result
    return text

def remove_stop_punct_num(tokenized_text):
    """
    Remove punctuation, stopwords, and numerals from a list of tokens.
    
    :param tokenized_text: List of tokens.
    :return: List of tokens without punctuation, stopwords, and numerals.
    """
    # Initialize stopwords, punctuation, and an empty list for output
    stop_words = stopwords.words('english')
    punct = list(string.punctuation)
    filtered_sentence = []

    # If the input is empty, return an empty list
    if not isinstance(tokenized_text, list):
        return []

    # Perform filtering
    for word in tokenized_text:
        if (word not in stop_words) and (word not in punct) and (not any(char.isdigit() for char in word)):
            filtered_sentence.append(word)

    return filtered_sentence

