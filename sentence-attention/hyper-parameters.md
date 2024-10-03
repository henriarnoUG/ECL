# Hyperparameter optimisation procedure

#### Overview
In this document, we describe the hyperparameter optimisation procedure from (Arno et al., 2024) applied to all next-year business failure prediction models. The 10K filings from the ECL benchmark dataset are split into training, validation, and test sets using a temporal split (i.e., based on the filing year). For each model, hyperparameters are selected to maximize the ROC-AUC on the validation set. After tuning, the models are retrained on the combined training and validation sets and then evaluated on the test set. Below, we provide tables detailing the pre-processing steps, the explored hyperparameters, and the optimization strategy for each model.

#### Numerical models 

The `N-Z'(num)` model is a logistic regression classifier without regularisation. As pre-processing step, the constructed features were mean-imputed. No hyperparameters were tuned.

The  `N-LR(num)` model is a logistic regression classifier with L2-regularisation. As pre-processing steps, the numerical features were mean-imputed and normalised. The following hyperparameters were tuned:

| Parameters | Explored | Selected |
|------------------------|----------|----------|
| neg/pos | [1, 2, 4, 122] | 1 |
| C (L2) | [1e-5, 1e-2, 1, 100] | 1e-2 |

The `N-MLP(num)` model is an MLP classifier with L2-regularisation and an Adam optimiser. As pre-processing steps, the numerical features were mean-imputed and normalised. The following hyperparameters were tuned:

| Parameters  | Explored | Selected |
|------------------------|----------|----------|
| neg/pos | [1, 2, 4, 122] | 2 |
| C (L2) | [1e-5, 1e-2, 1, 100] | 1 |
| hidden_layer_sizes | [(50), (50,50), (100), (100,100)] | (100,100) |
| learning_rate_init | (1e-5, 1e-3, 1e-1) | 1e-3 |

The `N-XGB(num)` model is an XGBoost classifier. As pre-processing steps, the numerical features were normalised. The following hyperparameters were tuned:

| Parameters| Explored | Selected |
|------------------------|----------|----------|
| neg/pos | [1, 2, 4, 122] | 1 |
| eta | [1e-5, 1e-3, 1e-1] | 1e-1 |
| max_depth | [1, 3, 7] | 1 |
| subsample | [0.5, 1] | 0.5 |
| n_estimators | [10, 100, 1000] | 1000 |

For documentation on these hyperparameters see: [`N-LR(num)`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression), [`N-MLP(num)`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), [`N-XGB(num)`](https://xgboost.readthedocs.io/en/stable/parameter.html). `neg/pos` refers to the fraction of negative over positive samples in the training set obtained by randomly oversampling the minority class instances.



#### Textual models 

The  `T-LR(tfidf)` model is a logistic regression classifier with L1-regularisation using `tfidf` features as input. As pre-processing steps, the text to construct the `tfidf` features was lowercased and stopwords, punctuation, numerals and word inflection (lemmatisation) was removed. The following hyperparameters were tuned:

| Parameters | Explored | Selected |
|------------------------|----------|----------|
| neg/pos | [1, 2, 4, 122] | 1 |
| C (L1) | [1e-5, 1e-2, 1, 100] | 1 |
| n_gram_range | [(1,1), (1,2)] | (1,2) |

The  `T-XGB(tfidf)` model is an XGBoost classifier. As pre-processing steps, the text to construct `tfidf` features was lowercased and stopwords, punctuation, numerals and word inflection (lemmatisation) was removed. Then, 1.500 `tfidf` features were selected based on a chi-squared test and used as input in the XGB classifier. The following hyperparameters were tuned:

| Parameters| Explored | Selected |
|------------------------|----------|----------|
| neg/pos | [1, 2, 4, 122] | 2 |
| eta | [1e-5, 1e-3, 1e-1] | 1e-1 |
| max_depth | [1, 3, 7] | 3 |
| subsample | [0.5, 1] | 0.5 |
| n_estimators | [10, 100, 1000] | 100 |

The  `T-LR(emb)` model is a logistic regression classifier with L2-regularisation using pre-trained document embeddings (from the `text-embedding-ada-002` encoder). No pre-processing steps were applied to the text before encoding. The following hyperparameters were tuned:

| Parameters | Explored | Selected |
|------------------------|----------|----------|
| neg/pos | [1, 2, 4, 122] | 4 |
| C (L2) | [1e-5, 1e-2, 1, 100] | 1 |

For documentation on these hyperparameters see: [`T-LR(tfidf)`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression), [`T-XGB(tfidf)`](https://xgboost.readthedocs.io/en/stable/parameter.html), [`T-LR(emb)`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression). `neg/pos` refers to the fraction of negative over positive samples in the training set obtained by randomly oversampling the minority class instances.

#### Early and late fusion models 

The `NT-XGB(num, tfidf)` model is an XGBoost classifier. The input consists of the concatenation of the numerical features and selected `tfidf` features. As pre-processing steps, the numerical features were normalised and the text to construct `tfidf` features was lowercased and stopwords, punctuation, numerals and word inflection (lemmatisation) was removed. Then, 1.500 `tfidf` features were selected based on a chi-squared test, concatenated with the numerical features and used as input in the XGB classifier. The following hyperparameters were tuned:

| Parameters| Explored | Selected |
|------------------------|----------|----------|
| neg/pos | [1, 2, 4, 122] | 2 |
| eta | [1e-5, 1e-3, 1e-1] | 1e-1 |
| max_depth | [1, 3, 7] | 3 |
| subsample | [0.5, 1] | 0.5 |
| n_estimators | [10, 100, 1000] | 100 |

The `NT-stack(XGB(num), XGB(tfidf))` model is a stacking classifier combining the predictions of the `XGB(num)` and the `XGB(tfidf)` models through a logistic regression classifier without regularisation. The predictions of the individual models were normalised. No hyperparameter were tuned.

For documentation on these hyperparameters see: [`NT-XGB(num, tfidf)`](https://xgboost.readthedocs.io/en/stable/parameter.html). `neg/pos` refers to the fraction of negative over positive samples in the training set obtained by randomly oversampling the minority class instances.
 
#### Sentence-attention model

The `NT-att(num, sent-emb)` is a custom PyTorch architecture (as described in the paper) trained with and Adam optimiser. The following hyperparameters were tuned:

| Parameters| Explored | Selected |
|------------------------|----------|----------|
| neg/pos | [1, 2, 4, 122] | 4 |
| weight-decay (L2) | [1e-5, 1e-3, 1e-1] | 1e-3 |
| lr | [1e-5, 1e-3, 1e-1] | 1e-3 |
| embedding_dimension | [32, 64] | 32 |

For documentation on these hyperparameters see: [`NT-att(num, sent-emb)`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html). `neg/pos` refers to the fraction of negative over positive samples in the training set obtained by randomly oversampling the minority class instances.
