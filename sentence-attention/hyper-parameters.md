# Hyperparameter optimisation procedure

#### Overview
In this document, we describe the hyperparameter optimisation procedure from (Arno et al., 2024) applied to all next-year business failure prediction models. The 10K filings from the ECL benchmark dataset are split into training, validation, and test sets using a temporal split (i.e., based on the filing year). For each model, hyperparameters are selected to maximize the ROC-AUC on the validation set. After tuning, the models are retrained on the combined training and validation sets and then evaluated on the test set. Below, we provide tables detailing the pre-processing steps, the explored hyperparameters, and the optimization strategy for each model.

#### Numerical models 

The `N-Z'(num)` model is a logistic regression classifier where no hyperparameters were tuned. As pre-processing step, the numerical features were mean-imputed.

| Parameters | Explored | Selected |
|------------------------|----------|----------|
| None | - | - |

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
| eta | (1e-5, 1e-3, 1e-1) | 1e-1 |
| max_depth | (1, 3, 7) | 1 |
| subsample | (0.5, 1) | 0.5 |
| n_estimators | (10, 100, 1000) | 1000 |

For documentation on these hyperparameters see: [`N-LR(num)`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression), [`N-MLP(num)`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), [`N-XGB(num)`](https://xgboost.readthedocs.io/en/stable/parameter.html)



#### Textual models 

#### Early-fusion models 

#### Late-fusion models 

#### Sentence-attention model
