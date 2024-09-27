# Hyperparameter optimisation procedure

#### Overview
In this document, we describe the hyperparameter optimisation procedure from (Arno et al., 2024) applied to all next-year business failure prediction models. The 10K filings from the ECL benchmark dataset are split into training, validation, and test sets using a temporal split (i.e., based on the filing year). For each model, hyperparameters are selected to maximize the ROC-AUC on the validation set. After tuning, the models are retrained on the combined training and validation sets and then evaluated on the test set. Below, we provide tables detailing the pre-processing steps, the explored hyperparameters, and the optimization strategy for each model.



#### Numerical models


| Model       | Pre-processing Steps                   | Explored Hyperparameters                               | Optimizer  |
|-------------|----------------------------------------|--------------------------------------------------------|------------|
| `N-Z'(num)` |  mean imputation (`num`)                 |  -                                                  |  -      |
| `N-LR(num)` |  mean imputation (`num`)                 |  neg/pos: [1, **2**, 4, 122]       | adam       |
|             |  normalization (`num`)                   |  c (l2): [1e-5, **1e-2**, 1, 100]                      |            |
| `N-MLP(num)`|  mean imputation (`num`)                 |  neg/pos: [1, **2**, 4, 122]        | adam       |
|             |  normalization (`num`)                   |  c (l2): [1e-5, 1e-2, **1**, 100]                      |            |
|             |                                        |  initial learning rate: [1e-5, **1e-3**, 1e-1]         |            |
|             |                                        |  hidden layers: [1, **2**]                             |            |
|             |                                        |  hidden neurons: [50, **100**]                         |            |
| `N-XGB(num)`|  normalization (`num`)                   |  neg/pos: [1, **2**, 4, 122]       | xgboost    |
|             |                                        |  eta (stepsize): [1e-5, 1e-3, **1e-1**]                |            |
|             |                                        |  n_estimators: [10, 100, **1000**]                     |            |
|             |                                        |  subsample: [**0.5**, 1]                               |            |
|             |                                        |  max depth: [**1**, 3, 7]                              |            |


#### References

[The Numbers and Narrative of Bankruptcy: Interpretable Multi-Modal Business Failure Prediction](https://scholar.google.be/citations?user=ce8BmFgAAAAJ&hl=nl) (Arno et al., 2024) [forthcoming]
