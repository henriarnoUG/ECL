# The ECL benchmark dataset

This repository can be used to:
1. Reconstruct the ECL (Edgar-CompuStat-Lopucki) benchmark dataset from (Arno et al., 2023).
2. Run the baseline models for business failure prediction from (Arno et al., 2022).

##### Reconstruct ECL

The ECL benchmark dataset combines three existing data sources: the Edgar corpus, Compustat and the Lopucki bankruptcy research database. Due to the paid access required for Compustat, we are unable to share the complete benchmark dataset. However, in this repository, we provide the necessary code that allows you to reconstruct the dataset. 

In the ECL csv-file, which is accesible [here](https://cloud.ilabt.imec.be/index.php/s/TFGZgF3EyS4jsz2), each row corresponds to a 10K filing. Each 10K filing can be matched (1) with a document from the Edgar corpus through the ```filename``` variable and (2) with an entry from Compustat through the ```gvkey``` and ```datadate``` variables. The Edgar corpus can be accessed [here](https://cloud.ilabt.imec.be/index.php/s/yrQneXG74bNYxLp) and Compustat requires paid access from [here](https://wrds-www.wharton.upenn.edu/). Note that the Lopucki BRD is no longer updated as of december 2022 (see [this link](https://lopucki.law.ufl.edu/index.php) for more information). More details on how to reconstruct the ECL benchmark dataset can be found in the ```/ecl.ipynb``` notebook.

##### Baseline models for business failure prediction

The code to run the baseline models for the next-year business failure prediction task can be found in ```/baselines/``` folder.

##### References
[Next-Year Bankruptcy Prediction from Textual Data: Benchmark and Baselines](https://scholar.google.be/citations?user=ce8BmFgAAAAJ&hl=nl) (Arno et al., 2022)

[From Numbers to Words: Multi-Modal Bankruptcy Prediction Using the ECL Dataset](https://scholar.google.be/citations?user=ce8BmFgAAAAJ&hl=nl) (Arno et al., 2023)