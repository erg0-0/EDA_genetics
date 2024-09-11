# Project group 8 genetics

Read further in order to configure and use the data from this project.

## How to create environment for the first time?
Use one-liner code in your Terminal:

Mac: 

conda create -n group8 python=3.11.0 numpy=1.23.5 pandas=2.1.4 matplotlib=3.8.0 plotly=5.9.0 python-slugify=5.0.2  blas=1.0=openblas scikit-learn=1.3.0 scipy=1.11.4 seaborn=0.12.2 sqlite=3.41.2 streamlit=1.30.0 statsmodels=0.14.0 altair=5.0.1 joblib=1.2.0 imbalanced-learn=0.11.0 xgboost=1.7.3

windows:

conda create -n group8 python=3.11.0 numpy=1.23.5 pandas=2.1.4 matplotlib=3.8.0 plotly=5.9.0 python-slugify=5.0.2  blas=1.0=mkl scikit-learn=1.3.0 scipy=1.11.4 seaborn=0.12.2 sqlite=3.41.2 streamlit=1.30.0 statsmodels=0.14.0 altair=5.0.1 joblib=1.2.0 imbalanced-learn=0.11.0 xgboost=1.7.3

## How to ACTIVATE this environment?
conda activate group8

## How to REMOVE this enviromnent? 
conda env remove --name group8

## How to UPDATE one library only to specified version?
Go to https://anaconda.org and search the package name. Use *conda install* for the installation. If no conflict, adjust the one-liner in readme.md above and commit.

## How to DOWNGRADE a library  to specified version?

conda install --name group8 python=3.11.0

## Git repository

git clone https://gitlab.com/Mr13Nice/group_8.git 

## How to import the data?

Go to the folder where the repository group_8 exist (it is the folder one level higher than group_8). 
Create a folder **local_files** . The directory looks like this:
- C:/something_something/wherever_your_group_project_is/group_8/
- C:/something_something/wherever_your_group_project_is/local_files/

Paste 3 folders with  raw data to the local_files folder, so it looks like this:
- C:/something_something/wherever_your_group_project_is/local_files/BRAKI ZĘBOWE_wyniki
- C:/something_something/wherever_your_group_project_is/local_files/KONTROLE_wyniki
- C:/something_something/wherever_your_group_project_is/local_files/ROZSZCZEPY_wyniki

Make sure you paste the whole folder. Don't change the folder names. 
Code is tested for Mac and Windows.

## Dataset description

Dataset contains data about genomic mutations of 444 genes with 19004 variants for 272 people. The exact meaning of each column is described in the Wiki section linked below.

Data includes 3 group of patiens:
1.  control group ("kontrole")
2. patients with cleft palate disease ("rozsczepy") 
3.  patients with hypodontia,  patients born with missing teeth ("zęby"). 

The tree groups are often refered in code in variables names as 'control', 'cleft' and 'teeth'.

[ ] [Wiki - Dataset description] (https://gitlab.com/Mr13Nice/group_8/-/wikis/Dataset-description )

## Targets

Each disease is treated separately, therefore there are 2 models with binary classification.

## Exploratory data analysis

In the notebook and in the src/visualisation are stored results of the statistical testing and visualisation of heatmap. 
Some insights are collected in the report diary but must importantly in the presentation. Insight about the most impactful genes are stored as variable and used directly in the models.

## Project structure

- notebooks - notebook main_project.ipynb contains:
    - data load
    - preparation and performing statistical tests
    - visualisations for statistical testing
    - implementing model functions that print model results

- reports - notes

- scr:
    - data: 
    main_project.py contains functions needed to generate the models. 
    There are 2 models per each disease.

    - models: storage for functions related to machine learning models

- utils:
    storage of functions for exploratory data analysis

- visualisation:
    images related to the EDA and model performance

## Various results

Choosing Mac over Windows may impact the result. In other places random seed was used for reproducibility.

## Authors and acknowledgment
Without support of dr Agnieszka Thomson this project would not be possible.
We would like to express our sincere gratitude to Prof. dr hab. Adrianna Mostowska for providing accessibility to the genetic data from the research conducted by her.
