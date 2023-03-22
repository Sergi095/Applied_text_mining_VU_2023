# Negation Cue Detection using SVM and XGBoost Classifiers (ATM VU 2023)


This repository contains code and data for an automatic negation cue classifier using the SEM 2012 SharedTask CD-SCO dataset. The code is written in Python, and the main algorithms used are thundersvm and xgboost. Both of these algorithms use GPU for their calculations, so it is necessary to have an NVIDIA GPU to run the code.


## Files in the Repository

* Applied_text_mining_conda_Env.yml: This file contains the environment used for this project.
* Tex project: The LaTeX project for the final report.
* Sections: folder for the sections.

Within the python_and_notebook folder:
* plots and results: folder with all plots, tables, saved models and metrics of the experiment.
* Data folder: folder with data files.
*  classifier_svm_vs_xgboost.ipynb: This is the main notebook that contains the experiments and results of the classifier.
* utils.py: This module contains helper functions for preprocessing the data.
* train.py: This module contains the training functions for the classifier.
* saved-5-sergio.zip: Individual annotations.
* sergio_jupyter.ipynb: Individual Assignment.
* customTfidf.py: custom class of vectorizer.


## How to Install ThunderSVM
Instructions for installing ThunderSVM can be found in the [official documentation.](https://thundersvm.readthedocs.io/en/latest/get-started.html)

## About the Project
This project was created for the Applied Text Mining course at Vrije Universiteit Amsterdam. The aim of the project is to build an automatic negation cue classifier using the SEM 2012 SharedTask CD-SCO dataset provided by the course lecturers.

## Credits
This project was created by [@Sergi095](https://github.com/Sergi095) for the Applied Text Mining course edition 2023 at Vrije Universiteit Amsterdam.




