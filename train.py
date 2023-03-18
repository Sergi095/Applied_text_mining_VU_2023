from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
# import sklearn_crfsuite
# from sklearn_crfsuite import metrics
from sklearn.decomposition import TruncatedSVD
import thundersvm
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple, Optional
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from customTifidf import *

def defining_gridsearch(type_model: str,
                        features: list) -> GridSearchCV:

    num = StandardScaler()      
    vectorizer1 = TfidfVectorizer()#.fit(corpus_df.text.to_list())

    numeric_transformer  = Pipeline(
    steps=[("scaler", StandardScaler())])
    str_transformer = Pipeline(
    steps=[
        ("tfidf", custom_tfidf(vectorizer1))
    ]) 

    features_numeric_control = ['root_path',
                                'neg_exp_list',
                                'affix_cue']
    feature_str_control =      [
                                'token',
                                'prev_lemma',
                                'lemma',
                                'tag',
                                'dependency',
                                'head']


    
    feature_str = [feat for feat in features if feat in feature_str_control]
    features_numeric = [feat for feat in features if feat in features_numeric_control]



    preprocessor = ColumnTransformer(
        transformers=[('tfidf', str_transformer, feature_str),
                      ('num',numeric_transformer,features_numeric)  
                      ],
                      remainder="passthrough"
                      )

    if type_model == 'svm':
        pipe_svm = Pipeline([
                        ('preprocessor', preprocessor),
                        ('sampler', RandomOverSampler(sampling_strategy='minority')),
                        ('undersampler', RandomUnderSampler(sampling_strategy='majority')),
                        ('classifier', thundersvm.SVC())])
        
        # SVM
        param_grid_svm = {
            'classifier__C':      [1.0, 10.0],
            'classifier__kernel': ['linear', 'polynomial'],
            'classifier__gamma':  [0.5, 1],
        }

        grid_svm     = GridSearchCV(pipe_svm, param_grid_svm, n_jobs=-1, cv=5)
        return grid_svm

    elif type_model == 'xgb':
        pipe_xgb = Pipeline([
                        ('preprocessor', preprocessor),
                        ('sampler', RandomOverSampler(sampling_strategy='minority')),
                        ('undersampler', RandomUnderSampler(sampling_strategy='majority')),
                        ('classifier', xgb.XGBClassifier(tree_method='gpu_hist'))])
        
        # XGB
        param_grid_xgb = {
            'classifier__max_depth':        [3, 5],
            'classifier__colsample_bytree': [0.6, 1.0],
            'classifier__learning_rate':    [0.1, 0.3],
        }

        grid_xgb     = GridSearchCV(pipe_xgb, param_grid_xgb, n_jobs=-1, cv=5)
        return grid_xgb
    else:
        raise ValueError('Please specify a valid model [svm,xgb]')
        


def train_model(grid: GridSearchCV,
                X_train: pd.DataFrame,
                y_train: pd.DataFrame) -> Tuple[GridSearchCV, dict]:
    # set seed
    np.random.seed(57)

    start_time = time.time()
    grid.fit(X_train,y_train)
    print('BEST PARAMS FOUND BY GRID SEARCH: ',grid.best_params_)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken: {:.2f} minutes".format(time_taken/ 60))
    return grid, grid.best_params_



def ablation_study(type_model: str,
                   X: pd.DataFrame, # X_train
                   y: pd.DataFrame) -> dict: #y_train
    """
    Train a model without random selected features to perform an ablation study
    
    :param type_model: str, the type of model to train. Either 'svm' or 'xgb'.
    :param X: pd.DataFrame, the training data.
    :param y: pd.DataFrame, the target variable.
    :return: dict, a dictionary of the evaluation metrics for each ablation study iteration.
    dict, a dictionary with the best params of each iteration.
    """
    # Set the seed for reproducibility
    np.random.seed(73) #sheldon's prime number
    
    # List of features to ablate
    all_features = [
                    'token',
                    'lemma',
                    'prev_lemma',
                    'tag',
                    'dependency',
                    'head',
                    'root_path',
                    'neg_exp_list',
                    'affix_cue']
    # Splitting into train val datasets
    X_train, X_val, y_train, y_val = train_test_split(X, y,random_state=73, test_size=(1/3), stratify=y)
    # Dictionary to store evaluation metrics for each ablation study iteration
    metrics_dict = {}
    best_params_dict = {}
    models = {}
    # Perform the ablation study
    # main for loop
    run = 1
    selected_features = set()  # keep track of selected features to avoid repetition
    for i in range(len(all_features)):
        print(f'run: {run}')
        # Remove one feature at random that has not been selected before
        remaining_features = list(set(all_features) - selected_features)
        feature_to_remove = np.random.choice(remaining_features)
        selected_features.add(feature_to_remove)
        
        # Train the model without the removed feature
        grid = defining_gridsearch(type_model, remaining_features)
        grid, best_params = train_model(grid, X_train[remaining_features], y_train)
        
        # Evaluate the model without the removed feature on val set
        y_pred = grid.predict(X_val[remaining_features])
        metrics = classification_report(y_val, y_pred, output_dict=True)
        metrics_dict[type_model+f'_{i+1}: '+str(remaining_features)] = metrics
        best_params_dict[f'{type_model}_model_{i+1}'] = best_params
        models[type_model+f'_{i+1}'] = grid

        run +=1
    
    return metrics_dict, best_params_dict, models

def save_metrics(metrics: dict, filename: str) -> None:
  pickle.dump(metrics, open(filename, 'wb'))

def plots(metrics: dict,
          main_title: str = None,
          file_name: str = None)-> None:

  # Define the classes
  classes = ['0', '1', '2']

  # Define the metrics to plot
  metrics_to_plot = ['precision', 'recall', 'f1-score']

  # Initialize the subplots
  fig, axs = plt.subplots(1, len(metrics_to_plot), figsize=(20, 5))

  # Set the title of each subplot
  axs[0].set_title('Precision')
  axs[1].set_title('Recall')
  axs[2].set_title('F1-Score')
  # Loop over the metrics to plot
  for i, metric in enumerate(metrics_to_plot):
      # Loop over the classes
      for j, cls in enumerate(classes):
          # Extract the scores for the current class and metric
          scores = [metrics[model][cls][metric] for model in metrics.keys()]
          # Plot the scores as a line
          axs[i].plot(scores, label=f'Class {cls}')
      # Add a legend to the plot
      axs[i].legend()
      # Set the x-axis label
      axs[i].set_xlabel('Model')
      # Set the y-axis label
      axs[i].set_ylabel(metric.capitalize())
      # Set the x-ticks and labels
      axs[i].set_xticks(range(len(metrics.keys())))
      axs[i].set_xticklabels(metrics.keys(), rotation=90)
      # Set the y-axis label
      axs[i].set_ylabel(metric.capitalize())
  if main_title  is not None:
    fig.suptitle(main_title) # main title
  # Display the plot
  if file_name is not None:
    plt.savefig(file_name)
  plt.show()

def get_best_model(metrics: dict) -> str:
    best_model = None
    best_f1_score = 0
    
    for key, values in metrics.items():
        if key.startswith('xgb_') or key.startswith('svm_'):
            f1_score = values['weighted avg']['f1-score']
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model = key
            print("Considering model", key, f' with score: {f1_score}')
    return best_model



def plot_classification_reports(X_dev: pd.DataFrame,
                                y_dev: pd.DataFrame,
                                X_test: pd.DataFrame,
                                y_test: pd.DataFrame,
                                X_test2: pd.DataFrame,
                                y_test2: pd.DataFrame,
                                model: GridSearchCV,
                                title: str = None,
                                file_name: str = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    # fig.tight_layout(pad=4.0)
    
    # Plot confusion matrix and classification report for dev set
    y_dev_pred = model.predict(X_dev)
    axes[0].set_title("Dev Set")
    sns.heatmap(confusion_matrix(y_dev, y_dev_pred), annot=True, fmt='d', ax=axes[0])
    
    # Plot confusion matrix and classification report for test set
    y_test_pred = model.predict(X_test)
    axes[1].set_title("Test Set Cardboard")
    sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', ax=axes[1])
    
    # Plot confusion matrix and classification report for test2 set
    y_test2_pred = model.predict(X_test2)
    axes[2].set_title("Test Set Circle")
    sns.heatmap(confusion_matrix(y_test2, y_test2_pred), annot=True, fmt='d', ax=axes[2])

    if file_name is not None:
      plt.savefig(file_name)

    if title is not None:
      fig.suptitle(title) # main title
    plt.show()


def get_classification_reports(X_dev: pd.DataFrame,
                                y_dev: pd.DataFrame,
                                X_test: pd.DataFrame,
                                y_test: pd.DataFrame,
                                X_test2: pd.DataFrame,
                                y_test2: pd.DataFrame,
                                model: GridSearchCV) -> pd.DataFrame:
    df = pd.DataFrame(columns=['Precision', 'Recall', 'F1-score', 'Support'],
                      index=['Test',
                             'Test 0',
                             'Test 1',
                             'Test 2',
                             'Test macro avg',
                             'Test weighted avg',

                             'Test1',
                             'Test1 0',
                             'Test1 1',
                             'Test1 2',
                             'Test1 macro avg',
                             'Test1 weighted avg',

                             'Dev',
                             'Dev 0',
                             'Dev 1',
                             'Dev 2',
                             'Dev macro avg',
                             'Dev weighted avg'])
    
    # Get classification report for dev set
    y_dev_pred = model.predict(X_dev)
    cr_dev = classification_report(y_dev, y_dev_pred, output_dict=True)
    df.loc['Dev'] = " ", " ", " ", " "
    df.loc['Dev 0'] = [cr_dev['0']['precision'], cr_dev['0']['recall'], cr_dev['0']['f1-score'], cr_dev['0']['support']]
    df.loc['Dev 1'] = [cr_dev['1']['precision'], cr_dev['1']['recall'], cr_dev['0']['f1-score'], cr_dev['1']['support']]
    df.loc['Dev 2'] = [cr_dev['2']['precision'], cr_dev['2']['recall'], cr_dev['2']['f1-score'], cr_dev['2']['support']]
    df.loc['Dev macro avg'] = [cr_dev['macro avg']['precision'], cr_dev['macro avg']['recall'], cr_dev['macro avg']['f1-score'], cr_dev['macro avg']['support']]
    df.loc['Dev weighted avg'] = [cr_dev['weighted avg']['precision'], cr_dev['weighted avg']['recall'], cr_dev['weighted avg']['f1-score'], cr_dev['weighted avg']['support']]

    # Get classification report for test set
    y_test_pred = model.predict(X_test)
    cr_test = classification_report(y_test, y_test_pred, output_dict=True)
    df.loc['Test'] = " ", " ", " ", " "
    df.loc['Test 0'] = [cr_test['0']['precision'], cr_test['0']['recall'], cr_test['0']['f1-score'], cr_test['0']['support']]
    df.loc['Test 1'] = [cr_test['1']['precision'], cr_test['1']['recall'], cr_test['1']['f1-score'], cr_test['1']['support']]
    df.loc['Test 2'] = [cr_test['2']['precision'], cr_test['2']['recall'], cr_test['2']['f1-score'], cr_test['2']['support']]
    df.loc['Test macro avg'] = [cr_test['macro avg']['precision'], cr_test['macro avg']['recall'], cr_test['macro avg']['f1-score'], cr_test['macro avg']['support']]
    df.loc['Test weighted avg'] = [cr_test['weighted avg']['precision'], cr_test['weighted avg']['recall'], cr_test['weighted avg']['f1-score'], cr_test['weighted avg']['support']]
    
    # Get classification report for test2 set
    y_test2_pred = model.predict(X_test2)
    cr_test2 = classification_report(y_test2, y_test2_pred, output_dict=True)
    df.loc['Test1'] = " ", " ", " ", " "
    df.loc['Test1 0'] = [cr_test2['0']['precision'], cr_test2['0']['recall'], cr_test2['0']['f1-score'], cr_test2['0']['support']]
    df.loc['Test1 1'] = [cr_test2['1']['precision'], cr_test2['1']['recall'], cr_test2['1']['f1-score'], cr_test2['1']['support']]
    df.loc['Test1 2'] = [cr_test2['2']['precision'], cr_test2['2']['recall'], cr_test2['2']['f1-score'], cr_test2['2']['support']]
    df.loc['Test1 macro avg'] = [cr_test2['macro avg']['precision'], cr_test2['macro avg']['recall'], cr_test2['macro avg']['f1-score'], cr_test2['macro avg']['support']]
    df.loc['Test1 weighted avg'] = [cr_test2['weighted avg']['precision'], cr_test2['weighted avg']['recall'], cr_test2['weighted avg']['f1-score'], cr_test2['weighted avg']['support']]
    
    return df