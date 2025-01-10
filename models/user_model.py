#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from datetime import datetime
import base64
import requests
from scipy.stats import boxcox, stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder 
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, roc_auc_score, cohen_kappa_score, accuracy_score, adjusted_mutual_info_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, matthews_corrcoef, average_precision_score,f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingRegressor, VotingClassifier
from sklearn.tree import  DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from collections import Counter
from yellowbrick.classifier import ROCAUC
import optuna
from abc import ABC, abstractmethod
from typing import Tuple, Type
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
# DATA VISUALIZATION
# ------------------------------------------------------
# import skimpy
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIGURATIONS
# ------------------------------------------------------
warnings.filterwarnings('ignore')
pd.set_option('float_format', '{:.3f}'.format)


class Model(ABC):
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for compatibility with multi-class scoring."""
        return self.model.predict_proba(X)
    
    def fit(self, X, y):
        """Fit method for compatibility with sklearn's cross_val_score."""
        self.train(X, y)
        return self
    
    def score(self, X, y, scoring: str = 'accuracy'):

        if scoring == 'accuracy':
            # Use standard accuracy for classification.
            predictions = self.predict(X)
            return accuracy_score(y, predictions)
        elif scoring == 'roc_auc_ovo':
            # Use ROC AUC score for multi-class problems.
            probabilities = self.predict_proba(X)
            return roc_auc_score(y, probabilities, multi_class='ovo', average='macro')
        else:
            raise ValueError(f"Unsupported scoring method: {scoring}. Supported metrics are 'accuracy' and 'roc_auc_ovo'.")
            
class LGBModel(Model, BaseEstimator, ClassifierMixin):
    """XGBoost Classifier model with extended parameter support."""
    def __init__(self, **kwargs):
        self.model = LGBMClassifier(**kwargs)            
    
class XGBoostModel(Model, BaseEstimator, ClassifierMixin):
    """XGBoost Classifier model with extended parameter support."""
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
        
class CatBoostModel(Model, BaseEstimator, ClassifierMixin):
    """CatBoost Classifier model with extended parameter support."""
    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(**kwargs)
        
class RandomForestModel(Model, BaseEstimator, ClassifierMixin):
    """Random Forest Classifier model."""
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)   

class VotingModel(Model, BaseEstimator, ClassifierMixin):
    """Voting Classifier combining RFC_1 and XGB_2."""
    def __init__(self, estimators, voting='soft', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.model = VotingClassifier(estimators=self.estimators, voting=self.voting, weights=self.weights)     
        
class ModelFactory:
    """Factory to create model instances."""
    @staticmethod
    def get_model(model_name: str, **kwargs) -> Model:
        model_class = globals()[model_name]
        return model_class(**kwargs)

class Workflow_6:
    """Main workflow class for model training and evaluation."""

    def run_workflow(self, 
                     model_name: str, 
                     model_kwargs: dict, 
                     X: pd.DataFrame, 
                     y: pd.Series,
                     test_size: float, 
                     random_state: int,
                     scoring: str = 'accuracy'
                    ) -> None:
        """
        Main entry point to run the workflow:
        - Splits the data.
        - Trains the model.
        - Evaluates the model.
        - Save results to file.
        - Upload results to github repository.
        """
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)

        if model_name == 'VotingModel':
            model = VotingModel(**model_kwargs)
        else:
            model = ModelFactory.get_model(model_name, **model_kwargs)

        model.train(X_train, y_train)
        
 
        load_dotenv()
        github_token = os.getenv("GITHUB_TOKEN")
        print(f"Loaded token: {github_token}")
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN is not set. Please configure it in your environment.")            
        github_repo_url = "https://github.com/AdamSmulczyk/014_Poisonous_Mushrooms/tree/performance_reports"
        
        
        # evaluate_model
        results = self.evaluate_model(model, X_train, X_test, y_train, y_test, scoring)  
        print("Model Evaluation Results:")
        print(results.to_string())
        self.save_results_to_file(results,
                                  save_dir=r"C:\Users\adams\OneDrive\Dokumenty\Python",
                                  prefix="mushroom_evaluate",
                                  file_type="csv",
                                  github_token=github_token, 
                                  github_repo_url=github_repo_url
                                 )
        
        # evaluate_plots
        plot = self.evaluate_plots(model, X_test, y_test, model_name)
        self.save_results_to_file(plot, 
                                  save_dir=r"C:\Users\adams\OneDrive\Dokumenty\Python", 
                                  prefix="mushroom_plot", 
                                  file_type="png",
                                  github_token=github_token, 
                                  github_repo_url=github_repo_url
                                 )       

       
    def split_data(self, 
                   X: pd.DataFrame, 
                   y: pd.Series, 
                   test_size: float, 
                   random_state: int
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    
    def evaluate_model(self, 
                       model: Model, 
                       X_train: pd.DataFrame, 
                       X_test: pd.DataFrame, 
                       y_train: pd.Series, 
                       y_test: pd.Series,
                       scoring: str) -> pd.DataFrame:

        """
        Evaluate the model using custom metrics.
        
        Parameters:
        - model: Trained model to evaluate.
        - X_train, X_test: Feature datasets for training and testing.
        - y_train, y_test: Target datasets for training and testing.
        
        Returns:
        - pd.DataFrame: DataFrame containing evaluation metrics for train and test sets.
        """
        def compute_metrics(y_true: pd.Series, y_pred_proba: np.ndarray) -> pd.Series:
            """Helper function to calculate metrics."""
            cutoff = np.sort(y_pred_proba)[-y_true.sum():].min()
            y_pred_class = np.array([1 if x >= cutoff else 0 for x in y_pred_proba])
        
            """Evaluate the model and return evaluation metrics as a dictionary."""
            predictions = model.predict(X_test)

            """Evaluate the model using Stratified K-Fold cross-validation."""
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            n_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=scoring)   
        
            return pd.Series({
                'F1_score': round(f1_score(y_true, y_pred_class), 4),
                'P-R_score': round(average_precision_score(y_true, y_pred_class), 4),
                'Matthews': round(matthews_corrcoef(y_true, y_pred_class), 4), 
                'Accuracy': round(accuracy_score(y_true, y_pred_class), 4),
                'Recall': round(recall_score(y_true, y_pred_class), 4),
                'Precision': round(precision_score(y_true, y_pred_class), 4), 
                'SKF': round(np.mean(n_scores), 4),                       
                'AUC': round(roc_auc_score(y_true, y_pred_class), 4), 
                'Min_cutoff': round(cutoff, 4),
            })
        
        train_metrics = compute_metrics(y_train, model.predict_proba(X_train)[:, 1])
        test_metrics = compute_metrics(y_test, model.predict_proba(X_test)[:, 1])
        
        return pd.DataFrame({'TRAIN': train_metrics, 'TEST': test_metrics}).T

        
    def evaluate_plots(self, 
                       model: Model,  
                       X_test: pd.DataFrame,  
                       y_test: pd.Series,
                       model_name: str,
                       save_dir: str = r"C:\Users\adams\OneDrive\Dokumenty\Python"):
        
        predictions = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if len(model.predict_proba(X_test).shape) > 1 else model.predict_proba(X_test)
#         cutoff = np.sort(y_pred_proba)[-y_test.sum():].min()
        cutoff=0.2
        y_pred_class = np.array([1 if x >= cutoff else 0 for x in y_pred_proba])
    
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        # Plot ROC curve
        roc_auc_test_cat = roc_auc_score(y_test, y_pred_proba)
        fpr_cat, tpr_cat, _ = roc_curve(y_test, y_pred_proba)
        axes[0].plot(fpr_cat, tpr_cat, label=f'ROC AUC = {roc_auc_test_cat:.4f}')
        axes[0].plot([0, 1], [0, 1], 'k--')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'Confusion Matrix - threshold = {cutoff:.3f}')
        axes[0].legend(loc='best')

        # Confusion matrix
        cm_cat = confusion_matrix(y_test, predictions)


        # Plot confusion matrix
        sns.heatmap(cm_cat, annot=True, fmt='.0f', cmap='viridis', cbar=False, ax=axes[1])
        axes[1].set_xlabel('Predicted labels')
        axes[1].set_ylabel('True labels')
        axes[1].set_title(f'Confusion Matrix {model_name}')

        plt.tight_layout()       
#         plt.show()
        
        return fig
       
    
    @staticmethod
    def save_results_to_file(data, 
                             save_dir: str, 
                             prefix: str, 
                             file_type: str = "csv",
                             github_token: str = None, 
                             github_repo_url: str = None) -> None:
        """
        Save data (results or plots) to a file.

        Parameters:
        - data: The data to save (pd.DataFrame for CSV or matplotlib figure for plots).
        - save_dir: Directory to save the file.
        - prefix: File prefix for naming the output file.
        - file_type: Type of file to save ("csv" for results, "png" for plots).
        """
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if file_type == "csv":
            file_name = f"{prefix}_{timestamp}.csv"
            output_path = os.path.join(save_dir, file_name)
            print(f"Saving results to: {output_path}")
            data.to_csv(output_path, index=True)
        elif file_type == "png":
            file_name = f"{prefix}_{timestamp}.png"
            output_path = os.path.join(save_dir, file_name)
            print(f"Saving plot to: {output_path}")
            data.savefig(output_path, bbox_inches='tight')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
         
        if github_token and github_repo_url:
            Workflow_6.upload_to_github(output_path, github_repo_url, github_token)
         
    
    @staticmethod
    def upload_to_github(output_path: str, 
                         github_repo_url: str, 
                         github_token: str, 
                         branch: str = "main",
                         commit_message: str = "Add file") -> None:
        """
        Uploads a file to a GitHub repository.

        Parameters:
        - output_path (str): Path to the file on the local disk.
        - github_repo_url (str): URL of the GitHub repository (e.g., 'https://github.com/User/RepoName/tree/main/folder').
        - github_token (str): GitHub personal access token for authentication.
        - branch (str): Branch to which the file should be uploaded. Default is 'main'.
        - commit_message (str): Commit message for the upload. Default is 'Add file'.
        """
        # Construct the API URL for the target file
        url_parts = github_repo_url.rstrip("/").split("/")
        repo_name = url_parts[4]  # Repository name
        user = url_parts[3]       # GitHub username
        folder_path = "/".join(url_parts[6:])  # Folder path inside the repository       
        file_name = os.path.basename(output_path)
        api_url = f"https://api.github.com/repos/{user}/{repo_name}/contents/{folder_path}/{file_name}"

        # Read the file content
        with open(output_path, "rb") as file:
            content = base64.b64encode(file.read()).decode("utf-8")

        # Prepare the payload for the request
        payload = {
            "message": commit_message,
            "branch": branch,
            "content": content,
        }

        # Prepare the headers
        headers = {"Authorization": f"Bearer {github_token}"}

        # Send the PUT request to GitHub API
        response = requests.put(api_url, json=payload, headers=headers)

        # Handle response
        if response.status_code == 201:
            print(f"File '{output_path}' successfully uploaded to GitHub at '{github_repo_url}'.")
        elif response.status_code == 404:
            print(f"Failed to upload file: {response.status_code} - Repository or path not found.")
        elif response.status_code == 422:
            print(f"File '{output_path}' already exists in the repository.")
        else:
            print(f"Failed to upload file: {response.status_code} - {response.text}")
