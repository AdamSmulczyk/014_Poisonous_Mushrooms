#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
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
        """
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)

        if model_name == 'VotingModel':
            model = VotingModel(**model_kwargs)
        else:
            model = ModelFactory.get_model(model_name, **model_kwargs)

        model.train(X_train, y_train)
        
        results = self.evaluate_model(model, X_train, X_test, y_train, y_test, scoring)
        
        print("Model Evaluation Results:")
        print(results.to_string())     
        
        plot = self.evaluate_plots(model, X_test, y_test, model_name)

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
                'P-R_score': round(average_precision_score(y_true, y_pred_class), 4), #Precision-Recall
                'Matthews': round(matthews_corrcoef(y_true, y_pred_class), 4),  #the Matthews correlation coefficient (MCC)
                'Accuracy': round(accuracy_score(y_true, y_pred_class), 4),
                'Recall': round(recall_score(y_true, y_pred_class), 4),
                'Precision': round(precision_score(y_true, y_pred_class), 4), 
                'SKF': np.mean(n_scores),                                        #Stratified K-Fold
                'AUC': roc_auc_score(y_true, y_pred_class),                      #the Area Under the Curve
                'Min_cutoff': cutoff,
            })
        
        train_metrics = compute_metrics(y_train, model.predict_proba(X_train)[:, 1])
        test_metrics = compute_metrics(y_test, model.predict_proba(X_test)[:, 1])
        
        return pd.DataFrame({'TRAIN': train_metrics, 'TEST': test_metrics}).T

    def evaluate_plots(self, 
                       model: Model,  
                       X_test: pd.DataFrame,  
                       y_test: pd.Series,
                       model_name: str):
        
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
        plt.show()
        
