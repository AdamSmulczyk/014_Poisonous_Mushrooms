#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from models import *
from data_processor import preprocess_data
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler

def main():
    print('-' * 80)
    print('train')
    
    # Preprocessing danych
    train_file = 'train.csv'
    train = preprocess_data(train_file)
    train = train.sample(50000)

    X = train.drop(columns=['id', 'class'])
    y = train['class']

    # Encoding celu
    label = LabelEncoder()
    y = label.fit_transform(y)
    y = y.astype('int16')

    # Scaling danych
    categorical_columns = X.select_dtypes(include=['category', 'object']).columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_columns] = encoder.fit_transform(X[categorical_columns].astype(str))

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Parametry modeli
    SEED = 42
    XGB_Params = {'max_depth': 13, 
                  'min_child_weight': 5, 
                  'learning_rate': 0.02,
                  'colsample_bytree': 0.6, 
                  'max_bin': 3000, 
                  'n_estimators': 1500,
                  'random_state': SEED}

    # Modele
    print("Initializing workflow...")
    RFC_1 = RandomForestModel(n_estimators=50, random_state=42)
    XGB_2 = XGBoostModel(**XGB_Params)

    voting_estimators = [('RandomForest', RFC_1.model), ('XGBoost', XGB_2.model)]
    workflow_voting = Workflow_6()
    workflow_voting.run_workflow(
        model_name='VotingModel',
        model_kwargs={'estimators': voting_estimators, 'voting': 'soft', 'weights': [1.0, 2.0]},
        X=X,
        y=y,
        test_size=0.2,
        random_state=42,
        scoring='accuracy'
    )

if __name__ == "__main__":
    main()

