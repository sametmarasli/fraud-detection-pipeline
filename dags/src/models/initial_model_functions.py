import pandas as pd
import os
import pickle
import logging
import joblib 
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import  GridSearchCV, train_test_split
from xgboost.sklearn import XGBClassifier
import src.models.pipeline_config as pipeline_config
from src.models.transformers import (KeepFeatures, CustomLabelEncoder, AmountVsOldAndNewBalanceOrig, AmountVsOldAndNewBalanceDest)
import mlflow.sklearn
from src.utils import init_logger

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_preprocess(**kwargs):
    logging.info('Start loading data')

    df = pd.read_csv(os.getcwd() + kwargs['path_data']).query("type=='TRANSFER' | type=='CASH_OUT' ")
    df = df.reset_index(drop=True).reset_index()
    df = df.rename(columns={
        'index':'id',
        'oldbalanceOrg':'oldBalanceOrig',
        'newbalanceOrig':'newBalanceOrig',
        'oldbalanceDest':'oldBalanceDest',
        'newbalanceDest':'newBalanceDest'
        })  

    X, y = df.drop('isFraud', axis=1), df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle=True, test_size=.3,stratify=y )
    X_train, X_stream, y_train, y_stream = train_test_split(X_train,y_train, shuffle=True, test_size=.97,stratify=y_train )

    train_set = [X_train, y_train]
    test_set = [X_test, y_test]
    stream_sample = [X_stream, y_stream]

    pickle.dump(train_set, open(os.getcwd() + kwargs['path_train_set'], "wb"))
    pickle.dump(test_set, open(os.getcwd() + kwargs['path_test_set'], "wb"))
    pickle.dump(stream_sample, open(os.getcwd() + kwargs['path_stream_sample'], "wb"))
    logging.info(f'Dimensions train: {y_train.shape}, test:{y_test.shape}, strea:{y_stream.shape}')
    logging.info(f'{sum(y_train)} frauds in the train set.')
    logging.info(f'Data is loaded successfully')
    
    return X_train, y_train, X_test, y_test

def construct_model():

    model_pipeline = Pipeline([
        ('keep_features', KeepFeatures(pipeline_config.FEATURES)),
        ('label_encode', CustomLabelEncoder(pipeline_config.LABEL_ENCODE_FEATURES)),
        # ('amount_vs_old_new_balance_orig', AmountVsOldAndNewBalanceOrig()),
        # ('amount_vs_old_new_balance_dest', AmountVsOldAndNewBalanceDest()),
        ('xgb', XGBClassifier())
        ])

    return model_pipeline

def fit_model(**kwargs):

	# fit model along preprocessed data and constructed model framework
    ti = kwargs['ti']
    loaded = ti.xcom_pull(task_ids='load_preprocess')
    
    # logging.info('variables successfully fetched from previous task')
    X_train = loaded[0]
    y_train = loaded[1]
    X_test = loaded[2]
    y_test = loaded[3]
    
    # initialize the model
    model_pipeline = construct_model()
    mlflow.set_tracking_uri('http://mlflow:5000')
    with mlflow.start_run():
        # hyperparameter tuning
        search = GridSearchCV(model_pipeline, param_grid=pipeline_config.MODEL_PARAMETER_GRID, n_jobs=-1, scoring=pipeline_config.EVAL_METRIC, cv=3, verbose=0)
        X_sample, X_, y_sample, y_ = train_test_split(X_train, y_train, train_size=.2, stratify=y_train)
        logging.info(f"population distribution of target :{y_train.mean()}, sample :{y_sample.mean()},  ")
        search.fit(X_sample, y_sample)

        logging.info("Best parameter (CV score=%0.3f):" % search.best_score_)
        logging.info(search.best_params_)

        # retrain the selected model
        logging.info('Fit the best estimator with all the data')
        model = search.best_estimator_.fit(X_train, y_train)
        joblib.dump(model, os.getcwd() + kwargs['initial_model_path'])
        logging.info('Finished training the model.')

        THRESHOLD = .5
        probabilities = model.predict_proba(X_test)
        f1_score_original = f1_score(y_test, probabilities[:, 1]>THRESHOLD)
        logging.info(f'F1 score of the initial model {f1_score_original}.')
        mlflow.log_metric('F1 score', f1_score_original)

        precision_score_original = precision_score(y_test, probabilities[:, 1]>THRESHOLD)
        logging.info(f'Precision score of the initial model {precision_score_original}.')
        mlflow.log_metric('Precision score', precision_score_original)

        recall_score_original = recall_score(y_test, probabilities[:, 1]>THRESHOLD)
        logging.info(f'Recall score of the initial model {recall_score_original}.')
        mlflow.log_metric('Recall score', recall_score_original)
        
        mlflow.log_metric('Train size', X_train.shape[0])
				