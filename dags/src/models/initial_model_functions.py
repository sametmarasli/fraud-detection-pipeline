import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
import logging
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from dags.src.models.transformers import (KeepFeatures, CustomLabelEncoder, AmountVsOldAndNewBalanceOrig, AmountVsOldAndNewBalanceDest)
import dags.src.models.pipeline_config as pipeline_config
from sklearn.model_selection import  GridSearchCV
import joblib 
from sklearn.metrics import f1_score, precision_score, recall_score

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
    X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle=True, test_size=.2,stratify=y )
    X_train, X_stream, y_train, y_stream = train_test_split(X_train,y_train, shuffle=True, test_size=.95,stratify=y_train )

    train_set = [X_train, y_train]
    test_set = [X_test, y_test]
    stream_sample = [X_stream, y_stream]

    pickle.dump(train_set, open(os.getcwd() + kwargs['path_train_set'], "wb"))
    pickle.dump(test_set, open(os.getcwd() + kwargs['path_test_set'], "wb"))
    pickle.dump(stream_sample, open(os.getcwd() + kwargs['path_stream_sample'], "wb"))
    logging.info(f'Dimensions train: {y_train.shape}, test:{y_test.shape}, strea:{y_stream.shape}')
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
    # ti = kwargs['ti']
    # loaded = ti.xcom_pull(task_ids='load_preprocess')
    loaded_train = pd.read_pickle('./data/train_set.p')
    loaded_test = pd.read_pickle('./data/test_set.p')
    # loaded_test = pd.read_pickle('./legacy_data/stream_sample.p')
    
    # logging.info('variables successfully fetched from previous task')
    X_train = loaded_train[0]
    y_train = loaded_train[1]
    X_test = loaded_test[0]
    y_test = loaded_test[1]
    
    # initialize the model
    model_pipeline = construct_model()

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
    
				

if __name__ == "__main__":
    PATH_STREAM_SAMPLE = "/data/stream_sample.p"
    PATH_TEST_SET = "/data/test_set.p"
    PATH_TRAIN_SET = "/data/train_set.p"
    PATH_DATA = "/data/fraud_sample.csv"
    INITIAL_MODEL_PATH = '/models/current_model/initial_model.p'


    logging.basicConfig(level=logging.INFO)
    INITIAL_MODEL_PATH = '/models/current_model/initial_model.p'
    op_kwargs={
    'initial_model_path': INITIAL_MODEL_PATH,
    'path_data':PATH_DATA,
    'path_train_set':PATH_TRAIN_SET,
    'path_test_set':PATH_TEST_SET,
    'path_stream_sample':PATH_STREAM_SAMPLE,
    
    }

    load_preprocess(**op_kwargs)
    fit_model(**op_kwargs)