import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
import logging
from src.transformers import SelectToModel
import json

def load_dataset(file_path, debug=False) -> pd.DataFrame:
    """load csv data"""
    
    if debug:
        size = int(5e5)
        logging.info(f'Debug mode is on. Size of training data {size}')
        return pd.read_csv(file_path, nrows=size)
    
    df = pd.read_csv(file_path)    
    logging.info(f'Debug mode is off. Size of training data {df.shape[0]}')
    return df

def save_pipeline(pipline_to_persist, pipeline_name) -> None:
    """Persist the pipeline"""
    save_path = os.path.join('./models',pipeline_name)
    joblib.dump(pipline_to_persist, save_path)

def load_pipeline(pipeline_name) -> Pipeline :
    """Load the pipeline"""
    pipeline_path = os.path.join('./models',pipeline_name)
    return joblib.load(pipeline_path)

def initialize_data():
    df = pd.read_csv('./data/fraud.csv')
    df_train_model = SelectToModel().transform(df)

    df_train = df_train_model[df_train_model['step']<=400]
    df_train = df_train.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                            'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})  

    df_test = df_train_model[df_train_model['step']>400]
    df_test = df_test.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                            'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})  

    df_train.to_csv('./data/train.csv', index=None)
    df_test.to_csv('./data/test.csv', index=None)


def load_new_training_data(path):
	data = []
	with open(path, "r") as f:
		for line in f:
			data.append(pd.DataFrame(json.loads(line)))
	return pd.concat(data)