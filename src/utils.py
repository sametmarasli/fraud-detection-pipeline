import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
import logging
import json

def init_logger():
    logging.getLogger('fraud-detection')
    logging.basicConfig(level=logging.INFO)

def save_pipeline(pipline_to_persist, pipeline_name) -> None:
    """Persist the pipeline"""
    save_path = os.path.join('./models',pipeline_name)
    joblib.dump(pipline_to_persist, save_path)

def load_pipeline(pipeline_name) -> Pipeline :
    """Load the pipeline"""
    pipeline_path = os.path.join('./models',pipeline_name)
    return joblib.load(pipeline_path)

def load_new_training_data(path):
	data = []
	with open(path, "r") as f:
		for line in f:
			data.append(pd.DataFrame(json.loads(line)))
	return pd.concat(data)

def load_dataset(file_path, debug=None) -> pd.DataFrame:
    """load csv data"""
    
    if debug:
        logging.info(f'Debug mode is on. Size of training data {int(debug)}')
        return pd.read_csv(file_path, nrows=debug)
    
    df = pd.read_csv(file_path)    
    logging.info(f'Debug mode is off. Size of training data {df.shape[0]}')
    return df

def initialize_data(step):
    logging.info(f'initialize data train until step: {step}')

    df = pd.read_csv('./data/fraud.csv')
    df_train_model = df.query("type=='TRANSFER' | type=='CASH_OUT' ")
    
    df_train_model = df_train_model.reset_index(drop=True).reset_index()
    df_train_model = df_train_model.rename(columns = {'index':'id'})

    df_train = df_train_model[df_train_model['step']<=step]
    df_train = df_train.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                            'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})  

    df_test = df_train_model[df_train_model['step']>step]
    df_test = df_test.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                            'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})  

    df_train.to_csv('./data/train.csv', index=None)
    df_test.to_csv('./data/test.csv', index=None)

def store_data(message):
    message_file = './data/stream_data.txt'
    with open(message_file, "a") as f:
        f.write("%s\n" % (json.dumps(message)))

def store_predictions(message, pipeline_name):
    message_file = f'./data/predictions_{pipeline_name}.txt'
    with open(message_file, "a") as f:
        f.write("%s\n" % (json.dumps(message)))

def stream_predict(message_data, model):
    data = pd.DataFrame(message_data, index=range(len(message_data)))
    output = {
        "id": list(data['id']), 
        "target": list(data['isFraud']), 
        "predictions": list(model.predict_proba(data)[:,-1].astype(float))
        }
    return output