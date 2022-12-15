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

def load_data(file_path, debug=None) -> pd.DataFrame:
    """load csv data"""
    
    if debug:
        logging.info(f'Debug mode is on. Size of training data {int(debug)}')
        return pd.read_csv(file_path, nrows=debug)
    
    df = pd.read_csv(file_path)    
    logging.info(f'Debug mode is off. Size of training data {df.shape[0]}')
    return df

def sample_data(file_path, step) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df[df['step']<step].to_csv('./data/fraud_sample.csv', index=None)
    print(f'Sample is created below step {step}')
    return
    
def initialize_data(**kwargs):
    # logging.info(f'initialize data train until step: {step}')

    df = pd.read_csv(kwargs['path_data']).query("type=='TRANSFER' | type=='CASH_OUT' ")

    df = df.reset_index(drop=True).reset_index()

    df = df.rename(columns={
        'index':'id',
        'oldbalanceOrg':'oldBalanceOrig',
        'newbalanceOrig':'newBalanceOrig',
        'oldbalanceDest':'oldBalanceDest',
        'newbalanceDest':'newBalanceDest'
        })  

    # df_train = df[df['step']<=step]
    # df_test = df[df['step']>step]
 
    # df_train.to_csv('./data/train.csv', index=None)
    # df_test.to_csv('./data/test.csv', index=None)
    from sklearn.model_selection import train_test_split
    import pickle

    X, y = df.drop('isFraud', axis=1), df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle=True, 
    test_size=.2,stratify=y )

    X_train, X_stream, y_train, y_stream = train_test_split(X_train,y_train, shuffle=True, test_size=.2,stratify=y_train )

    train_set = [X_train, y_train]
    test_set = [X_test, y_test]
    stream_sample = [X_stream, y_stream]

    pickle.dump(test_set, open(os.getcwd() + kwargs['path_test_set'], "wb"))
    pickle.dump(stream_sample, open(os.getcwd() + kwargs['path_stream_sample'], "wb"))


    return X_train, y_train, X_test, y_test

def store_data(message):
    message_file = './data/stream_data.txt'
    with open(message_file, "a") as f:
        f.write("%s\n" % (json.dumps(message)))

def store_predictions(message, pipeline_name):
    message_file = f'./data/predictions_{pipeline_name}.txt'
    with open(message_file, "a") as f:
        f.write("%s\n" % (json.dumps(message)))

def predict(message_data, model):
    data = pd.DataFrame(message_data, index=range(len(message_data)))
    output = {
        "id": list(data['id']), 
        "target": list(data['isFraud']), 
        "predictions": list(model.predict_proba(data)[:,-1].astype(float))
        }
    return output