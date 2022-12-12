import json 
import pandas as pd
from src.kafka_utils import publish_prediction
from src.utils import load_pipeline
import yaml
from kafka import KafkaConsumer
import os 

with open('./ml_config.yaml', "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def is_application_message(msg):
    message = json.loads(msg.value)
    return msg.topic == 'app_messages' and 'prediction' not in message

def init_predictions(model_name):
    message_file = f'predictions_{model_name}.txt'
    if message_file in os.listdir('./'):
        os.remove(message_file)

def store_predictions(message, model_name):
    message_file = f'predictions_{model_name}.txt'
    with open(message_file, "a") as f:
        f.write("%s\n" % (json.dumps(message)))

def predict(message, model,request_id):
    data = pd.DataFrame(message, index=range(len(message)))
    output = {
        "request_id": request_id,
        "target": list(data['isFraud']), 
        "preds": list(model.predict_proba(data)[:,-1].astype(float))
        }
    return output

def start():
    
    for msg in consumer:
        message = json.loads(msg.value)

        if is_application_message(msg):
            request_id = message['request_id']
            pred = predict(message['data'], model, request_id)
      
            store_predictions(pred , config['PIPELINE_NAME'])
            # append_message_db(message['data'])


if __name__ == '__main__':

    init_predictions(config['PIPELINE_NAME'])
    model = load_pipeline(config['PIPELINE_NAME'])

    KAFKA_HOST = 'localhost:9092'
    TOPICS = ['app_messages']
    
    consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST)
    consumer.subscribe(TOPICS)
    start()