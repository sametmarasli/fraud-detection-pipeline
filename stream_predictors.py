import json 
import pandas as pd
from kafka import KafkaConsumer
from src.utils import load_pipeline
from src.pipeline_config import PIPELINE_NAME
import logging
from src.utils import load_pipeline, store_data, store_predictions, stream_predict, init_logger



def start():
    
    for msg in consumer:
        message = json.loads(msg.value)
        request_id = message['request_id']
        step = message['step']
        predictions = stream_predict(message['data'], model)   
        store_data(message['data'])
        store_predictions(predictions, PIPELINE_NAME)
        logging.info(f"Prediction is done for step: {step}")

if __name__ == '__main__':
    init_logger()
    model = load_pipeline(PIPELINE_NAME)
    
    KAFKA_HOST = 'localhost:9092'
    TOPICS = ['transaction_messages']
    consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST)

    consumer.subscribe(TOPICS)
    start()