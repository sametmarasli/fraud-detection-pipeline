import pandas as pd
import json
import threading
import uuid

from pathlib import Path
from kafka import KafkaProducer, KafkaConsumer
from time import sleep
from src import pipeline_config

KAFKA_HOST = 'localhost:9092'
df_test = pd.read_csv(pipeline_config.TEST_DATA)


def start_producing():
    producer = KafkaProducer(bootstrap_servers=KAFKA_HOST)    
    for step, df in  df_test.groupby('step').__iter__():

        message_id = str(uuid.uuid4())
        message = {'request_id': message_id, 'step':str(step), 'data': json.loads(df.sample(n=5).to_json(orient='records'))}
        
        producer.send('transaction_messages', json.dumps(message).encode('utf-8'))
        producer.flush()
        print(message)
        print("\033[1;31;40m -- PRODUCER: Sent message with id {}".format(message_id))
        sleep(2)
        



threads = []
t = threading.Thread(target=start_producing)
threads.append(t)
t.start()