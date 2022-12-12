import pandas as pd
import json
import threading
import uuid

from pathlib import Path
from kafka import KafkaProducer, KafkaConsumer
from time import sleep
import yaml

with open('./ml_config.yaml', "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


KAFKA_HOST = 'localhost:9092'
df_test = pd.read_csv(config["TEST_DATA"])


def start_producing():
    producer = KafkaProducer(bootstrap_servers=KAFKA_HOST)    
    for step, df in  df_test.groupby('step').__iter__():

        message_id = str(uuid.uuid4())
        message = {'request_id': message_id, 'data': json.loads(df.to_json(orient='records'))}
        
        producer.send('app_messages', json.dumps(message).encode('utf-8'))
        producer.flush()
        
        print("\033[1;31;40m -- PRODUCER: Sent message with id {}".format(message_id))
        sleep(2)


def start_consuming():
	consumer = KafkaConsumer('app_messages', bootstrap_servers=KAFKA_HOST)

	for msg in consumer:
		message = json.loads(msg.value)
		if 'prediction' in message:
			request_id = message['request_id']
			print("\033[1;32;40m ** CONSUMER: Received prediction {} for request id {}".format(message['prediction'], request_id))



threads = []
t = threading.Thread(target=start_producing)
t2 = threading.Thread(target=start_consuming)
threads.append(t)
threads.append(t2)
t.start()
t2.start()

print('started Producing/Consuming')