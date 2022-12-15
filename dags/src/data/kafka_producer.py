from kafka  import KafkaProducer
import pickle
import os
import logging
from time import sleep
from json import dumps
from sklearn.model_selection import train_test_split
import numpy as np


def encode_to_json(x_train, y_train):
	x = dumps(x_train.tolist())
	y = dumps(y_train.tolist())

	jsons_comb = [x, y]

	return jsons_comb

def generate_stream(**kwargs):

	producer = KafkaProducer(bootstrap_servers=['localhost:9092'],                              # set up Producer
                         value_serializer=lambda x: dumps(x).encode('utf-8'))

	stream_sample = pickle.load(open(os.getcwd() + kwargs['path_stream_sample'], "rb"))       # load stream sample file
	# stream_sample = pickle.load(open('./data/stream_sample.p', "rb"))       # load stream sample file

	x_new = stream_sample[0]
	y_new = stream_sample[1]
	
	frauds_sent = 0
	for _ in range(10):	
		x_send, x_keep, y_send, y_keep = train_test_split(x_new, y_new, shuffle=True, train_size= int(1e3)*np.random.choice(range(1,5)), stratify=y_new)
		json_comb = encode_to_json(x_send.values, y_send.values)                               # pick observation and encode to JSON
		producer.send('TopicA', value=json_comb)                                               # send encoded observation to Kafka topic
		frauds_sent += sum(y_send.values)
		
		sleep(.1)
	logging.info(f'{frauds_sent} frauds sent')
	producer.close()

if __name__ == "__main__":
	PATH_STREAM_SAMPLE = "/data/stream_sample.p"

	logging.basicConfig(level=logging.INFO)
	input_dict = {
		"path_stream_sample" : PATH_STREAM_SAMPLE,

	}

	generate_stream(**input_dict)
