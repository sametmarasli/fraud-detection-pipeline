from kafka import KafkaConsumer, KafkaProducer
import json

producer = KafkaProducer(bootstrap_servers='localhost:9092')
def publish_prediction(pred, request_id):
	producer.send('app_messages', json.dumps({'request_id': request_id, 'prediction': pred}).encode('utf-8'))
	producer.flush()