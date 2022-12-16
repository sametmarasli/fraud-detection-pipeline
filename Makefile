service := airflow

docker-up-service:
	docker-compose -f docker-compose-fraud.yml up --build $(service)

docker-up:
	docker-compose -f  docker-compose-fraud.yml up --build -d 

local-up:
	docker-compose -f  docker-compose-local.yml up --build -d 

flask-up:
	docker-compose -f  docker-compose-flask.yml up --build -d

flask-bash:
	docker exec -ti flask_container bash

docker-build:
	docker-compose -f docker-compose-fraud.yml build

docker-down:
	docker-compose down

train:
	python train.py --path_config=./ml_config.yaml --debug=True

evaluate:
	python evaluate.py --path_config=./ml_config.yaml

bash-airflow:
	docker exec -ti airflow_container bash