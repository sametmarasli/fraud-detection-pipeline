service := airflow

docker-up-service:
	docker-compose -f docker-compose-project.yml up --build $(service)

docker-up:
	docker-compose -f docker-compose-project.yml up --build -d

docker-build:
	docker-compose -f docker-compose-project.yml build

docker-down:
	docker-compose down

train:
	python train.py --path_config=./ml_config.yaml --debug=True

evaluate:
	python evaluate.py --path_config=./ml_config.yaml

bash-airflow:
	docker exec -ti airflow_container bash