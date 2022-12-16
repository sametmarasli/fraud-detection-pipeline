# import keras
# from keras.models import Sequential, load_model
import os
import time
import logging
import mlflow.sklearn
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  GridSearchCV, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib

from src.models.transformers import (KeepFeatures, CustomLabelEncoder, AmountVsOldAndNewBalanceOrig, AmountVsOldAndNewBalanceDest)
import src.models.pipeline_config as pipeline_config

def load_current_model(model_path, file_m):
	logging.info("loading the current model")
	model = joblib.load(os.getcwd()+model_path + str(file_m))
	return model

def construct_model():

    model_pipeline = Pipeline([
        ('keep_features', KeepFeatures(pipeline_config.FEATURES)),
        ('label_encode', CustomLabelEncoder(pipeline_config.LABEL_ENCODE_FEATURES)),
        # ('amount_vs_old_new_balance_orig', AmountVsOldAndNewBalanceOrig()),
        # ('amount_vs_old_new_balance_dest', AmountVsOldAndNewBalanceDest()),
        ('xgb', XGBClassifier())
        ])

    return model_pipeline

def update_model(**kwargs):

	ti = kwargs['ti']
	loaded = ti.xcom_pull(task_ids='preprocessing')

	# logging.info('variables successfully fetched from previous task')
	
	train_set = loaded[0]
	test_set = loaded[1]
	new_samples = loaded[2]
	

	# load new samples

	x_train = train_set[0]
	y_train = train_set[1]

	x_test = test_set[0]
	y_test = test_set[1]
	
	x_new = new_samples[0]
	y_new = new_samples[1]
	
	# get current_model

	for file_m in os.listdir(os.getcwd()+kwargs['path_current_model']):
		if '.p' in file_m:

			mlflow.set_tracking_uri('http://mlflow:5000')

			with mlflow.start_run():

				model = load_current_model(kwargs['path_current_model'], file_m)

				# get score of current model
				
				THRESHOLD = .5
				probabilities = model.predict_proba(x_test)

				f1_score_original = f1_score(y_test, probabilities[:, 1]>THRESHOLD)
				logging.info(f'F1 score of the original model {f1_score_original}.')
				precision_score_original = precision_score(y_test, probabilities[:, 1]>THRESHOLD)
				logging.info(f'Precision score of the original model {precision_score_original}.')
				recall_score_original = recall_score(y_test, probabilities[:, 1]>THRESHOLD)
				logging.info(f'Recall score of the original model {recall_score_original}.')
								
				# update model with new data and evaluate score
				
				# for old_data in os.listdir(os.getcwd()+kwargs['path_used_data']):

				# 	old_data_i = pd.read_pickle(os.getcwd()+kwargs['path_used_data']+old_data)
				# 	x_old = old_data_i[0]
				# 	y_old = old_data_i[1]

				# 	x_new_data = pd.concat((x_new, x_old))
				# 	y_new_data = pd.concat((y_new, y_old))
				# 	logging.info(f'data {old_data} is added to the original data')

				x_new_data = pd.concat((x_train, x_new))
				y_new_data = pd.concat((y_train, y_new))
				logging.info(f'new data is added to the original data')


				## Reset Hyperparameters with new data and retrain 

				model_pipeline = construct_model()

				logging.info('Reset the hyperparameters and retrain the model with new data')

				search = GridSearchCV(model_pipeline, param_grid=pipeline_config.MODEL_PARAMETER_GRID, n_jobs=-1, scoring=pipeline_config.EVAL_METRIC, cv=3, verbose=0)
				X_sample, X_, y_sample, y_ = train_test_split(x_new_data, y_new_data, train_size=.3, stratify=y_new_data)
				search.fit(X_sample, y_sample)
				logging.info("Best parameter (CV score=%0.3f):" % search.best_score_)
				logging.info(search.best_params_)

				# Retrain the selected model with all the data
				logging.info('Fit the best estimator with all the data')
				new_model = search.best_estimator_.fit(x_new_data, y_new_data)

				probabilities = new_model.predict_proba(x_test)

				f1_score_updated = f1_score(y_test, probabilities[:, 1]>THRESHOLD)
				logging.info(f'F1 score of the updated model {f1_score_original}.')
				precision_score_updated = precision_score(y_test, probabilities[:, 1]>THRESHOLD)
				logging.info(f'Precision score of the updated model {precision_score_original}.')
				recall_score_updated = recall_score(y_test, probabilities[:, 1]>THRESHOLD)
				logging.info(f'Recall score of the updated model {recall_score_original}.')

				# log results to MLFlow


				mlflow.log_metric('F1 score', f1_score_updated)
				mlflow.log_metric('Precision score', precision_score_updated)
				mlflow.log_metric('Recall score', recall_score_updated)
				
				mlflow.log_metric('Number of samples used for original model', x_train.shape[0])
				mlflow.log_metric('Number of new samples used for updated model', x_new.shape[0])

				mlflow.log_metric('Train size', x_new_data.shape[0])


				# if the updated model outperforms the current model -> move current version to archive and promote the updated model

				if f1_score_updated - f1_score_original > 0:

					logging.info('Updated model stored')
					mlflow.set_tag('status', 'the model from this run replaced the current version ')

					updated_model_name = 'model_' + str(time.strftime("%Y%m%d_%H%M"))
					joblib.dump(new_model, os.getcwd()+kwargs['path_current_model'] + updated_model_name + '.p')

					os.rename(os.getcwd()+kwargs['path_current_model']+file_m, os.getcwd()+kwargs['path_model_archive']+file_m)

				else:
					logging.info('Current model maintained')
					mlflow.set_tag('status', 'the model from this run did not replace the current version ')

		else:

			logging.info(file_m + ' is not a model')

def data_to_archive(**kwargs):

	# store data that was used for updating the model in archive along date + time tag

	for file_d in os.listdir(os.getcwd()+kwargs['path_new_data']):
		if 'new_samples.p' in file_d:

			os.rename(os.getcwd()+kwargs['path_new_data'] + file_d, os.getcwd()+kwargs['path_used_data'] + file_d)

			logging.info('data used for updating the model has been moved to archive')

		else:
			print('no data found')