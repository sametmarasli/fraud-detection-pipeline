from src.utils import load_dataset, save_pipeline, load_pipeline
from src import transformers

from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import average_precision_score
import logging
import click
import yaml


logging.basicConfig(level=logging.INFO)

@click.command()
@click.option("--path_config", required=True)

def predict(path_config, debug=False):



if __name__ == "__main__":
    train_model()
    




