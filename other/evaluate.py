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
def evaluate_model(path_config):
    """Evaluate the model"""

    with open(path_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    df = load_dataset(config["TEST_DATA"])
    y_test = df.pop(config["TARGET"])
    X_test = df
    logging.info(f"Evaluating for size {df.shape[0]}")
    
    model_pipeline = load_pipeline(config["PIPELINE_NAME"])
    probabilities = model_pipeline.predict_proba(X_test)
    
    score = average_precision_score(y_test, probabilities[:, 1])

    logging.info(f'AUPRC = {score}') 



if __name__ == "__main__":
    evaluate_model()
    




