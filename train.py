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
@click.option("--debug", default=True)
def train_model(path_config, debug=False):

    with open(path_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    """Train the model"""

    df = load_dataset(file_path=config["TRAIN_DATA"], debug=debug)

    y_train = df.pop(config["TARGET"])
    X_train = df

    model_pipeline = Pipeline([
        ('keep_features', transformers.KeepFeatures(config["FEATURES"])),
        ('label_encode', transformers.CustomLabelEncoder(config["LABEL_ENCODE_FEATURES"])),
        ('amount_vs_old_new_balance_orig', transformers.AmountVsOldAndNewBalanceOrig()),
        # ('amount_vs_old_new_balance_dest', transformers.AmountVsOldAndNewBalanceDest()),
        ('xgbclassifier', XGBClassifier(max_depth = 1, scale_pos_weight = 99, n_jobs = -1)) ,
        ])
    
    model_pipeline.fit(X_train, y_train)

    save_pipeline(pipline_to_persist=model_pipeline, pipeline_name=config["PIPELINE_NAME"])
    logging.info(f'Pipeline {config["PIPELINE_NAME"]} is saved')


if __name__ == "__main__":
    train_model()
    




