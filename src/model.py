from src.utils import load_dataset, save_pipeline, load_pipeline
from src.pipeline import model_pipeline
import logging
# from pipeline_config import TRAIN_DATA, TARGET, PIPELINE_NAME
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split



def train_model(path_config, debug=False):
    """Train the model"""
    df = load_dataset(file_path=path_config.TRAIN_DATA, debug=1e5)
    ##
    # concat with other model to reload
    ##
    
    df, _ = train_test_split(df, train_size=.3, stratify=df[path_config.TARGET] )
    logging.info(f"Tuning the algorithm by using sample size {int(df.shape[0])}")

    y_train = df.pop(path_config.TARGET)
    X_train = df
    
    search = GridSearchCV(model_pipeline, param_grid=path_config.MODEL_PARAMETER_GRID, n_jobs=-1, scoring=path_config.EVAL_METRIC, cv=3, verbose=0)
    search.fit(X_train, y_train)

    logging.info("Best parameter (CV score=%0.3f):" % search.best_score_)
    logging.info(search.best_params_)

    save_pipeline(pipline_to_persist=search.best_estimator_, pipeline_name=path_config.PIPELINE_NAME)






