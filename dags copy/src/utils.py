import logging
import joblib
from sklearn.pipeline import Pipeline
import os

def init_logger():
    logging.getLogger('fraud-detection')
    logging.basicConfig(level=logging.INFO)



# def load_pipeline(pipeline_name) -> Pipeline :
#     """Load the pipeline"""
#     pipeline_path = os.path.join('./models',pipeline_name)
#     return joblib.load(pipeline_path, ==)