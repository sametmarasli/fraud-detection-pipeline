from src.transformers import (KeepFeatures, CustomLabelEncoder, AmountVsOldAndNewBalanceOrig, AmountVsOldAndNewBalanceDest)
import src.pipeline_config as pipeline_config
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier

model_pipeline = Pipeline([
      ('keep_features', KeepFeatures(pipeline_config.FEATURES)),
      ('label_encode', CustomLabelEncoder(pipeline_config.LABEL_ENCODE_FEATURES)),
      ('amount_vs_old_new_balance_orig', AmountVsOldAndNewBalanceOrig()),
      ('amount_vs_old_new_balance_dest', AmountVsOldAndNewBalanceDest()),
      ('xgb', XGBClassifier())
      ])