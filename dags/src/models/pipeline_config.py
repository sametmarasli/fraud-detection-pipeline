EVAL_METRIC = 'f1'

FEATURES = ['step', 'type', 'amount', 'oldBalanceOrig', 'newBalanceOrig', 'oldBalanceDest', 'newBalanceDest']

LABEL_ENCODE_FEATURES = ['type']

MODEL_PARAMETER_GRID = {
    "xgb__max_depth": [1, 3],
    "xgb__scale_pos_weight": [1, 9],
}
