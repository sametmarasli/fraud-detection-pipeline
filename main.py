from src.model import train_model
import src.pipeline_config as pipeline_config
from src.utils import init_logger

init_logger()

if __name__ == "__main__":
    train_model(pipeline_config, debug=1e3)