# tracking/mlflow_utils.py

import mlflow
from config.config import MLFLOW_URI, EXPERIMENT_NAME


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def start_run(name):
    return mlflow.start_run(run_name=name)