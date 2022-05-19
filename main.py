import mlflow
import mlflow.sklearn

from ingest_data import ingest_data
from score import score
from train import train

remote_server_uri = "http://0.0.0.0:5000"
mlflow.set_tracking_uri(remote_server_uri)
exp_name = "House_price_prediction"
mlflow.set_experiment(exp_name)

with mlflow.start_run(experiment_id=1, run_name="House Price Prediction") as run:
    ingest_data()
    train()
    score()
