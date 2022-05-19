import argparse
import logging
import os
import pickle
import sys
import tarfile
from cmath import exp

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

LINEAR_MODEL_FILE = "linear_regression.pkl"
DECISION_TREE_MODEL_FILE = "tree_regression.pkl"
RANDOM_FOREST_MODEL_FILE = "randomforest_regression.pkl"


def score(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", help="path for model folder", type=str)
    parser.add_argument("--data-folder", help="path for data folder", type=str)
    parser.add_argument("--log-level", help="logging level", type=str, choices=["DEBUG", "ERROR", "WARNING", "DEBUG"], const="DEBUG", nargs="?")
    parser.add_argument("--log-path", help="log file path", type=str)
    parser.add_argument("--no-console-log", help="log to console", type=str)

    args = parser.parse_args()
    print(args)
    if args.model_folder is not None:
        MODEL_PATH = os.path.join(args.model_folder, "models")
    else:
        MODEL_PATH = os.path.join("data", "models")

    if args.data_folder is not None:
        OUTPUT_FOLDER = os.path.join(args.data_folder, "test")
    else:
        OUTPUT_FOLDER = os.path.join("data", "test")

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("%(asctime)s,%(name)s,%(message)s")

    if args.log_level == "INFO":
        logger.setLevel(logging.INFO)
    elif args.log_level == "WARNING":
        logger.setLevel(logging.WARNING)
    elif args.log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)

    if args.log_path is not None:
        os.makedirs(args.log_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(args.log_path, "score_log.txt"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if args.no_console_log == "false" or args.no_console_log is None:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.debug("Loading test data")
    X_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, "test.csv"))
    y_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, "test_labels.csv"))
    y_test = np.array(y_test).ravel()

    table = PrettyTable(["Model", "MSE", "RMSE", "MAE"])

    linear_list = ["Linear Regression"]
    decision_tree_list = ["Decision Tree"]
    rf_list = ["Random Forest"]

    logger.debug("Predicting and calculating metric for linear model")
    with open(os.path.join(MODEL_PATH, DECISION_TREE_MODEL_FILE), "rb") as file:
        model = pickle.load(file=file)

    housing_predictions = model.predict(X_test)
    lin_mse = mean_squared_error(y_test, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(y_test, housing_predictions)
    linear_list.extend([lin_mse, lin_rmse, lin_mae])
    linear_regression_metrics = {
        "lr_mse": lin_mse,
        "lr_rmse": lin_rmse,
        "lr_mae": lin_mae,
    }

    logger.debug("Predicting and calculating metric for decsion tree model")
    with open(os.path.join(MODEL_PATH, DECISION_TREE_MODEL_FILE), "rb") as file:
        model = pickle.load(file=file)

    housing_predictions = model.predict(X_test)
    tree_mse = mean_squared_error(y_test, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(y_test, housing_predictions)
    decision_tree_list.extend([lin_mse, lin_rmse, lin_mae])
    decision_tree_metrics = {
        "dt_mse": tree_mse,
        "dt_rmse": tree_rmse,
        "dt_mae": tree_mae,
    }

    logger.debug("Predicting and calculating metric for Random Forest model")
    with open(os.path.join(MODEL_PATH, RANDOM_FOREST_MODEL_FILE), "rb") as file:
        model = pickle.load(file=file)

    housing_predictions = model.predict(X_test)
    rf_mse = mean_squared_error(y_test, housing_predictions)
    rf_rmse = np.sqrt(tree_mse)
    rf_mae = mean_absolute_error(y_test, housing_predictions)
    rf_list.extend([lin_mse, lin_rmse, lin_mae])
    random_forest_metrics = {
        "rf_mse": tree_mse,
        "rf_rmse": tree_rmse,
        "rf_mae": tree_mae,
    }

    logger.debug("printing scores")
    table.add_row(linear_list)
    table.add_row(decision_tree_list)
    table.add_row(rf_list)

    print(table)
    logger.debug("completed")

    with mlflow.start_run(experiment_id=1, run_name="score", nested=True):
        mlflow.log_metrics(linear_regression_metrics)
        mlflow.log_metrics(decision_tree_metrics)
        mlflow.log_metrics(random_forest_metrics)


if __name__ == "__main__":
    score(sys.argv)
