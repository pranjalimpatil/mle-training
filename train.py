import argparse
import logging
import os
import pickle
import sys
import tarfile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
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


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-folder", help="path for train folder", type=str)
    parser.add_argument("--output-folder", help="path for output folder", type=str)
    parser.add_argument("--log-level", help="logging level", type=str, choices=["DEBUG", "ERROR", "WARNING", "DEBUG"], const="DEBUG", nargs="?")
    parser.add_argument("--log-path", help="log file path", type=str)
    parser.add_argument("--no-console-log", help="log to console", type=str)

    args = parser.parse_args()
    print(args)
    if args.train_folder is not None:
        TRAIN_DATA_PATH = os.path.join(args.train_folder, "train")
    else:
        TRAIN_DATA_PATH = os.path.join("data", "train")

    if args.output_folder is not None:
        OUTPUT_FOLDER = os.path.join(args.output_folder, "models")
    else:
        OUTPUT_FOLDER = os.path.join("data", "models")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
        file_handler = logging.FileHandler(os.path.join(args.log_path, "train_log.txt"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if args.no_console_log == "false" or args.no_console_log is None:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.debug("loading the dataset...")
    housing_prepared = pd.read_csv(os.path.join(TRAIN_DATA_PATH, "train.csv"))
    housing_labels = pd.read_csv(os.path.join(TRAIN_DATA_PATH, "train_labels.csv"))
    housing_labels = np.array(housing_labels).ravel()

    with mlflow.start_run(experiment_id=1, run_name="Train", nested=True):
        logger.debug("building linear model")
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)
        mlflow.log_param(key="linear regression", value=lin_reg.get_params())
        mlflow.sklearn.log_model(lin_reg, "linear_model")

        logger.debug("saving linear model")
        with open(os.path.join(OUTPUT_FOLDER, "linear_regression.pkl"), "wb") as file:
            pickle.dump(lin_reg, file)
        logger.debug("saved linear model")

        logger.debug("Building decision tree model")
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)
        mlflow.log_param(key="decision tree regression", value=tree_reg.get_params())
        mlflow.sklearn.log_model(tree_reg, "tree_model")

        logger.debug("saving decision tree model")
        with open(os.path.join(OUTPUT_FOLDER, "tree_regression.pkl"), "wb") as file:
            pickle.dump(tree_reg, file)
        logger.debug("saved decision tree model")

        logger.debug("Performing Hyperparameter tuning")
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
        ]

        forest_reg = RandomForestRegressor(random_state=42)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
            n_jobs=-1,
        )
        grid_search.fit(housing_prepared, housing_labels)

        final_model = grid_search.best_estimator_

        mlflow.log_param(key="random forest ", value=final_model.get_params())
        mlflow.sklearn.log_model(final_model, "Random_Forest_Model")

        with open(os.path.join(OUTPUT_FOLDER, "randomforest_regression.pkl"), "wb") as file:
            pickle.dump(final_model, file)
        logger.debug("saved best of model after hyperparameter tuning")


if __name__ == "__main__":
    train(sys.argv)
