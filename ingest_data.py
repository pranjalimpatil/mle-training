import argparse
import logging
import os
import sys
import tarfile

import matplotlib.pyplot as plt
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


def ingest_data(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path for training dataset", type=str)
    parser.add_argument("--log-level", help="logging level", type=str, choices=["DEBUG", "ERROR", "WARNING", "DEBUG"])
    parser.add_argument("--log-path", help="log file path", type=str)
    parser.add_argument("--no-console-log", help="log to console", type=str)

    args = parser.parse_args()
    print(args)
    if args.path is None:
        HOUSING_PATH = os.path.join("datasets", "housing")
    else:
        HOUSING_PATH = os.path.join(args.path, "housing")

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
        file_handler = logging.FileHandler(os.path.join(args.log_path, "ingest_data.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if args.no_console_log == "false" or args.no_console_log is None:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()

    def load_housing_data(housing_path=HOUSING_PATH):
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)

    logging.debug("Downloading dataset started")
    fetch_housing_data()
    logging.debug("Downloading datset completed")
    logging.debug("Loading dataset started")
    housing = load_housing_data()
    logging.debug("Loading dataset completed")

    logging.debug("Processing train and test data")
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    def income_cat_proportions(data):
        return data["income_cat"].value_counts() / len(data)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = 100 * compare_props["Random"] / (compare_props["Overall"] - 100)
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / (compare_props["Overall"] - 100)

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    logging.debug("calculating the correlation matrix")
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    logging.debug("perform feature engineering")
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    logging.debug("computing imputer")
    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(X_test_prepared, columns=X_test_num.columns, index=X_test.index)
    X_test_prepared["rooms_per_household"] = X_test_prepared["total_rooms"] / X_test_prepared["households"]
    X_test_prepared["bedrooms_per_room"] = X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    X_test_prepared["population_per_household"] = X_test_prepared["population"] / X_test_prepared["households"]

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    logger.debug("Data processing completed")

    logger.debug("saving the train and test data files to the directory")
    TRAIN_DATA_PATH = os.path.join("data", "train")
    os.makedirs(TRAIN_DATA_PATH, exist_ok=True)
    housing_prepared.to_csv(os.path.join(TRAIN_DATA_PATH, "train.csv"), index=False)
    housing_labels.to_csv(os.path.join(TRAIN_DATA_PATH, "train_labels.csv"), index=False)

    TRAIN_DATA_PATH = os.path.join("data", "test")
    os.makedirs(TRAIN_DATA_PATH, exist_ok=True)
    X_test_prepared.to_csv(os.path.join(TRAIN_DATA_PATH, "train.csv"), index=False)
    y_test.to_csv(os.path.join(TRAIN_DATA_PATH, "train_labels.csv"), index=False)

    logger.debug("data ingestion completed")

    with mlflow.start_run(experiment_id=1, run_name="Ingest Data", nested=True):
        mlflow.log_artifacts(os.path.join(os.getcwd(), "data", "train"), artifact_path="train")
        mlflow.log_artifacts(os.path.join(os.getcwd(), "data", "test"), artifact_path="test")
        mlflow.log_artifacts(os.path.join(os.getcwd(), "datasets", "housing"), artifact_path="dataset")


if __name__ == "__main__":
    ingest_data(sys.argv)
