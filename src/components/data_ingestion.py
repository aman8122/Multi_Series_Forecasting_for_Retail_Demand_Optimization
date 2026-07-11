import os
import sys
import yaml
import pandas as pd
from src.logger import logging
from src.exception import CustomException


class DataIngestion:

    def __init__(self):
        try:
            # load config
            with open("config.yaml", "r") as f:
                self.config=yaml.safe_load(f)
        except Exception as e:
            logging.error("Failed to load config.yaml file.")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion...")

            # get paths
            train_p=os.path.abspath(self.config["paths"]["raw_data_path"])
            test_p=os.path.abspath(self.config["paths"]["test_data_path"])

            # load train
            logging.info(f"Loading train data from: {train_p}")
            train_df=pd.read_csv(train_p)
            
            # load test
            logging.info(f"Loading test data from: {test_p}")
            test_df=pd.read_csv(test_p)

            logging.info("Data loaded successfully.")
            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)