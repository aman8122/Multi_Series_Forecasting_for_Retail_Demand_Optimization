import os
import sys
import yaml
import joblib
from sklearn.preprocessing import MinMaxScaler
from src.exception import CustomException
from src.logger import logging


class DataTransformation:

    def __init__(self):
        try:
            # load config
            with open("config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.error("Failed to load config.yaml file.")
            raise CustomException(e, sys)
            
        # init scaler
        self.scaler = MinMaxScaler()

    def scale_features(self, engineered_train, engineered_test, feature_cols):
        try:
            logging.info("MinMax feature array scaling init.")
            
            # copy data
            train_df = engineered_train.copy()
            test_df = engineered_test.copy()

            # scale columns
            train_df[feature_cols] = self.scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = self.scaler.transform(test_df[feature_cols])

            # save scaler
            target_dir = os.path.abspath(self.config["paths"]["models_dir"])
            os.makedirs(target_dir, exist_ok=True)
            output_scaler_path = os.path.join(target_dir, "scaler.pkl")
            joblib.dump(self.scaler, output_scaler_path)
            
            logging.info(f"Scaler saved successfully at destination path: {output_scaler_path}")
            return train_df, test_df, self.scaler

        except Exception as e:
            raise CustomException(e, sys)