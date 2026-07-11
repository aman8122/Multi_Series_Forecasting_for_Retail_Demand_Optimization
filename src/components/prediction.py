import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from src.exception import CustomException
from src.logger import logging

# imports
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering


class PredictionPipeline:

    def __init__(self):
        try:
            # load config
            with open("config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.error("Failed to load the config.yaml file.")
            raise CustomException(e, sys)

        # set paths
        self.models_dir = os.path.abspath(self.config["paths"]["models_dir"])
        self.scaler_path = os.path.join(self.models_dir, "scaler.pkl")

    def predict_demand(self, raw_test_df, model_type="lgb"):
        try:
            logging.info(f"Prediction pipeline started using model type: {model_type}")

            # load scaler
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler pkl file not found at {self.scaler_path}! Please run train pipeline first.")
            
            scaler = joblib.load(self.scaler_path)

            # load context
            processed_test_path = os.path.abspath(self.config["paths"]["processed_test_path"])
            processed_dir = os.path.dirname(processed_test_path)
            processed_train_path = os.path.join(processed_dir, "train_processed.csv")

            if os.path.exists(processed_train_path):
                train_context = pd.read_csv(processed_train_path)
            else:
                raise FileNotFoundError(f"Processed training context not found at: {processed_train_path}")

            # transform data
            logging.info("Validating and transforming raw prediction input...")
            
            validator = DataValidation()
            clean_train_context = validator.validate_and_clean(train_context, is_train=True)
            clean_raw_test = validator.validate_and_clean(raw_test_df, is_train=False)

            engineer = FeatureEngineering()
            _, final_test = engineer.extract_features(clean_train_context, clean_raw_test)
            
            # get columns
            feature_columns = [col for col in final_test.columns if col not in ["sales", "date"]]
            
            # scale data
            final_test[feature_columns] = scaler.transform(final_test[feature_columns])
            X_test = final_test[feature_columns]

            # predict data
            predictions = None
            model_type = model_type.lower()

            if model_type == "xgb":
                # run xgb
                model_path = os.path.join(self.models_dir, self.config["models"]["xgb"])
                logging.info(f"Loading XGBoost model from {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"XGBoost model file missing: {model_path}")
                model = joblib.load(model_path)
                predictions = model.predict(X_test)

            elif model_type == "lgb":
                # run lgb
                model_path = os.path.join(self.models_dir, self.config["models"]["lgb"])
                logging.info(f"Loading LightGBM model from {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"LightGBM model file missing: {model_path}")
                model = joblib.load(model_path)
                predictions = model.predict(X_test)

            elif model_type == "lstm":
                # run lstm
                model_path = os.path.join(self.models_dir, self.config["models"]["lstm"])
                logging.info(f"Loading LSTM deep learning model from {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"LSTM model file missing: {model_path}")
                
                model = tf.keras.models.load_model(model_path)
                # reshape tensor
                X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))
                predictions = model.predict(X_test_lstm).flatten()

            else:
                raise ValueError("Invalid model_type! Choose from 'xgb', 'lgb', or 'lstm'.")

            # save output
            logging.info("Mapping predictions back to output dataframe")
            output_df = raw_test_df.copy()
            output_df['predicted_sales'] = predictions

            output_path = os.path.abspath(self.config["paths"]["predictions_output"])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df.to_csv(output_path, index=False)
            
            print(f" Predictions saved successfully using [{model_type.upper()}] at: {output_path}")
            return output_path

        except Exception as e:
            raise CustomException(e, sys)