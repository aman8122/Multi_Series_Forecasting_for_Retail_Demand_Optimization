import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from src.exception import CustomException
from src.logger import logging
import warnings
warnings.filterwarnings('ignore')

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
            logging.error("Unable to load config.yaml file.")
            raise CustomException(e, sys)

        # set paths
        self.models_dir = os.path.abspath(self.config["paths"]["models_dir"])
        self.scaler_path = os.path.join(self.models_dir, "scaler.pkl")

    def predict_demand(self, raw_test_df, model_type="lgb"):
        try:
            logging.info(f"Prediction pipeline triggered. Selected model: {model_type}")

            # load scaler
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}! Please run training first.")
            scaler = joblib.load(self.scaler_path)

            # load context
            processed_test_path = os.path.abspath(self.config["paths"]["processed_test_path"])
            processed_dir = os.path.dirname(processed_test_path)
            processed_train_path = os.path.join(processed_dir, "train_processed.csv")

            if not os.path.exists(processed_train_path):
                raise FileNotFoundError(f"Processed training context missing at: {processed_train_path}")
            train_context = pd.read_csv(processed_train_path)

            # transform data
            logging.info("Running data validation on incoming raw test stream...")
            validator = DataValidation()
            clean_train_context = validator.validate_and_clean(train_context, is_train=True)
            clean_raw_test = validator.validate_and_clean(raw_test_df, is_train=False)

            logging.info("Extracting lag and rolling window features...")
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
                logging.info(f"Loading XGBoost model: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"XGBoost file missing: {model_path}")
                model = joblib.load(model_path)
                predictions = model.predict(X_test)

            elif model_type == "lgb":
                # run lgb
                model_path = os.path.join(self.models_dir, self.config["models"]["lgb"])
                logging.info(f"Loading LightGBM model: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"LightGBM file missing: {model_path}")
                model = joblib.load(model_path)
                predictions = model.predict(X_test)

            elif model_type == "lstm":
                # run lstm
                model_path = os.path.join(self.models_dir, self.config["models"]["lstm"])
                logging.info(f"Loading LSTM Keras model: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"LSTM file missing: {model_path}")
                
                model = tf.keras.models.load_model(model_path)
                # reshape tensor
                X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))
                predictions = model.predict(X_test_lstm).flatten()

            else:
                raise ValueError("Invalid model type provided. Choose from 'xgb', 'lgb', or 'lstm'.")

            # save output
            logging.info("Formatting output dataframe and mapping predictions...")
            output_df = raw_test_df.copy()
            output_df['predicted_sales'] = np.round(predictions).astype(int)

            output_path = os.path.abspath(self.config["paths"]["predictions_output"])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df.to_csv(output_path, index=False)
            
            print(f" Predictions saved successfully using [{model_type.upper()}] at: {output_path}")
            return output_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # run main
        from src.components.data_ingestion import DataIngestion
        ingestion = DataIngestion()
        _, test_df = ingestion.initiate_data_ingestion()

        pipeline = PredictionPipeline()
        pipeline.predict_demand(test_df, model_type="lgb")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")