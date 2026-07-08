import sys
import os
import joblib
import yaml
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.exception import CustomException
from src.logger import logging


class ModelEvaluation:

    def __init__(self):
        """
        Initializes the ModelEvaluation class and loads the configurations from config.yaml.
        """
        try:
            with open("config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.error("Failed to load the config.yaml file.")
            raise CustomException(e, sys)

    def evaluate_model(self, X_val, y_val, models_dir=None):
        """
        Evaluates pre-trained XGBoost, LightGBM, and LSTM models using validation data.
        Generates and prints a comparative leaderboard of model metrics.
        """
        try:
            logging.info("Multi-Model evaluation started using config keys")
            
            # If explicit path is not provided, fallback to the default directory from config
            if models_dir is None:
                abs_models_dir = os.path.abspath(self.config["paths"]["models_dir"])
            else:
                abs_models_dir = os.path.abspath(models_dir)
                
            results = {}

            # 1. XGBoost Evaluation
            xgb_filename = self.config["models"]["xgb"]
            xgb_path = os.path.join(abs_models_dir, xgb_filename)
            if not os.path.exists(xgb_path):
                raise FileNotFoundError(f"Missing evaluation file: {xgb_path}")
            
            logging.info(f"Evaluating XGBoost from: {xgb_path}")
            xgb_model = joblib.load(xgb_path)
            xgb_preds = xgb_model.predict(X_val)
            results['XGBoost'] = self._get_metrics(y_val, xgb_preds)

            # 2. LightGBM Evaluation
            lgb_filename = self.config["models"]["lgb"]
            lgb_path = os.path.join(abs_models_dir, lgb_filename)
            if not os.path.exists(lgb_path):
                raise FileNotFoundError(f"Missing evaluation file: {lgb_path}")
            
            logging.info(f"Evaluating LightGBM from: {lgb_path}")
            lgb_model = joblib.load(lgb_path)
            lgb_preds = lgb_model.predict(X_val)
            results['LightGBM'] = self._get_metrics(y_val, lgb_preds)

            # 3. LSTM Evaluation
            lstm_filename = self.config["models"]["lstm"]
            lstm_path = os.path.join(abs_models_dir, lstm_filename)
            if not os.path.exists(lstm_path):
                raise FileNotFoundError(f"Missing evaluation file: {lstm_path}")
            
            logging.info(f"Evaluating LSTM from: {lstm_path}")
            lstm_model = tf.keras.models.load_model(lstm_path)
            
            # Reshape input data to 3D tensor format required for LSTM layer (samples, timesteps, features)
            X_val_lstm = np.reshape(X_val.values, (X_val.shape[0], 1, X_val.shape[1]))
            lstm_preds = lstm_model.predict(X_val_lstm).flatten()
            results['LSTM'] = self._get_metrics(y_val, lstm_preds)

            # Format and display the performance leaderboard
            print("\nModel Evaluation Leaderboard:")
          
            print(f"{'Model':<15} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
            print("-" * 55)
            for model_name, metrics in results.items():
                print(f"{model_name:<15} | {metrics['MAE']:<10.4f} | {metrics['RMSE']:<10.4f} | {metrics['R2']:<10.4f}")
            

            return results

        except Exception as e:
            raise CustomException(e, sys)

    def _get_metrics(self, y_true, y_pred):
        """
        Helper method to calculate standard regression evaluation metrics.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}
