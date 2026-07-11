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
        try:
            # load config
            with open("config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.error("Failed to load the config.yaml file.")
            raise CustomException(e, sys)

    def evaluate_model(self, X_val, y_val, models_dir=None):
        try:
            logging.info("Multi-Model evaluation started using config keys")
            
            # set path
            if models_dir is None:
                target_models_dir = os.path.abspath(self.config["paths"]["models_dir"])
            else:
                target_models_dir = os.path.abspath(models_dir)
                
            performance_map = {}

            # eval xgb
            xgb_filename = self.config["models"]["xgb"]
            xgb_locator_path = os.path.join(target_models_dir, xgb_filename)
            if not os.path.exists(xgb_locator_path):
                raise FileNotFoundError(f"Missing evaluation file: {xgb_locator_path}")
            
            logging.info(f"Evaluating XGBoost from: {xgb_locator_path}")
            xgb_estimator = joblib.load(xgb_locator_path)
            xgb_predictions = xgb_estimator.predict(X_val)
            performance_map['XGBoost'] = self._get_metrics(y_val, xgb_predictions)

            # eval lgb
            lgb_filename = self.config["models"]["lgb"]
            lgb_locator_path = os.path.join(target_models_dir, lgb_filename)
            if not os.path.exists(lgb_locator_path):
                raise FileNotFoundError(f"Missing evaluation file: {lgb_locator_path}")
            
            logging.info(f"Evaluating LightGBM from: {lgb_locator_path}")
            lgb_estimator = joblib.load(lgb_locator_path)
            lgb_predictions = lgb_estimator.predict(X_val)
            performance_map['LightGBM'] = self._get_metrics(y_val, lgb_predictions)

            # eval lstm
            lstm_filename = self.config["models"]["lstm"]
            lstm_locator_path = os.path.join(target_models_dir, lstm_filename)
            if not os.path.exists(lstm_locator_path):
                raise FileNotFoundError(f"Missing evaluation file: {lstm_locator_path}")
            
            logging.info(f"Evaluating LSTM from: {lstm_locator_path}")
            lstm_network = tf.keras.models.load_model(lstm_locator_path)
            
            # reshape tensor
            X_val_tensor = np.reshape(X_val.values, (X_val.shape[0], 1, X_val.shape[1]))
            lstm_predictions = lstm_network.predict(X_val_tensor).flatten()
            performance_map['LSTM'] = self._get_metrics(y_val, lstm_predictions)

            # show leaderboard
            print("\nModel Evaluation Leaderboard:")
            print(f"{'Model':<15} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
            print("-" * 55)
            for structural_key, accuracy_metrics in performance_map.items():
                print(f"{structural_key:<15} | {accuracy_metrics['MAE']:<10.4f} | {accuracy_metrics['RMSE']:<10.4f} | {accuracy_metrics['R2']:<10.4f}")

            return performance_map

        except Exception as e:
            raise CustomException(e, sys)

    def _get_metrics(self, y_true, y_pred):
        # calc metrics
        absolute_error_score = mean_absolute_error(y_true, y_pred)
        root_mse_score = np.sqrt(mean_squared_error(y_true, y_pred))
        r2_fit_score = r2_score(y_true, y_pred)
        return {"MAE": absolute_error_score, "RMSE": root_mse_score, "R2": r2_fit_score}