import sys
import os
import joblib
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.exception import CustomException
from src.logger import logging


class ModelTrainer:

    def __init__(self):
        try:
            # load config
            with open("config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.error("Failed to load the config.yaml file.")
            raise CustomException(e, sys)

        # set path
        self.models_dir = os.path.abspath(self.config["paths"]["models_dir"])
        os.makedirs(self.models_dir, exist_ok=True)

    def build_lstm(self, input_shape):
        # build lstm
        model=Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam',loss='mse')
        return model

    def initiate_model_trainer(self, X_train, y_train):
        try:
            logging.info(f"Multi-Model training initiated. Target directory: {self.models_dir}")

            # train xgb
            logging.info("Training XGBoost Regressor...")
            xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            
            # save xgb
            xgb_path = os.path.join(self.models_dir, self.config["models"]["xgb"])
            joblib.dump(xgb_model, xgb_path)
            logging.info(f"XGBoost saved successfully at {xgb_path}")

            # train lgb
            logging.info("Training LightGBM Regressor...")
            lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
            lgb_model.fit(X_train, y_train)
            
            # save lgb
            lgb_path = os.path.join(self.models_dir, self.config["models"]["lgb"])
            joblib.dump(lgb_model, lgb_path)
            logging.info(f"LightGBM saved successfully at {lgb_path}")

            # reshape data
            logging.info("Preparing data for LSTM (Reshaping to 3D)...")
            X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
            
            # train lstm
            lstm_model = self.build_lstm(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
            logging.info("Training LSTM Model...")
            
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
            lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=512, callbacks=[early_stop], verbose=1)
            
            # save lstm
            lstm_path = os.path.join(self.models_dir, self.config["models"]["lstm"])
            lstm_model.save(lstm_path)
            logging.info(f"LSTM saved successfully at {lstm_path}")

           
            return self.models_dir

        except Exception as e:
            logging.error("An error occurred while training the models.")
            raise CustomException(e, sys)