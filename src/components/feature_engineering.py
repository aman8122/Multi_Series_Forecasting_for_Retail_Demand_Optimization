import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

class FeatureEngineering:
    def __init__(self):
        logging.info("Feature Engineering component initialized.")

    def extract_features(self, cleaned_train_df, cleaned_test_df):
        try:
            logging.info("Time-series feature extraction initialized.")
            
            train_df = cleaned_train_df.copy()
            test_df = cleaned_test_df.copy()

            if 'sales' not in test_df.columns:
                test_df['sales'] = np.nan

            train_df['is_train'] = 1
            test_df['is_train'] = 0

            # Temporary chronological merge for gapless sequential calculation
            merged_dataset = pd.concat([train_df, test_df], ignore_index=True)

            # Date features extraction
            merged_dataset['year'] = merged_dataset['date'].dt.year
            merged_dataset['month'] = merged_dataset['date'].dt.month
            merged_dataset['day'] = merged_dataset['date'].dt.day
            merged_dataset['dayofweek'] = merged_dataset['date'].dt.dayofweek

            # Mandatory sorting before lagging
            merged_dataset = merged_dataset.sort_values(['store', 'item', 'date']).reset_index(drop=True)

            # Lag features calculation
            merged_dataset['lag_1'] = merged_dataset.groupby(['store', 'item'])['sales'].shift(1)
            merged_dataset['lag_7'] = merged_dataset.groupby(['store', 'item'])['sales'].shift(7)
            merged_dataset['lag_30'] = merged_dataset.groupby(['store', 'item'])['sales'].shift(30)
            
            # Leakage-proof rolling window configuration
            merged_dataset['rolling_mean_7'] = (
                merged_dataset.groupby(['store', 'item'])['sales']
                .shift(1)
                .rolling(7)
                .mean()
            )

            # Forward-fill gaps securely (Strictly chronological history propagation)
            lag_columns = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7']
            for current_col in lag_columns:
                merged_dataset[current_col] = merged_dataset.groupby(['store', 'item'])[current_col].ffill()

            # Reseparating target components safely
            processed_train = merged_dataset[merged_dataset['is_train'] == 1].drop(columns=['is_train']).copy()
            processed_test = merged_dataset[merged_dataset['is_train'] == 0].drop(columns=['is_train']).copy()

            # Safe cleanup of initial index NaNs on training data
            processed_train = processed_train.dropna(subset=lag_columns)

            if 'sales' in processed_test.columns:
                processed_test = processed_test.drop(columns=['sales'])

            # Align feature index structure by dropping structural 'id' keys
            if 'id' in processed_train.columns:
                processed_train = processed_train.drop(columns=['id'])
            if 'id' in processed_test.columns:
                processed_test = processed_test.drop(columns=['id'])

            logging.info("Feature engineering processed successfully without look-ahead bias.")
            return processed_train, processed_test

        except Exception as e:
            raise CustomException(e, sys)