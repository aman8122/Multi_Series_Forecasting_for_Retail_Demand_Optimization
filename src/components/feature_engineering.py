import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging


class FeatureEngineering:

    def __init__(self):
        logging.info("Feature Engineering component init.")

    def extract_features(self, cleaned_train_df, cleaned_test_df):
        try:
            logging.info("Time-series feature extraction initialized.")
            
            # copy data
            raw_train_set = cleaned_train_df.copy()
            raw_test_set = cleaned_test_df.copy()

            if 'sales' not in raw_test_set.columns:
                raw_test_set['sales'] = np.nan

            raw_train_set['is_train'] = 1
            raw_test_set['is_train'] = 0

            # merge data
            unified_matrix = pd.concat([raw_train_set, raw_test_set], ignore_index=True)

            # date features
            unified_matrix['year'] = unified_matrix['date'].dt.year
            unified_matrix['month'] = unified_matrix['date'].dt.month
            unified_matrix['day'] = unified_matrix['date'].dt.day
            unified_matrix['dayofweek'] = unified_matrix['date'].dt.dayofweek

            # sort data
            unified_matrix = unified_matrix.sort_values(['store', 'item', 'date']).reset_index(drop=True)

            # lag features
            unified_matrix['lag_1'] = unified_matrix.groupby(['store', 'item'])['sales'].shift(1)
            unified_matrix['lag_7'] = unified_matrix.groupby(['store', 'item'])['sales'].shift(7)
            unified_matrix['lag_30'] = unified_matrix.groupby(['store', 'item'])['sales'].shift(30)
            
            # rolling feature
            unified_matrix['rolling_mean_7'] = (
                unified_matrix.groupby(['store', 'item'])['sales']
                .shift(1)
                .rolling(7)
                .mean()
            )

            # fill missing
            temporal_series_keys = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7']
            for active_metric_key in temporal_series_keys:
                unified_matrix[active_metric_key] = unified_matrix.groupby(['store', 'item'])[active_metric_key].ffill()

            # split data
            processed_train = unified_matrix[unified_matrix['is_train'] == 1].drop(columns=['is_train']).copy()
            processed_test = unified_matrix[unified_matrix['is_train'] == 0].drop(columns=['is_train']).copy()

            # drop na
            processed_train = processed_train.dropna(subset=temporal_series_keys)

            if 'sales' in processed_test.columns:
                processed_test = processed_test.drop(columns=['sales'])

            # clean ids
            if 'id' in processed_train.columns:
                processed_train = processed_train.drop(columns=['id'])
            if 'id' in processed_test.columns:
                processed_test = processed_test.drop(columns=['id'])

            logging.info("Feature engineering processed successfully.")
            return processed_train, processed_test

        except Exception as e:
            raise CustomException(e, sys)