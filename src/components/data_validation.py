import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging


class DataValidation:

    def __init__(self):
        logging.info("Data Validation component initialized.")

    def validate_and_clean(self, data_frame, is_train=True):
        try:
            logging.info(f"Data validation/cleaning started.")
            
            # copy data
            cleaned_df = data_frame.copy()

            # fix datetime
            cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
            
            # drop duplicates
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=['date', 'store', 'item'])
            if len(cleaned_df) < initial_count:
                logging.warning(f"Dropped {initial_count - len(cleaned_df)} duplicate records.")

            # clean train
            if is_train and 'sales' in cleaned_df.columns:
                cleaned_df = cleaned_df.dropna(subset=['sales'])
                cleaned_df['sales'] = cleaned_df['sales'].clip(lower=0)

                # handle outliers
                q25_val = cleaned_df['sales'].quantile(0.25)
                q75_val = cleaned_df['sales'].quantile(0.75)
                iqr_range = q75_val - q25_val
                upper_bound_limit = q75_val + (1.5 * iqr_range)
                
                cleaned_df['sales'] = cleaned_df['sales'].clip(upper=upper_bound_limit)
                logging.info(f"Outlier treatment complete. Cap threshold limit: {upper_bound_limit:.2f}")

            return cleaned_df
            
        except Exception as e:
            raise CustomException(e, sys)