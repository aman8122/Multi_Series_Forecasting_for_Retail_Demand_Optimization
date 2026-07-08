import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

class DataValidation:
    def __init__(self):
        logging.info("Data Validation component initialized.")

    def validate_and_clean(self, data_frame, is_train=True):
        try:
            logging.info(f"Data validation/cleaning started. Mode: {'Train' if is_train else 'Test'}")
            working_df = data_frame.copy()

            # Date parsed into datetime object
            working_df['date'] = pd.to_datetime(working_df['date'])
            
            # Check for duplicates across primary time keys
            rows_before = len(working_df)
            working_df = working_df.drop_duplicates(subset=['date', 'store', 'item'])
            if len(working_df) < rows_before:
                logging.warning(f"Dropped {rows_before - len(working_df)} duplicate records.")

            # Target validation only applied to training data
            if is_train and 'sales' in working_df.columns:
                working_df = working_df.dropna(subset=['sales'])
                working_df['sales'] = working_df['sales'].clip(lower=0)

                # Outlier boundary capping using IQR Method
                lower_q = working_df['sales'].quantile(0.25)
                upper_q = working_df['sales'].quantile(0.75)
                iqr_val = upper_q - lower_q
                max_cutoff = upper_q + (1.5 * iqr_val)
                
                working_df['sales'] = working_df['sales'].clip(upper=max_cutoff)
                logging.info(f"Outlier treatment complete. Cap threshold limit: {max_cutoff:.2f}")

            return working_df
        except Exception as e:
            raise CustomException(e, sys)