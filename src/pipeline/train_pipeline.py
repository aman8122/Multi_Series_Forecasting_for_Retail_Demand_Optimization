import os
import sys
import yaml

# root path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.exception import CustomException
from src.logger import logging
import warnings
warnings.filterwarnings('ignore')

# imports
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


class TrainPipeline:

    def __init__(self):
        try:
            # load config
            with open("config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.error("Failed to load the config.yaml file.")
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("=== TRAIN PIPELINE STARTED ===")

            # data ingestion
            ingestion = DataIngestion()
            train_df, test_df = ingestion.initiate_data_ingestion()

            # data validation
            validator = DataValidation()
            clean_train_df = validator.validate_and_clean(train_df, is_train=True)
            clean_test_df = validator.validate_and_clean(test_df, is_train=False)

            # feature engineering
            engineer = FeatureEngineering()
            engineered_train, engineered_test = engineer.extract_features(clean_train_df, clean_test_df)

            # save processed
            processed_test_path = os.path.abspath(self.config["paths"]["processed_test_path"])
            processed_dir = os.path.dirname(processed_test_path)
            os.makedirs(processed_dir, exist_ok=True)
            
            processed_train_path = os.path.join(processed_dir, "train_processed.csv")
            
            logging.info(f"Saving unscaled processed features data to {processed_dir}")
            engineered_train.to_csv(processed_train_path, index=False)
            engineered_test.to_csv(processed_test_path, index=False)
            
            print(f" SUCCESS: Processed datasets saved at {processed_dir}!")

            # get columns
            feature_columns = [col for col in engineered_train.columns if col not in ["sales", "date"]]

            # data transformation
            transformer = DataTransformation()
            transformed_train, transformed_test, scaler_obj = transformer.scale_features(
                engineered_train, engineered_test, feature_columns
            )

            # split data
            split = int(len(transformed_train) * 0.8)
            train_set = transformed_train.iloc[:split]
            X_val = transformed_train[feature_columns].iloc[split:]
            y_val = transformed_train["sales"].iloc[split:]

            # init tscv
            print(" RUNNING TIME SERIES CROSS-VALIDATION FOR ALL MODELS...")
            tscv = TimeSeriesSplit(n_splits=5)
            X_cv = transformed_train[feature_columns].values
            y_cv = transformed_train["sales"].values

            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor

            xgb_cv_model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
            lgb_cv_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)

            # run cv
            xgb_scores, lgb_scores = [], []
            for fold, (train_index, val_index) in enumerate(tscv.split(X_cv)):
                X_tr, X_va = X_cv[train_index], X_cv[val_index]
                y_tr, y_va = y_cv[train_index], y_cv[val_index]
                
                xgb_cv_model.fit(X_tr, y_tr)
                xgb_score = xgb_cv_model.score(X_va, y_va)
                xgb_scores.append(xgb_score)

                lgb_cv_model.fit(X_tr, y_tr)
                lgb_score = lgb_cv_model.score(X_va, y_va)
                lgb_scores.append(lgb_score)
                
                print(f" Fold {fold+1} | XGBoost R2: {xgb_score:.4f} | LightGBM R2: {lgb_score:.4f}")
            
            print("-" * 55)
            print(f"Mean XGBoost R2 Score       : {np.mean(xgb_scores):.4f}")
            print(f" Mean LightGBM R2 Score      : {np.mean(lgb_scores):.4f}")

            # train model
            logging.info("Training final combination of models on 80% train set split")
            trainer = ModelTrainer()
            saved_models_dir = trainer.initiate_model_trainer(train_set[feature_columns], train_set["sales"])

            # eval model
            evaluator = ModelEvaluation()
            evaluator.evaluate_model(X_val, y_val, models_dir=saved_models_dir)

            print("WHOLE MODULAR TRAINING PIPELINE EXECUTED SUCCESSFULLY WITH ADVANCED ENSEMBLE MODELS!")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()