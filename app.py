import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import yaml
import joblib
import tensorflow as tf
from src.logger import logging
from src.exception import CustomException

# Importing the new decentralized modular components
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering

# Page Configuration
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

# Configuration Load Kar Rahe Hain
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    st.error("❌ config.yaml file load nahi ho payi!")
    logging.error("config.yaml file load karne mein dikkat aayi Streamlit App mein.")
    raise CustomException(e, sys)

# Config se paths extract karna
models_dir = os.path.abspath(config["paths"]["models_dir"])
processed_test_path = os.path.abspath(config["paths"]["processed_test_path"])
processed_dir = os.path.dirname(processed_test_path)
processed_train_path = os.path.join(processed_dir, "train_processed.csv")

st.title("📈 Multi-Series Retail Demand Forecasting App")
st.markdown("---")

# 1. Sidebar - Model Details & File Uploads
st.sidebar.header("📁 Data & Configuration")

# Model drop-down option for users to select the best model
model_choice = st.sidebar.selectbox(
    "🧠 Choose Forecasting Model",
    options=["LightGBM", "XGBoost", "LSTM"]
)

# Map choices dynamically using config keys
model_mapping = {
    "XGBoost": os.path.join(models_dir, config["models"]["xgb"]),
    "LightGBM": os.path.join(models_dir, config["models"]["lgb"]),
    "LSTM": os.path.join(models_dir, config["models"]["lstm"])
}
model_path = model_mapping[model_choice]

# Check if the selected model exists
if os.path.exists(model_path):
    st.sidebar.success(f"✅ {model_choice} Model Found!")
else:
    st.sidebar.error(f"❌ {model_choice} file missing! Please run train_pipeline.py first.")

# File Uploader for Test Data
uploaded_file = st.sidebar.file_uploader("Upload Testing CSV Data", type=["csv"])

# 2. Main Dashboard Logic
if uploaded_file is not None:
    try:
        logging.info("New test file uploaded via Streamlit UI")
        # Read Uploaded Data
        test_df = pd.read_csv(uploaded_file)
        
        st.subheader("👀 Raw Uploaded Data Preview")
        # FIXED: Replaced use_container_width=True with width='stretch'
        st.dataframe(test_df.head(), width='stretch') 

        if os.path.exists(processed_train_path) and os.path.exists(model_path):
            if st.button("🚀 Generate Demand Forecast"):
                with st.spinner(f"Processing features and predicting using {model_choice}..."):
                    
                    logging.info(f"Demand Forecast button clicked. Model: {model_choice}")
                    
                    # Load processed training context for secure sequence-wise lag calculation
                    train_context = pd.read_csv(processed_train_path)
                    logging.info("Processed training context data loaded successfully for lag context")
                    
                    # Transform features using new decoupled steps (Validation -> FeatureEngineering)
                    validator = DataValidation()
                    clean_train_context = validator.validate_and_clean(train_context, is_train=True)
                    clean_raw_test = validator.validate_and_clean(test_df, is_train=False)

                    engineer = FeatureEngineering()
                    _, final_test = engineer.extract_features(clean_train_context, clean_raw_test)
                    
                    # Load Scaler object
                    scaler_path = os.path.join(models_dir, "scaler.pkl")
                    if os.path.exists(scaler_path):
                        scaler = joblib.load(scaler_path)
                        logging.info("Scaler loaded successfully for UI inference scaling")
                    else:
                        raise FileNotFoundError(f"Scaler pkl file missing at {scaler_path}")
                    
                    # Base features context (without target and timestamp keys)
                    base_features = [col for col in final_test.columns if col not in ["sales", "date", "id"]]
                    
                    # Scale dataset features using production safe transform mode
                    final_test[base_features] = scaler.transform(final_test[base_features])
                    
                    # 🔄 STRICT FEATURE ALIGNMENT BLOCK
                    if model_choice != "LSTM":
                        model = joblib.load(model_path)
                        
                        if hasattr(model, "feature_names_in_"):
                            feature_columns = list(model.feature_names_in_)
                        elif hasattr(model, "feature_name_"):
                            feature_columns = model.feature_name_()
                        else:
                            feature_columns = base_features
                        
                        logging.info(f"Model expected features: {feature_columns}")
                        
                        X_test = final_test[feature_columns]
                        predictions = model.predict(X_test)
                    
                    else:
                        # LSTM Inference Flow
                        logging.info("Loading Deep Learning Model: LSTM")
                        model = tf.keras.models.load_model(model_path)
                        
                        X_test = final_test[base_features]
                        # 3D Tensor conversion for Deep Learning input layer
                        X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))
                        predictions = model.predict(X_test_lstm).flatten()
                        
                    logging.info("Predictions generated successfully with strict columns mapping")
                    
                    # Map predictions back to visible dataframe
                    output_df = test_df.copy()
                    output_df['Predicted_Sales'] = np.round(predictions, 2)
                    
                    st.markdown("---")
                    st.subheader(f"📊 Forecast Results ({model_choice})")
                    # FIXED: Replaced use_container_width=True with width='stretch'
                    st.dataframe(output_df, width='stretch')
                    
                    # Download button for predictions
                    csv = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"demand_predictions_{model_choice.lower()}.csv",
                        mime="text/csv"
                    )
                    logging.info("Forecast dataframe displayed and download link generated")
                    
        else:
            st.warning(f"⚠️ Processed context metadata missing! Make sure '{processed_train_path}' and '{model_path}' exist.")
            logging.warning("Prediction attempted but context target storage points are missing")

    except Exception as e:
        logging.error("An error occurred in the Streamlit App interface execution")
        st.error(f"Something went wrong! Check your logs. Error: {str(e)}")
        raise CustomException(e, sys)

else:
    st.info("💡 Please upload a testing CSV file from the sidebar to start forecasting.")