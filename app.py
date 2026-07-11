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

# 1. Sidebar - Model Details & Configuration
st.sidebar.header("🧠 Engine Configuration")

# Model drop-down option for users to select the best model
model_choice = st.sidebar.selectbox(
    "Choose Forecasting Model",
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

# Tab Selection - Allows user to choose between Form Input or CSV Upload
st.markdown("### Select Input Method")
tab1, tab2 = st.tabs(["🎯 Single Item Prediction Form", "📁 Bulk CSV Batch File Upload"])

# Variable to hold data for prediction engine
input_df = None
is_single_input = False

# --- TAB 1: SINGLE ITEM MANUAL FORM ---
with tab1:
    st.markdown("#### Enter Parameters Manually")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_date = st.date_input("📅 Target Prediction Date")
    with col2:
        selected_store = st.number_input("🏪 Store ID", min_value=1, max_value=50, value=1, step=1)
    with col3:
        selected_item = st.number_input("📦 Item ID", min_value=1, max_value=100, value=1, step=1)
        
    if st.button("🚀 Run Single Forecast"):
        # Building an automated runtime DataFrame mimicking pipeline entry structure
        input_df = pd.DataFrame([{
            'date': str(selected_date),
            'store': int(selected_store),
            'item': int(selected_item)
        }])
        is_single_input = True

# --- TAB 2: BULK CSV BATCH FILE UPLOAD ---
with tab2:
    uploaded_file = st.file_uploader("Upload Testing CSV Data", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("👀 Raw Uploaded Data Preview")
        st.dataframe(input_df.head(), width='stretch')

# --- MAIN FORECASTING PROCESSING PIPELINE ENGINE ---
if input_df is not None:
    try:
        if os.path.exists(processed_train_path) and os.path.exists(model_path):
            with st.spinner(f"Processing structural maps and predicting using {model_choice}..."):
                
                logging.info(f"Forecasting cycle triggered via Streamlit UI. Model: {model_choice}")
                
                # Load processed training context for secure sequence-wise lag calculation
                train_context = pd.read_csv(processed_train_path)
                logging.info("Processed training context data loaded successfully for lag context")
                
                # Transform features using new decoupled steps (Validation -> FeatureEngineering)
                validator = DataValidation()
                clean_train_context = validator.validate_and_clean(train_context, is_train=True)
                clean_raw_test = validator.validate_and_clean(input_df, is_train=False)

                engineer = FeatureEngineering()
                _, final_test = engineer.extract_features(clean_train_context, clean_raw_test)
                
                # Ensure runtime single tracking indices aren't lost
                if final_test.empty:
                    st.error("Context Boundary Matrix generation failed for this specific entity date combination!")
                else:
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
                    
                    # STRICT FEATURE ALIGNMENT BLOCK
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
                    output_df = input_df.copy()
                    
                    # FIXED FOR INTEGER ROUNDING: Float hataya aur pure clean integers me cast kiya
                    output_df['Predicted_Sales'] = np.round(predictions).astype(int)
                    
                    st.markdown("---")
                    st.subheader(f" Forecast Results ({model_choice})")
                    
                    if is_single_input:
                        # Displaying as clean, optimized metrics box if it's a form input
                        predicted_value = output_df['Predicted_Sales'].values[0]
                        st.metric(label=f"Predicted Retail Demand Quantity", value=f"{predicted_value} Units")
                    
                    # Display Full Detailed Grid Matrix anyway
                    st.dataframe(output_df, width='stretch')
                    
                    # Download button for predictions matrix
                    csv = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇ Download Predictions CSV",
                        data=csv,
                        file_name=f"demand_predictions_{model_choice.lower()}.csv",
                        mime="text/csv"
                    )
                    logging.info("Forecast dataframe displayed and download link generated")
                    
        else:
            st.warning(f" Processed context metadata missing! Make sure '{processed_train_path}' and '{model_path}' exist.")
            logging.warning("Prediction attempted but context target storage points are missing")

    except Exception as e:
        logging.error("An error occurred in the Streamlit App interface execution")
        st.error(f"Something went wrong! Check your logs. Error: {str(e)}")
        raise CustomException(e, sys)

else:
    st.info("💡 Choose a parameter from the form tab or upload a CSV file from the bulk tab to see demand predictions.")