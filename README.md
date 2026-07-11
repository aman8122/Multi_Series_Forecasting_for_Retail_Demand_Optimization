# 📈 Multi-Series Forecasting for Retail Demand Optimization

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)
![LightGBM](https://img.shields.io/badge/LightGBM-GradientBoosting-success)
![LSTM](https://img.shields.io/badge/LSTM-DeepLearning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-blue)

A **production-grade End-to-End Machine Learning pipeline** for forecasting retail demand across multiple stores and products. The project integrates advanced feature engineering, robust preprocessing, ensemble learning (XGBoost & LightGBM), deep learning (LSTM), and a real-time interactive Streamlit dashboard.

---

# 📌 Table of Contents

- Project Overview
- Dashboard
- System Architecture
- Features
- Model Performance
- Project Structure
- Installation
- Running the Project
- Machine Learning Pipeline
- Feature Engineering
- Technologies Used
- Future Improvements
- Author

---

# 🚀 Project Overview

Retail demand forecasting is one of the most important problems in supply chain management. Incorrect forecasting leads to:

- Overstocking
- Understocking
- Revenue Loss
- Inventory Imbalance

This project predicts future demand for **multiple stores and multiple products simultaneously** using advanced machine learning and deep learning techniques.

The complete pipeline includes:

- Data Ingestion
- Data Validation
- Feature Engineering
- Scaling
- Model Training
- Model Evaluation
- Prediction Pipeline
- Streamlit Dashboard

---



---

# 🏗️ System Architecture

```text
                 Raw CSV Files
                       │
                       ▼
               Data Ingestion
                       │
                       ▼
              Data Validation
                       │
                       ▼
          Data Transformation
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 Feature Engineering           Robust Scaling
 (Lag Features)                (MinMaxScaler)
        │
        ▼
     Training Dataset
        │
        ▼
 ┌──────────────────────────────────────┐
 │        Model Training Engine         │
 │--------------------------------------│
 │  • XGBoost Regressor                 │
 │  • LightGBM Regressor                │
 │  • LSTM Neural Network               │
 └──────────────────────────────────────┘
                    │
                    ▼
           Ensemble Predictions
                    │
                    ▼
           Prediction Pipeline
                    │
                    ▼
          Streamlit Dashboard
```

---

# ✨ Key Features

## 📥 Data Ingestion

- Automated dataset loading
- Config-driven pipeline
- YAML configuration support
- Modular architecture

---

## ✅ Data Validation

- Schema validation
- Duplicate removal
- Missing value handling
- Data quality checks

---

## ⚙️ Data Transformation

- Date conversion
- Encoding
- Feature mapping
- Pipeline-based preprocessing

---

## 📈 Feature Engineering

Advanced time-series features including:

- Year
- Month
- Day
- Day of Week
- Week Number
- Lag 1
- Lag 7
- Lag 30
- Rolling Mean (7 Days)
- Rolling Standard Deviation
- Previous Sales History

---

## 📊 Robust Scaling

Continuous variables are normalized using:

- MinMaxScaler

The fitted scaler is serialized into:

```
artifacts/scaler.pkl
```

This ensures identical preprocessing during inference.

---

## 🤖 Machine Learning Models

### 🚀 XGBoost

- Gradient Boosted Trees
- Handles nonlinear relationships
- Excellent forecasting capability

---

### 🚀 LightGBM

- Leaf-wise tree growth
- Fast training
- Memory efficient
- High accuracy

---

### 🧠 LSTM

Deep Learning model specialized for sequential forecasting.

Capabilities:

- Temporal learning
- Long-term dependencies
- Sequential demand prediction

---

# 🏆 Model Evaluation

Performance was evaluated using Time-Series Cross Validation.

| Model | MAE | RMSE | R² Score |
|------|------:|------:|------:|
| XGBoost | **6.2025** | **7.9996** | **0.9204** |
| LightGBM | 6.2095 | 8.0038 | 0.9203 |
| LSTM | 6.3631 | 8.2125 | 0.9161 |

---

# 📈 Performance Insights

### ✅ High Predictive Accuracy

Tree-based models achieved approximately **92% prediction accuracy**.

---

### ✅ Strong Generalization

Sequential cross-validation ensures the model captures genuine demand patterns while minimizing overfitting and preventing data leakage.

---

### ✅ Deep Temporal Learning

LSTM effectively models long-term temporal dependencies, making it suitable for forecasting sequential retail demand.

---

# 📂 Project Structure

```
Multi_Series_Forecasting_for_Retail_Demand_Optimization/
│
├── artifacts/
│
├── data/
│
├── notebook/
│
├── saved_models/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── prediction.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── app.py
├── config.yaml
├── setup.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone <repository_url>
```

Move into project directory

```bash
cd Retail_Demand_Forecasting
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

## 1️⃣ Exploratory Data Analysis

```bash
jupyter notebook
```

---

## 2️⃣ Train Pipeline

```bash
python -m src.pipeline.train_pipeline
```

Pipeline Execution Flow:

```
Data Ingestion

↓

Data Validation

↓

Data Transformation

↓

Feature Engineering

↓

Scaling

↓

Model Training

↓

Model Evaluation

↓

Model Saving
```

---

## 3️⃣ Prediction Pipeline

```bash
python -m src.pipeline.prediction_pipeline
```

This pipeline:

- Loads trained model
- Loads scaler
- Generates lag features
- Makes predictions
- Saves outputs

---

## 4️⃣ Launch Streamlit Dashboard

```bash
streamlit run app.py
```

---

# ⚡ Machine Learning Workflow

```
Raw Data

↓

Cleaning

↓

Validation

↓

Feature Engineering

↓

Scaling

↓

Train/Test Split

↓

Model Training

↓

Hyperparameter Optimization

↓

Evaluation

↓

Prediction

↓

Dashboard Visualization
```

---

# 🛠️ Technologies Used

### Programming

- Python

### Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM

### Deep Learning

- TensorFlow
- Keras (LSTM)

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn
- Streamlit

### Model Serialization

- Joblib
- Pickle

---


---

# 👨‍💻 Author

**Aman Sain**

B.Tech Computer Science Engineering

Machine Learning | Data Science | Time Series Forecasting | Deep Learning

---

