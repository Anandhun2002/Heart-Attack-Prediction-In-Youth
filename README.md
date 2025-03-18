# Heart Attack Prediction 

## Overview
This project is a **Heart Attack Prediction App** that predicts the likelihood of a heart attack based on various health parameters such as **age, BMI, blood pressure, cholesterol, blood oxygen levels, and heart rate metrics**.

The app is built using **Streamlit**, providing a simple web interface for real-time health risk analysis.

## Features
- **Machine Learning Model**: Uses a pre-trained predictive model.
- **User-Friendly Interface**: Built with **Streamlit** for ease of use.
- **Real-Time Prediction**: Users can enter health data and instantly receive a prediction on heart attack risk.
- **Data Storage**: Inputs and predictions are saved to a CSV file for future reference.

## Files
- **`app2.py`** - Streamlit app that loads the trained heart attack prediction model, collects user input, and predicts the risk of a heart attack.
- **`heartattack_Eda.ipynb`** - Jupyter Notebook containing exploratory data analysis (EDA) on heart attack-related factors.
- **`heartattack_model.ipynb`** - Jupyter Notebook containing the model training and preprocessing workflow.
- **`best_model.pkl`** - Pre-trained machine learning model used for predictions.
- **`preprocessor.pkl`** - Preprocessing pipeline used to transform input data before feeding it into the model.


## Model & Data Processing
- The model was trained using a dataset of health metrics related to heart attack risk.
- Preprocessing steps include feature scaling, handling missing values, and data transformation.
- The model classifies input data as **high risk or low risk** for a heart attack.


