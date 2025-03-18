import streamlit as st
import pandas as pd
import pickle
import os

# Load the model and preprocessor
model_path = r'C:\Users\ANANDHU\OneDrive\Desktop\Documents\capstone_project_BIA\best_model.pkl'
preprocessor_path = r'C:\Users\ANANDHU\OneDrive\Desktop\Documents\capstone_project_BIA\preproccesor.pkl'
output_file = r'C:\Users\ANANDHU\OneDrive\Desktop\Documents\capstone_project_BIA\New_Data.csv'

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(preprocessor_path, 'rb') as f:
    pre = pickle.load(f)

# Add custom CSS for styling
page_element = """
<style>
[data-testid="stAppViewContainer"] {
  background-image: url("https://cdn.wallpapersafari.com/88/75/cLUQqJ.jpg");
  background-size: cover;
}
[data-testid="stHeader"] {
  background-color: rgba(0,0,0,0);
}
.success-message {
  color: white;
  background-color: #4CAF50;
  padding: 10px;
  border-radius: 5px;
  font-size: 20px;
  text-align: center;
}
.error-message {
  color: white;
  background-color: #FF6347;
  padding: 10px;
  border-radius: 5px;
  font-size: 20px;
  text-align: center;
}
</style>
"""
st.markdown(page_element, unsafe_allow_html=True)

# App title and header
st.title("Heart Attack Prediction in Youth 🫀")
st.header("Enter or Paste Patient Information")

# Auto-fill option
auto_fill_data = st.text_input(
    "Paste data (comma-separated):",
    placeholder="Systolic_BP,Triglyceride Levels,Diastolic_BP,BMI,Cholesterol,Blood Oxygen,Max Heart Rate,Resting Heart Rate,Age,Screen Time",
)

# Split data and pre-fill fields if input is provided
if auto_fill_data:
    try:
        auto_data = list(map(float, auto_fill_data.split(',')))
        (
            systolic_bp,
            triglyceride_levels,
            diastolic_bp,
            bmi,
            cholesterol_levels,
            blood_oxygen,
            max_heart_rate,
            resting_heart_rate,
            age,
            screen_time,
        ) = auto_data
    except ValueError:
        st.error("Invalid input format. Please ensure all values are numeric and comma-separated.")
else:
    # Default values for manual input
    age = 30
    screen_time = 2.0
    bmi = 25.0
    resting_heart_rate = 70
    max_heart_rate = 150
    systolic_bp = 120
    diastolic_bp = 80
    cholesterol_levels = 200
    blood_oxygen = 98
    triglyceride_levels = 150

# User input fields
age = st.number_input("Age", min_value=0, max_value=120, value=int(age))
screen_time = st.number_input("Screen Time (hrs/day)", min_value=0.0, max_value=24.0, value=screen_time, step=0.1)
bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=bmi, step=0.1)
resting_heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=30, max_value=130, value=int(resting_heart_rate))
max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=int(max_heart_rate))
systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=50, max_value=250, value=int(systolic_bp))
diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=30, max_value=150, value=int(diastolic_bp))
cholesterol_levels = st.number_input("Cholesterol Levels (mg/dL)", min_value=100, max_value=600, value=int(cholesterol_levels))
blood_oxygen = st.number_input("Blood Oxygen Levels (SpO2%)", min_value=50, max_value=100, value=int(blood_oxygen))
triglyceride_levels = st.number_input("Triglyceride Levels (mg/dL)", min_value=50, max_value=1000, value=int(triglyceride_levels))

# Button for prediction
submit = st.button('Submit')

if submit:
    # Prepare the input data
    data = {
        'Age': [age],
        'Screen Time (hrs/day)': [screen_time],
        'BMI (kg/m²)': [bmi],
        'Resting Heart Rate (bpm)': [resting_heart_rate],
        'Maximum Heart Rate Achieved': [max_heart_rate],
        'Systolic_BP': [systolic_bp],
        'Diastolic_BP': [diastolic_bp],
        'Cholesterol Levels (mg/dL)': [cholesterol_levels],
        'Blood Oxygen Levels (SpO2%)': [blood_oxygen],
        'Triglyceride Levels (mg/dL)': [triglyceride_levels],
    }
    data_df = pd.DataFrame(data)

    try:
        # Apply preprocessing and make predictions
        data_processed = pre.transform(data_df)
        prediction = model.predict(data_processed)

        # Add prediction result to the DataFrame
        data_df['Heart Attack Chance'] = ['Yes' if prediction[0] == 1 else 'No']

        # Save the input data and prediction to a CSV file
        if os.path.exists(output_file):
            data_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            data_df.to_csv(output_file, index=False)

        # Display result
        if prediction[0] == 0:
            st.markdown(
                '<div class="success-message">The patient is at low risk for a heart attack.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="error-message">The patient is at high risk for a heart attack.</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Display images after the output
    st.subheader("Reference Images for Key Metrics")

    # Systolic Blood Pressure
    st.image(r'c:\Users\ANANDHU\Downloads\systolic.png', caption='Systolic Blood Pressure Ranges', use_column_width=True)

    # Diastolic Blood Pressure
    st.image(r'c:\Users\ANANDHU\Downloads\diatolic.png', caption='Diastolic Blood Pressure Ranges', use_column_width=True)

    # Maximum Heart Rate
    st.image(r'c:\Users\ANANDHU\Downloads\maxheartrate.png', caption='Maximum Heart Rate', use_column_width=True)

    # Cholesterol Levels
    st.image(r'c:\Users\ANANDHU\Downloads\cholesterol-levels.png', caption='Cholesterol Levels', use_column_width=True)

    # Triglyceride Levels
    st.image(r'c:\Users\ANANDHU\Downloads\triglyceridelevels.png', caption='Triglyceride Levels', use_column_width=True)

    # BMI Ranges
    st.image(r'c:\Users\ANANDHU\Downloads\bmi.png', caption='BMI Ranges', use_column_width=True)
