import pickle
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Load trained model
loaded_model = pickle.load(open('Trained_model.pkl','rb')) 

def main():
    st.set_page_config(page_title="Heart Disease Prediction App")
    st.title("Heart Disease Prediction App")

    # image
    img = Image.open("HEART.png")
    st.image(
        img, width=700 ,channels='RGB'
        )

    with st.form("HDP"):
        st.subheader("Enter Patient Details Below")

         # Input variables
        age = st.number_input('Age', min_value=1, max_value=100, value=30)
        sex = st.text_input('Gender (Male:1, Female:0)')
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
        chol = st.number_input('Serum Cholesterol', min_value=50, max_value=600, value=200)
        fbs = st.selectbox('Fasting Blood Sugar (1=Yes, 0=No)', [0, 1])
        restecg = st.selectbox('Resting ECG', [0, 1, 2])
        thalach = st.number_input('Maximum Heart Rate', min_value=60, max_value=250, value=150)
        exang = st.selectbox('Exercise-Induced Angina (1=Yes, 0=No)', [0, 1])
        oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
        ca = st.selectbox('Number of Major Vessels', [0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia Type', [0, 1, 2, 3])

        # Submit button inside the form
        submitted = st.form_submit_button("Predict")

    # Prediction on form submission
    if submitted:
        input_data = pd.DataFrame([[float(age), float(sex), float(cp), float(trestbps),
                                    float(chol), float(fbs), float(restecg), float(thalach),
                                    float(exang), float(oldpeak), float(slope), float(ca), float(thal)]],
                                    columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
                                             'exang','oldpeak','slope','ca','thal'])
        
        # Make prediction
        prediction = int(loaded_model.predict(input_data)[0])

        # Display result
        if prediction == 1:
            st.success("Good News! Patient doesn't have heart disease")
        else:
            st.error("Oh! Patient should visit the doctor")

if __name__ == "__main__":
    main()