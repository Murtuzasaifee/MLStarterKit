import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():
    # Streamlit app setup
    st.set_page_config(page_title="Student Score Prediction App", layout="centered")

    # Title of the app
    st.title("Prediction App")

    # Sidebar for navigation
    st.sidebar.header("User Input Features")
    st.sidebar.write("Fill out the form to make a prediction.")

    # Mappings for user-friendly display and backend-ready values
    gender_mapping = {"Male": "male", "Female": "female"}
    race_mapping = {
        "Group A": "group A",
        "Group B": "group B",
        "Group C": "group C",
        "Group D": "group D",
        "Group E": "group E",
    }
    parental_education_mapping = {
        "Some college": "some college",
        "Associate's degree": "associate's degree",
        "Bachelor's degree": "bachelor's degree",
        "Master's degree": "master's degree",
        "High school": "high school",
    }
    lunch_mapping = {"Standard": "standard", "Free/reduced": "free/reduced"}
    test_prep_mapping = {"None": "none", "Completed": "completed"}

    # Collect user input
    gender_display = st.sidebar.selectbox("Gender", options=list(gender_mapping.keys()))
    race_ethnicity_display = st.sidebar.selectbox(
        "Race/Ethnicity", options=list(race_mapping.keys())
    )
    parental_level_of_education_display = st.sidebar.selectbox(
        "Parental Level of Education", options=list(parental_education_mapping.keys())
    )
    lunch_display = st.sidebar.selectbox("Lunch", options=list(lunch_mapping.keys()))
    test_preparation_course_display = st.sidebar.selectbox(
        "Test Preparation Course", options=list(test_prep_mapping.keys())
    )
    reading_score = st.sidebar.number_input(
        "Reading Score", min_value=0, max_value=100, value=50, step=1
    )
    writing_score = st.sidebar.number_input(
        "Writing Score", min_value=0, max_value=100, value=50, step=1
    )

    # Map user-friendly values to backend-ready values
    gender = gender_mapping[gender_display]
    race_ethnicity = race_mapping[race_ethnicity_display]
    parental_level_of_education = parental_education_mapping[parental_level_of_education_display]
    lunch = lunch_mapping[lunch_display]
    test_preparation_course = test_prep_mapping[test_preparation_course_display]

    # Button to trigger prediction
    if st.sidebar.button("Predict"):
        # Prepare input data
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )
        pred_df = data.get_data_as_data_frame()
        
        # Display input DataFrame to the user in a user-friendly format
        user_friendly_df = pd.DataFrame({
            "Gender": [gender_display],
            "Race/Ethnicity": [race_ethnicity_display],
            "Parental Level of Education": [parental_level_of_education_display],
            "Lunch": [lunch_display],
            "Test Preparation Course": [test_preparation_course_display],
            "Reading Score": [reading_score],
            "Writing Score": [writing_score],
        })
        st.write("Input Data:")
        st.write(user_friendly_df)

        # Predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Display results
        st.write("Prediction Result:")
        st.success(results[0])

if __name__ == "__main__":
    main()