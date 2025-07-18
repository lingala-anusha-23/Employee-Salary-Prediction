import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the trained model
try:
    model = joblib.load("best_model.pkl")
    # Get the feature names the model expects
    feature_names = model.feature_names_in_
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Define all possible categories from the original dataset
categories = {
    'workclass': ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", 
                 "Self-emp-inc", "Federal-gov", "Others"],
    'education': ["Bachelors", "Some-college", "11th", "HS-grad", 
                "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", 
                "7th-8th", "12th", "Masters", "1st-4th", 
                "10th", "Doctorate", "5th-6th", "Preschool"],
    'marital-status': ["Married-civ-spouse", "Divorced", "Never-married", 
                      "Separated", "Widowed", "Married-spouse-absent", 
                      "Married-AF-spouse"],
    'occupation': ["Prof-specialty", "Craft-repair", "Exec-managerial", 
                  "Adm-clerical", "Sales", "Other-service", 
                  "Machine-op-inspct", "Transport-moving", 
                  "Handlers-cleaners", "Farming-fishing", 
                  "Tech-support", "Protective-serv", 
                  "Priv-house-serv", "Armed-Forces", "Others"],
    'relationship': ["Husband", "Not-in-family", "Own-child", 
                    "Unmarried", "Wife", "Other-relative"],
    'race': ["White", "Black", "Asian-Pac-Islander", 
            "Amer-Indian-Eskimo", "Other"],
    'gender': ["Male", "Female"],
    'native-country': ["United-States", "Mexico", "Philippines", "Germany", "Canada",
                      "Puerto-Rico", "El-Salvador", "India", "Cuba", "England",
                      "China", "South", "Jamaica", "Italy", "Dominican-Republic",
                      "Vietnam", "Guatemala", "Japan", "Poland", "Columbia",
                      "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua",
                      "Peru", "Greece", "France", "Ecuador", "Ireland",
                      "Hong", "Trinadad&Tobago", "Cambodia", "Laos", "Thailand",
                      "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Hungary",
                      "Honduras", "Scotland", "Holand-Netherlands"]
}

# Education to numerical mapping
education_num_mapping = {
    "Bachelors": 13,
    "Some-college": 10,
    "11th": 7,
    "HS-grad": 9,
    "Prof-school": 15,
    "Assoc-acdm": 12,
    "Assoc-voc": 11,
    "9th": 5,
    "7th-8th": 4,
    "12th": 8,
    "Masters": 14,
    "1st-4th": 2,
    "10th": 6,
    "Doctorate": 16,
    "5th-6th": 3,
    "Preschool": 1
}

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# Create input dictionary with default values for all features
default_values = {
    'age': 35,
    'workclass': 'Private',
    'education': 'Bachelors',
    'marital-status': 'Never-married',
    'occupation': 'Prof-specialty',
    'relationship': 'Husband',
    'race': 'White',
    'gender': 'Male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'fnlwgt': 100000,
    'native-country': 'United-States',
    'educational-num': 13  # Default for Bachelors
}

# Create input widgets for each feature
input_data = {}
for feature in feature_names:
    if feature == 'age':
        input_data[feature] = st.sidebar.slider("Age", 17, 90, default_values['age'])
    elif feature == 'fnlwgt':
        input_data[feature] = st.sidebar.number_input("Final Weight (fnlwgt)", 
                                                     min_value=0, 
                                                     value=default_values['fnlwgt'])
    elif feature == 'educational-num':
        # Will be calculated from education level
        continue
    elif feature == 'capital-gain':
        input_data[feature] = st.sidebar.number_input("Capital Gain", 
                                                     min_value=0, 
                                                     value=default_values['capital-gain'])
    elif feature == 'capital-loss':
        input_data[feature] = st.sidebar.number_input("Capital Loss", 
                                                     min_value=0, 
                                                     value=default_values['capital-loss'])
    elif feature == 'hours-per-week':
        input_data[feature] = st.sidebar.slider("Hours per Week", 
                                              1, 99, 
                                              default_values['hours-per-week'])
    elif feature in categories:
        input_data[feature] = st.sidebar.selectbox(
            feature.replace('-', ' ').title(), 
            categories[feature],
            index=categories[feature].index(default_values.get(feature, categories[feature][0]))
        )
    else:
        st.warning(f"Unexpected feature: {feature}")

# Calculate educational-num based on education level
if 'education' in input_data and 'educational-num' in feature_names:
    input_data['educational-num'] = education_num_mapping.get(input_data['education'], 9)

# Ensure all expected features are present
for feature in feature_names:
    if feature not in input_data:
        input_data[feature] = default_values.get(feature, 0)

# Convert to DataFrame with columns in correct order
input_df = pd.DataFrame([input_data])[feature_names]

# Display raw input
st.write("### 🔎 Input Data")
st.write(input_df)

# Preprocessing function
def preprocess_input(data):
    # Create a copy to avoid modifying original
    processed = data.copy()
    
    # List of categorical columns (excluding numerical ones)
    categorical_cols = list(set(feature_names) & set(categories.keys()))
    
    # Initialize and fit encoders
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on all possible values
        le.fit(categories[col])
        processed[col] = le.transform(processed[col])
    
    # Ensure no NaN values remain
    processed = processed.fillna(0)  # Replace any potential NaN with 0
    
    return processed

if st.button("Predict Salary Class"):
    try:
        # Preprocess the input
        processed_input = preprocess_input(input_df)
        
        # Ensure columns are in exact same order as training
        processed_input = processed_input[feature_names]
        
        # Final check for NaN values
        if processed_input.isnull().values.any():
            st.error("Input contains missing values after preprocessing")
            st.stop()
        
        # Make prediction
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)
        
        # Display results
        st.success(f"✅ Prediction: {'>50K' if prediction[0] == 1 else '≤50K'}")
        st.write(f"Confidence: {max(probability[0]):.1%}")
        
        # Show feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            features = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(features.set_index('Feature'))
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Add some info about the model
st.markdown("---")
st.subheader("About the Model")
st.write(f"""
This prediction model was trained using the Adult Census Income dataset with these features:
{', '.join(feature_names)}.
""")

# Add footer
st.markdown("---")
st.caption("Employee Salary Prediction App | Created with Streamlit")