import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model and scaler
MODEL_PATH = "trained_model.h5"  # Path to the saved model
SCALER_PATH = "scaler.pkl"       # Path to the saved scaler

# Load model
model = load_model(MODEL_PATH)

# Load scaler
with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

# Define the 12 features used in the model and their descriptions
features = {
    "bedrooms": "Number of bedrooms",
    "bathrooms": "Number of bathrooms",
    "sqft_living": "Total living area in square feet",
    "sqft_lot": "Lot size in square feet",
    "floors": "Number of floors",
    "waterfront": "1 if the property has a waterfront, 0 otherwise",
    "view": "Quality of the view (0-4 scale)",
    "condition": "Condition of the property (1-5 scale)",
    "sqft_above": "Living area above ground in square feet",
    "sqft_basement": "Living area in the basement in square feet",
    "yr_built": "Year the property was built",
    "yr_renovated": "Year the property was renovated (0 if never renovated)",
}

# Streamlit app layout
st.title("House Price Prediction App")
st.markdown("""
This application predicts house prices based on various features. Below are the feature descriptions:

- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms
- **Sqft Living**: Total living area in square feet
- **Sqft Lot**: Lot size in square feet
- **Floors**: Number of floors
- **Waterfront**: Yes if the property has a waterfront, No otherwise
- **View**: Quality of the view (0-4 scale)
- **Condition**: Condition of the property (1-5 scale)
- **Sqft Above**: Living area above ground in square feet
- **Sqft Basement**: Living area in the basement in square feet
- **Year Built**: Year the property was built
- **Year Renovated**: Year the property was renovated (0 if never renovated)
""")

# Initialize session state for inputs
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "bedrooms": 0,
        "bathrooms": 0.0,
        "sqft_living": 0.0,
        "sqft_lot": 0.0,
        "floors": 0.0,
        "waterfront": "No",
        "view": 0,
        "condition": 1,
        "sqft_above": 0.0,
        "sqft_basement": 0.0,
        "yr_built": 0,
        "yr_renovated": 0,
    }

# Input fields for user
st.subheader("Input Features")
user_input = {}
for feature, description in features.items():
    if feature == "waterfront":
        user_input[feature] = st.radio(f"{feature.replace('_', ' ').title()} ({description})", ["Yes", "No"])
    elif feature == "view":
        user_input[feature] = st.slider(f"{feature.replace('_', ' ').title()} ({description})", 0, 4, 0)
    elif feature == "condition":
        user_input[feature] = st.slider(f"{feature.replace('_', ' ').title()} ({description})", 1, 5, 1)
    elif feature in ["bedrooms", "yr_built", "yr_renovated"]:
        user_input[feature] = st.number_input(
            f"{feature.replace('_', ' ').title()} ({description})",
            min_value=0, value=0, step=1, format="%d"
        )
    elif feature in ["bathrooms", "floors"]:
        user_input[feature] = st.number_input(
            f"{feature.replace('_', ' ').title()} ({description})",
            min_value=0.0, value=0.0, step=0.1, format="%.1f"
        )
    else:
        user_input[feature] = st.number_input(
            f"{feature.replace('_', ' ').title()} ({description})",
            min_value=0.0, value=0.0, step=0.1
        )

# Convert Yes/No to 1/0 for waterfront
user_input["waterfront"] = 1 if user_input["waterfront"] == "Yes" else 0

# Clear button logic
if st.button("Clear"):
    st.session_state.inputs = {key: 0 if key in ["bedrooms", "view", "condition", "yr_built", "yr_renovated"] else 0.0 for key in user_input.keys()}
    st.experimental_rerun()

# Predict button logic
if st.button("Predict"):
    input_data = pd.DataFrame([user_input])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    predicted_price = np.expm1(prediction[0][0])  # Assuming log transformation was used during training
    st.success(f"Predicted House Price: ${predicted_price:,.2f}")




# To run StreamLit use---> streamlit run app.py


