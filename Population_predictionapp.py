import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
file_path = "/mnt/data/population_data/Metadata_Country_API_SP.POP.TOTL.FE.IN_DS2_en_CSV_v2_7318.csv"
df = pd.read_csv(file_path)

# Clean data
df = df.dropna()

# Select features and target
X = df[['Year']]
y = df['Population']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("Population Prediction App")

# User input for Year
year = st.slider("Enter Year", min_value=int(X.min()), max_value=2100, step=1)

# Real-time Prediction
predicted_population = model.predict(np.array([[year]]))[0]
st.write(f"Predicted Population in {year}: {predicted_population:.0f}")
