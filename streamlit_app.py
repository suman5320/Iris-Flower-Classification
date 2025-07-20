import streamlit as st
import requests

st.title("🌸 Iris Flower Predictor")

st.write("Enter flower measurements:")

sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
petal_length = st.number_input("Petal Length (cm)", value=1.4)
petal_width = st.number_input("Petal Width (cm)", value=0.2)

if st.button("Predict"):
    features = [sepal_length, petal_length, petal_width]

    # Change URL if deployed
    url = "https://iris-flower-classification-f2gp.onrender.com/predict"

    response = requests.post(url, json={"features": features})

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Class: {result['class_name']}")
    else:
        st.error("Prediction failed.")
