import streamlit as st
import joblib
import numpy as np

# Load your trained model
# model = pickle.load(open("model.pkl", "rb"))
try:
    model = joblib.load('iris_model')
except Exception as e:
    print(f"Error loading the model: {e}")

st.set_page_config(page_title="Specie Prediction", page_icon="🌼")

st.title("🌼 Species Prediction Using ML")

st.write("Predict species based on **Length and width**")

# Input field (replaces <input type="float">)

sepal_length=st.number_input(
    "Sepal Length"
)
sepal_width=st.number_input(
    "Sepal Width"
)
petal_length=st.number_input(
    "Petal Length"
)
petal_width=st.number_input(
    "Petal Width"
)
# Submit button (replaces <form> submit)
if st.button("Submit"):
    # Example prediction logic
    prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    if prediction[0]==0:
        species="Setosa"
    elif  prediction[0]==1:
        species="Versicolor"
    else:
        species="Virginica"


    st.success("Species Prediction:")
    st.write(f"{species:}")
