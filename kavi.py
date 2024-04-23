import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict using the loaded model
def predict_target( age, sex ,cp ,trestbps,chol,fbs,thalach,oldpeak,restecg,exang,slope,ca,thal):
    features = np.array([age, sex ,cp ,trestbps,chol,fbs,thalach,oldpeak,restecg,exang,slope,ca,thal])
    features = features.reshape(1,-1)
    target = model.predict(features)
    return target[0]

# Streamlit UI
st.title('EMISSION Prediction')
st.write("""
## Input Features
Enter the values for the input features .
""")

# Input fields for user 
age = st.number_input('age')
sex = st.number_input('sex')
cp = st.number_input('cp')
trestbps = st.number_input('trestbps')
chol = st.number_input('chol')
fbs = st.number_input('fbs')
thalach = st.number_input('thalach')
oldpeak=st.number_input('oldpeak')
restecg=st.number_input('restecg')
exang=st.number_input('exang')
slope=st.number_input('slope')
ca=st.number_input('ca')
thal=st.number_input('thal')
# Prediction button
if st.button('Predict'):
    
    target_prediction = predict_target(age, sex ,cp ,trestbps,chol,fbs,thalach,oldpeak,restecg,exang,slope,ca,thal)
    st.write(f"Predicted target: {target_prediction}")