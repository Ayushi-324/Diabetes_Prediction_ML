import streamlit as st
import pandas as pd
import joblib

# 1. Load the "Saved Brain" (Your Pipeline)
model = joblib.load('diabetes_final_model.pkl')

# 2. App Title and Description
st.set_page_config(page_title="Pima Diabetes Predictor", page_icon="🩺")
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter patient metrics below to calculate the probability of diabetes.")

# 3. Create Input Fields for all 8 Features
st.sidebar.header("Patient Data")

def get_user_input():
    # We use sliders and number inputs for ease of use
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 2)
    glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
    bp = st.sidebar.slider("Blood Pressure", 0, 140, 70)
    skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin Level", 0, 850, 80)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.5)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
    age = st.sidebar.slider("Age", 21, 100, 33)

    # Put data into a DataFrame with the EXACT names your model expects
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

# 4. Get the Input
input_df = get_user_input()

# 5. Display the Patient Summary
st.subheader("Patient Summary")
st.write(input_df)

# 6. Make the Prediction
if st.button("Calculate Risk Score"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1][0] # Get the first score

    st.subheader("Results")
    
    # 🌈 Colorful Result Boxes
    if prediction[0] == 1:
        st.error(f"🚨 High Risk: Diabetic Pattern Detected")
        st.info("💡 Tip: Please consult a healthcare professional for a formal diagnosis.")
    else:
        st.success(f"✅ Low Risk: Healthy Pattern Detected")
        st.info("💡 Tip: Keep up the healthy habits like exercise and a balanced diet!")
        
    # 📊 Confidence Score & Progress Bar
    st.write(f"**Confidence Score:** {probability:.2%}")
    st.progress(float(probability))

# 7. The Important "Safety Label" (Disclaimer)
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This app is for educational purposes only. It uses the Pima Indians Diabetes dataset and is NOT a substitute for professional medical advice, diagnosis, or treatment.")
