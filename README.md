# 🩺 Pima Indians Diabetes Predictor

A high-accuracy machine learning web application that predicts diabetes risk based on diagnostic measurements. This project was developed using a reverse-engineering approach to master model tuning and deployment pipelines.

## 🚀 Live Demo
[Click here to view the Live App](https://diabetespredictionml-ayushi.streamlit.app/)

## 📊 Project Overview
This project uses the **Pima Indians Diabetes Database** to build a predictive model. The primary goal was to achieve a high **ROC-AUC** while ensuring a robust, leakage-proof deployment pipeline.

### Key Performance Metrics:
* **Best Tuned ROC-AUC:** `0.801`
* **High-Risk Prediction Confidence:** Up to `93.45%`
* **Model:** Gaussian Naive Bayes (Optimized with `var_smoothing=0.001`)

## 🛠️ Technical Workflow
1. **Data Cleaning:** Handled missing values (zeros in Glucose, BMI, etc.) using `SimpleImputer` with a median strategy.
2. **Pipeline Architecture:** Built a `scikit-learn` Pipeline to bundle Imputation, Scaling, and the Model to prevent **Data Leakage**.
3. **Model Interpretation:** Identified **Glucose** (1.027 impact) and **BMI** (0.67 impact) as the top statistical drivers for prediction.
4. **Validation:** Verified model stability using **Stratified K-Fold Cross-Validation**.
5. **Deployment:** Serialized the final pipeline using `joblib` and deployed as an interactive **Streamlit** web app.

## 🧬 Feature Impact (Model Logic)
Based on the Z-score differences between classes, the model prioritizes:
1. **Glucose:** 1.027
2. **BMI:** 0.670
3. **Age:** 0.589
4. **Pregnancies:** 0.478

## 💻 How to Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com

2. Install dependencies
pip install -r requirements.txt

3. Run the application
streamlit run app.py
