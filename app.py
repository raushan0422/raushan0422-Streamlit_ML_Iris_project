import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load trained model
@st.cache_resource
def load_model():
    model_path = "model/iris_model.pkl"
    if not os.path.exists(model_path):
        st.error("Trained model not found. Please run train_model.py first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()
class_names = model.classes_

# Page setup
st.set_page_config(page_title="🌼 Iris Classifier", layout="wide")

st.title("🌸 Iris Flower Species Classifier")
st.markdown("This app uses a trained **Random Forest** model to predict the species of Iris flower based on measurements of sepal and petal.")

st.markdown("---")

# Sidebar input section
st.sidebar.header("📝 Input Options")

def manual_input():
    sl = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sw = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    pl = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
    pw = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)
    return pd.DataFrame({
        "sepal_length": [sl],
        "sepal_width": [sw],
        "petal_length": [pl],
        "petal_width": [pw]
    })

uploaded = st.sidebar.file_uploader("📂 Upload CSV file", type=["csv"])
if uploaded:
    try:
        input_df = pd.read_csv(uploaded)
        expected_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        if not all(col in input_df.columns for col in expected_cols):
            st.error("Uploaded CSV must contain the columns: " + ", ".join(expected_cols))
            st.stop()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    input_df = manual_input()

st.subheader("🔍 Input Features")
st.dataframe(input_df, use_container_width=True)

# Prediction
st.markdown("---")
st.subheader("📈 Model Prediction")

try:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    for i in range(len(prediction)):
        st.success(f"Prediction {i+1}: **{prediction[i]}**")
        proba_df = pd.DataFrame(prediction_proba[i].reshape(1, -1), columns=class_names)
        st.bar_chart(proba_df.T)
except Exception as e:
    st.error(f"Prediction failed: {e}")

# Iris dataset overview and visualizations
st.markdown("---")
st.subheader("📊 Iris Dataset Insights")

if os.path.exists("data/iris.csv"):
    iris_df = pd.read_csv("data/iris.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌼 Species Count")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=iris_df, x="species", palette="pastel", ax=ax1)
        ax1.set_xlabel("Species")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with col2:
        st.markdown("### 📌 Feature Distribution")
        selected_feat = st.selectbox("Choose feature", iris_df.columns[:-1])
        fig2, ax2 = plt.subplots()
        sns.histplot(iris_df[selected_feat], kde=True, color="skyblue", ax=ax2)
        st.pyplot(fig2)

    st.markdown("### 🧾 Dataset Preview")
    st.dataframe(iris_df.head(), use_container_width=True)
else:
    st.warning("Dataset file not found in /data directory.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "Built with ❤️ using Streamlit | Internship Project by <b>Raushan Kumar</b>"
    "</div>", unsafe_allow_html=True
)
