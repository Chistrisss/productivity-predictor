import streamlit as st
import pandas as pd
import joblib
import os

# IMPORTS NECESARIOS PARA RECONSTRUIR EL MODELO .JOBLIB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Importar el preprocessor original
from model_definitions import preprocessor

# Ruta del modelo
MODEL_PATH = "results/best_productivity_model.joblib"

# Verificar si el modelo existe
if not os.path.exists(MODEL_PATH):
    st.error("❌ Error: El modelo no se encontró. Asegúrate de subir 'best_productivity_model.joblib' dentro de /results/.")
    st.stop()

# Cargar el modelo
model = joblib.load(MODEL_PATH)

st.title("📉 Predicción de Baja Productividad por Uso de Redes Sociales")
st.write("Completa los campos y obtén una predicción del desempeño.")

# Inputs del usuario
age = st.number_input("Edad", min_value=10, max_value=100, value=20)
daily_usage = st.number_input("Tiempo diario en redes (min)", min_value=0, max_value=2000, value=120)
posts = st.number_input("Posts por día", min_value=0, max_value=500, value=1)
likes = st.number_input("Likes recibidos por día", min_value=0, max_value=20000, value=50)
comments = st.number_input("Comentarios por día", min_value=0, max_value=20000, value=10)
messages = st.number_input("Mensajes enviados por día", min_value=0, max_value=20000, value=20)

gender = st.selectbox("Género", ["Male", "Female", "Other"])
platform = st.selectbox("Plataforma principal", ["Instagram", "TikTok", "Facebook", "Twitter", "YouTube"])
emotion = st.selectbox("Emoción predominante", ["Happy", "Sad", "Angry", "Neutral"])

# Crear DataFrame para predicción
input_data = pd.DataFrame([{
    "Age": age,
    "Daily_Usage_Time (minutes)": daily_usage,
    "Posts_Per_Day": posts,
    "Likes_Received_Per_Day": likes,
    "Comments_Received_Per_Day": comments,
    "Messages_Sent_Per_Day": messages,
    "Gender": gender,
    "Platform": platform,
    "Dominant_Emotion": emotion,
    "usage_bin": "medium"  # valor neutral, ya que el usuario no lo ingresa
}])

# Botón de predicción
if st.button("Predecir"):
    pred = model.predict(input_data)[0]

    st.subheader("📊 Resultado:")

    if pred == 1:
        st.error("🔴 *Alta probabilidad de baja productividad*")
    else:
        st.success("🟢 *Buena productividad esperada*")
