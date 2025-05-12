import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Churn Prediction App", layout="centered")
# Paths
MODEL_PATH = Path("model.pkl")
SCALER_PATH = Path("scaler.pkl")
PCA_PATH = Path("pca.pkl")


# Chargement des objets
@st.cache_resource
def load_model_components():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    return model, scaler, pca


model, scaler, pca = load_model_components()

# Titre

st.title("📞 Client Churn Prediction")
st.write(
    "Remplissez les informations du client pour prédire s'il va quitter le service."
)

# Formulaire d'entrée
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Account_Length = st.number_input("Account Length", min_value=1, value=100)
        Area_Code = st.number_input(
            "Area Code", min_value=100, max_value=999, value=408
        )
        Customer_Service_Calls = st.slider("Customer Service Calls", 0, 10, 1)
        International_Plan = st.selectbox(
            "International Plan", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )
        Number_of_Voicemail_Messages = st.slider("Voicemail Messages", 0, 50, 5)
        Total_Day_Calls = st.slider("Total Day Calls", 0, 200, 100)
        Total_Day_Minutes = st.number_input("Total Day Minutes", 0.0, 400.0, 180.0)
        Total_Day_Charge = st.number_input("Total Day Charge", 0.0, 60.0, 30.0)

    with col2:
        Total_Night_Calls = st.slider("Total Night Calls", 0, 200, 90)
        Total_Night_Minutes = st.number_input("Total Night Minutes", 0.0, 400.0, 200.0)
        Total_Night_Charge = st.number_input("Total Night Charge", 0.0, 20.0, 9.0)
        Total_Evening_Calls = st.slider("Total Evening Calls", 0, 200, 100)
        Total_Evening_Minutes = st.number_input(
            "Total Evening Minutes", 0.0, 400.0, 180.0
        )
        Total_Evening_Charge = st.number_input("Total Evening Charge", 0.0, 40.0, 20.0)
        International_Calls = st.slider("International Calls", 0, 20, 3)
        Voicemail_Plan = st.selectbox(
            "Voicemail Plan", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )
        Total_Intl_Calls = st.slider("Total Intl Calls", 0, 20, 5)
        Total_Intl_Charge = st.number_input("Total Intl Charge", 0.0, 5.0, 1.5)

    submitted = st.form_submit_button("🔍 Prédire")

# Prédiction
if submitted:
    try:
        input_features = np.array(
            [
                [
                    Account_Length,
                    Area_Code,
                    Customer_Service_Calls,
                    International_Plan,
                    Number_of_Voicemail_Messages,
                    Total_Day_Calls,
                    Total_Day_Charge,
                    Total_Day_Minutes,
                    Total_Night_Calls,
                    Total_Night_Charge,
                    Total_Night_Minutes,
                    Total_Evening_Calls,
                    Total_Evening_Charge,
                    Total_Evening_Minutes,
                    International_Calls,
                    Voicemail_Plan,
                    Total_Intl_Calls,
                    Total_Intl_Charge,
                ]
            ]
        )

        input_scaled = scaler.transform(input_features)
        input_pca = pca.transform(input_scaled)

        prediction = model.predict(input_pca)[0]

        if prediction == 1:
            st.error("❌ Ce client est **à risque** de résiliation.")
        elif prediction == 0:
            st.success("✅ Ce client est **fidèle** et peu susceptible de résilier.")
        else:
            st.error("Erreur lors de la prédiction")

    except Exception as e:
        st.warning(f"Erreur lors de la prédiction : {e}")
