from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from elasticsearch import Elasticsearch
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime

# Paths
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
PCA_PATH = "pca.pkl"
TRAINING_DATA_PATH = "churn_modelling.csv"

# FastAPI app
app = FastAPI()

# Elasticsearch client
es = Elasticsearch("http://localhost:9200")  # ou ton URL Docker/Cloud

# Logger MLflow vers Elasticsearch
logger = logging.getLogger("mlflow")
logger.setLevel(logging.INFO)
logger.info("Test log vers Elasticsearch depuis FastAPI")


class ElasticsearchHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        es.index(
            index="mlflow-metrics",
            document={
                "message": log_entry,
                "@timestamp": datetime.utcnow().isoformat(),
            },
        )


# Ajout du handler à logger mlflow
es_handler = ElasticsearchHandler()
es_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(es_handler)


# Pydantic model pour les hyperparamètres d'entraînement
class TrainingParams(BaseModel):
    n_estimators: int
    max_depth: int


# Fonction de prétraitement
def preprocess_data(df):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA

    label_encoder = LabelEncoder()
    df["International plan"] = label_encoder.fit_transform(df["International plan"])
    df["Voice mail plan"] = label_encoder.fit_transform(df["Voice mail plan"])

    X = df.drop(columns=["Churn", "State"])
    y = df["Churn"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(10, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, scaler, pca


# Route de réentrainement
@app.post("/retrain")
def retrain_model(params: TrainingParams):
    try:
        if not os.path.exists(TRAINING_DATA_PATH):
            raise Exception("Training dataset not found.")

        import pandas as pd

        df = pd.read_csv(TRAINING_DATA_PATH)

        if "Churn" not in df.columns:
            raise Exception("Dataset must include 'Churn' column.")

        X_pca, y, scaler, pca = preprocess_data(df)

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42
        )

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=params.n_estimators, max_depth=params.max_depth
        )
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        mlflow.set_tracking_uri("http://127.0.0.1:5002")

        with mlflow.start_run():
            mlflow.log_param("n_estimators", params.n_estimators)
            mlflow.log_param("max_depth", params.max_depth)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"Model retrained with accuracy: {acc}")

        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(pca, PCA_PATH)

        return {
            "message": "✅ Model retrained successfully.",
            "params_used": params.dict(),
            "accuracy": acc,
        }
    except Exception as e:
        return {"error": f"Retraining failed: {str(e)}"}


# Modèle Pydantic pour les données de prédiction
class InputData(BaseModel):
    Account_Length: int
    Area_Code: int
    Customer_Service_Calls: int
    International_Plan: int  # 0 ou 1
    Number_of_Voicemail_Messages: int
    Total_Day_Calls: int
    Total_Day_Charge: float
    Total_Day_Minutes: float
    Total_Night_Calls: int
    Total_Night_Charge: float
    Total_Night_Minutes: float
    Total_Evening_Calls: int
    Total_Evening_Charge: float
    Total_Evening_Minutes: float
    International_Calls: int
    Voicemail_Plan: int  # 0 ou 1
    Total_Intl_Calls: int
    Total_Intl_Charge: float


class PredictionResponse(BaseModel):
    prediction: int


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        pca = joblib.load(PCA_PATH)

        expected_features = [
            "Account_Length",
            "Area_Code",
            "Customer_Service_Calls",
            "International_Plan",
            "Number_of_Voicemail_Messages",
            "Total_Day_Calls",
            "Total_Day_Charge",
            "Total_Day_Minutes",
            "Total_Night_Calls",
            "Total_Night_Charge",
            "Total_Night_Minutes",
            "Total_Evening_Calls",
            "Total_Evening_Charge",
            "Total_Evening_Minutes",
            "International_Calls",
            "Voicemail_Plan",
            "Total_Intl_Calls",
            "Total_Intl_Charge",
        ]

        input_data_dict = data.dict()
        input_features = np.array(
            [input_data_dict[feature] for feature in expected_features]
        ).reshape(1, -1)

        input_scaled = scaler.transform(input_features)
        input_pca = pca.transform(input_scaled)

        prediction = model.predict(input_pca)[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}"
        )
