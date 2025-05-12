import os
import sys
import argparse
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)
from sklearn.model_selection import train_test_split
from datetime import datetime
from elasticsearch import Elasticsearch

# ✅ MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5002")

# ✅ Dossier pour les artefacts
ARTIFACTS_DIR = "mlartifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ✅ Connexion Elasticsearch
es = Elasticsearch("http://localhost:9200")


def log_to_elasticsearch(step, message, data=None):
    """Envoie des logs vers Elasticsearch"""
    doc = {
        "timestamp": datetime.utcnow().isoformat(),
        "step": step,
        "message": message,
        "data": data or {},
    }
    try:
        es.index(index="ml-pipeline-logs", document=doc)
    except Exception as e:
        print(f"❌ ERREUR en loggant dans Elasticsearch: {e}")


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline CLI")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")

    args = parser.parse_args()

    if args.prepare:
        print("📊 Exécution de prepare_data()...")
        X_processed, y_processed, scaler, pca = prepare_data()

        # Sauvegarde locale
        joblib.dump(X_processed, os.path.join(ARTIFACTS_DIR, "X_processed.pkl"))
        joblib.dump(y_processed, os.path.join(ARTIFACTS_DIR, "y_processed.pkl"))
        joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
        joblib.dump(pca, os.path.join(ARTIFACTS_DIR, "pca.pkl"))

        print("✅ Données préparées et enregistrées avec succès !")
        log_to_elasticsearch("prepare", "Données préparées et sauvegardées.")

    if args.train:
        print("🚀 Entraînement du modèle...")

        X_processed = joblib.load(os.path.join(ARTIFACTS_DIR, "X_processed.pkl"))
        y_processed = joblib.load(os.path.join(ARTIFACTS_DIR, "y_processed.pkl"))
        scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
        pca = joblib.load(os.path.join(ARTIFACTS_DIR, "pca.pkl"))

        X_train, _, y_train, _ = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42
        )

        with mlflow.start_run():
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            trained_model = train_model(X_train, y_train)
            mlflow.sklearn.log_model(trained_model, "model")

            save_model(trained_model, scaler, pca)
            print("✅ model.pkl saved successfully!")

            accuracy = evaluate_model(trained_model, X_train, y_train)
            mlflow.log_metric("accuracy", accuracy)
            print(f"✅ Accuracy logged in MLflow: {accuracy:.4f}")

            log_to_elasticsearch(
                "train",
                "Modèle entraîné",
                {
                    "accuracy": accuracy,
                    "model_type": "RandomForestClassifier",
                    "data_shape": list(X_train.shape),
                },
            )

    if args.evaluate:
        print("📊 Évaluation du modèle...")

        X_processed = joblib.load(os.path.join(ARTIFACTS_DIR, "X_processed.pkl"))
        y_processed = joblib.load(os.path.join(ARTIFACTS_DIR, "y_processed.pkl"))

        _, X_test, _, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42
        )

        try:
            model, _, _ = load_model()
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print(
                "❌ Error: Trained model file not found. Run `python main.py --train` first."
            )
            return

        accuracy = evaluate_model(model, X_test, y_test)

        print(f"🔍 Accuracy: {accuracy:.4f}")

        if accuracy is not None:
            log_to_elasticsearch(
                "evaluate",
                "Modèle évalué",
                {
                    "accuracy": accuracy,
                    "dataset": "test",
                },
            )
        else:
            print("⚠️ Warning: Accuracy is None, skipping.")

        print(f"✅ Évaluation terminée avec une précision de {accuracy:.4f}")


if __name__ == "__main__":
    main()
