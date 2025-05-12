"""
This module contains the machine learning pipeline for data preparation,
model training, and evaluation.
"""

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.combine import SMOTEENN


def prepare_data():
    """Load and preprocess the dataset."""
    print("🚀 Starting data preparation...")

    try:
        df = pd.read_csv("churn_modelling.csv")
        print("✅ Dataset loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: 'churn_modelling.csv' not found!")
        return None, None, None, None

    print("🔍 Checking for missing values...")
    print(df.isnull().sum())

    print(f"🔍 Number of duplicate rows: {df.duplicated().sum()}")

    # Separate target and features
    target = df["Churn"]
    features = df.drop(columns=["Churn"])
    print("✅ Target variable separated.")

    # Frequency encoding for 'State'
    print("🔄 Encoding 'State' using frequency...")
    features["State"] = features["State"].map(features["State"].value_counts())

    # Convert 'Yes'/'No' to 1/0
    print("🔄 Converting categorical variables...")
    binary_map = {"Yes": 1, "No": 0}
    features["International plan"] = features["International plan"].map(binary_map)
    features["Voice mail plan"] = features["Voice mail plan"].map(binary_map)

    # Standardize features
    print("🔄 Standardizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA to retain 95% variance
    print("🔄 Applying PCA...")
    pca = PCA(n_components=0.95)
    features_pca = pca.fit_transform(features_scaled)
    print(f"✅ PCA completed. Reduced to {features_pca.shape[1]} components.")

    return features_pca, target, scaler, pca


def train_model(X_train, y_train):
    """Train Random Forest with SMOTEENN for balancing."""
    print("🚀 Balancing data with SMOTEENN...")
    smote_enn = SMOTEENN(random_state=100)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    print(f"✅ Resampled dataset shape: {X_resampled.shape}")

    print("🚀 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        random_state=100,
        max_depth=6,
        min_samples_leaf=8,
    )
    model.fit(X_resampled, y_resampled)
    print("✅ Model training completed.")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    print("📊 Evaluating model...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Accuracy: {accuracy:.4f}")
    print("🔍 Classification Report:")
    print(classification_report(y_test, predictions))
    print("🔍 Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return accuracy


def save_model(model, scaler, pca):
    """Save model and preprocessing pipeline."""
    print("💾 Saving model and preprocessing tools...")
    if model is None:
        print("❌ Error: Model is None, cannot save.")
        return

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca.pkl")

    if os.path.exists("model.pkl"):
        print("✅ Model and artifacts saved successfully.")
    else:
        print("❌ Saving failed: model.pkl not found.")


def load_model():
    """Load saved model and preprocessing tools."""
    print("📂 Loading model and preprocessing pipeline...")
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
        print("✅ All components loaded successfully.")
        return model, scaler, pca
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None


if __name__ == "__main__":
    print("🏁 Pipeline execution started...")

    # Step 1: Prepare data
    X, y, scaler, pca = prepare_data()
    if X is None:
        print("❌ Data preparation failed. Exiting pipeline.")
        exit()

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Train model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate model
    evaluate_model(model, X_test, y_test)

    # Step 5: Save artifacts
    save_model(model, scaler, pca)

    # Step 6: Load model example
    final_model, final_scaler, final_pca = load_model()

    print("🎉 Pipeline execution completed successfully!")
