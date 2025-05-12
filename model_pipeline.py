"""
This module contains the machine learning pipeline for data preparation,
model training, and evaluation.
"""

import os
import sys

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

    df = pd.read_csv("churn_modelling.csv")
    print("✅ Dataset loaded successfully!")

    # Checking for missing values
    print("Missing values per column:\n", df.isnull().sum())

    # Checking for duplicates
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

    # Separating features and target
    target = df["Churn"]
    features = df.drop(columns=["Churn"])
    print("✅ Target variable separated")

    # Frequency encoding for 'State'
    print("🔄 Applying frequency encoding for 'State'...")
    features["State"] = features["State"].map(
        features["State"].value_counts().to_dict()
    )
    print("✅ Frequency encoding applied")

    # Convert categorical 'Yes'/'No' columns to numeric 0/1
    print("🔄 Converting categorical variables to numeric...")
    features["International plan"] = features["International plan"].map(
        {"Yes": 1, "No": 0}
    )
    features["Voice mail plan"] = features["Voice mail plan"].map({"Yes": 1, "No": 0})
    print("✅ Categorical conversion completed")

    # Standardizing numerical features
    print("🔄 Applying standardization...")
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)
    print("✅ Standardization completed")

    # Apply PCA for feature reduction (keeping 95% variance)
    print("🔄 Performing PCA for dimensionality reduction...")
    pca_transformer = PCA(n_components=0.95)
    features_pca = pca_transformer.fit_transform(features_scaled)
    print(f"✅ PCA completed: Reduced to {features_pca.shape[1]} components")

    print("✅ Data preprocessing completed!")
    return features_pca, target, feature_scaler, pca_transformer


def train_model(train_features, train_labels):
    """Train a RandomForest model with SMOTEENN for handling class imbalance."""
    print("🚀 Applying SMOTEENN for class balancing...")
    smote_enn = SMOTEENN(sampling_strategy="auto", random_state=100)
    resampled_features, resampled_labels = smote_enn.fit_resample(
        train_features, train_labels
    )
    print(f"✅ SMOTEENN applied: {resampled_features.shape[0]} samples after resampling")

    print("🚀 Training the Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        random_state=100,
        max_depth=6,
        min_samples_leaf=8,
    )
    rf_model.fit(resampled_features, resampled_labels)
    print("✅ Model training completed!")

    return rf_model


def evaluate_model(model_instance, test_features, test_labels):
    """Evaluate the model's performance."""
    print("📊 Evaluating the model...")
    predictions = model_instance.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    conf_matrix = confusion_matrix(test_labels, predictions)

    print(f"✅ Model Accuracy: {accuracy:.4f}")
    print("🔍 Classification Report:\n", report)
    print("🔍 Confusion Matrix:\n", conf_matrix)
    return accuracy


def save_model(model_instance, feature_scaler, pca_transformer):
    """Save model and preprocessing artifacts."""
    print("💾 Saving model and preprocessing artifacts...")

    # Debugging: Check if the model instance is valid before saving
    if model_instance is None:
        print("❌ Error: Model instance is None, cannot save!")
        return

    joblib.dump(model_instance, "model.pkl")
    joblib.dump(feature_scaler, "scaler.pkl")
    joblib.dump(pca_transformer, "pca.pkl")

    # Debugging: Check if the file was actually created
    if os.path.exists("model.pkl"):
        print("✅ Model saved successfully!")
    else:
        print("❌ Error: model.pkl was NOT created!")


def load_model():
    """Load the trained model, scaler, and PCA."""
    print("📂 Loading model and preprocessing artifacts...")
    loaded_model_instance = joblib.load("model.pkl")
    loaded_scaler_instance = joblib.load("scaler.pkl")
    loaded_pca_instance = joblib.load("pca.pkl")
    print("✅ Model, scaler, and PCA loaded successfully!")
    return loaded_model_instance, loaded_scaler_instance, loaded_pca_instance


if __name__ == "__main__":
    print("🚀 Script execution started...")

    # Data Preparation
    X_processed, y_processed, data_scaler, pca_processor = prepare_data()
    print("✅ Data preparation finished")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )

    # Model Training
    trained_rf_model = train_model(X_train, y_train)
    print("✅ Model training finished")

    # Model Evaluation
    evaluate_model(trained_rf_model, X_test, y_test)
    print("✅ Model evaluation completed")

    # Save the model and preprocessing artifacts
    save_model(trained_rf_model, data_scaler, pca_processor)

    # Example of loading the model (Fixing Redefinition)
    final_model, final_scaler, final_pca = load_model()
    print("🎉 Script execution completed!")
