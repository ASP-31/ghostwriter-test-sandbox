
# Lines 1-15: Import all necessary libraries for the machine learning pipeline
import logging
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Lines 16-25: Configure logging and environment settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(42)

# Lines 26-40: Define the core data loader and preprocessing function
def load_and_preprocess_data():
    logger.info("Loading Breast Cancer Wisconsin dataset...")
    raw_data = load_breast_cancer()
    X = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    y = raw_data.target
    logger.info(f"Dataset shapes - Features: {X.shape}, Target: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_split=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, raw_data.target_names

# Lines 41-55: Define model tuning using Grid Search Cross-Validation
def tune_and_train_model(X_train, y_train):
    logger.info("Initializing Random Forest Classifier...")
    base_estimator = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    logger.info("Starting GridSearchCV for hyperparameter optimization...")
    grid_search = GridSearchCV(
        estimator=base_estimator, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Optimal Hyperparameters Found: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Lines 56-75: Evaluate the trained model performance
def evaluate_model_performance(model, X_test, y_test, class_names):
    logger.info("Generating predictions on the test partition...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions, target_names=class_names)
    print("\n" + "="*50)
    print(f"MODEL ACCURACY PERFORMANCE: {accuracy:.4f}")
    print("="*50)
    print("CONFUSION MATRIX OVERVIEW:")
    print(conf_matrix)
    print("="*50)
    print("DETAILED CLASSIFICATION REPORT:")
    print(class_report)
    print("="*50 + "\n")
    return accuracy

# Lines 76-88: Serialize and save artifacts to disk
def save_pipeline_artifacts(model, scaler, model_path="model.pkl", scaler_path="scaler.pkl"):
    logger.info(f"Saving trained model artifact to: {model_path}")
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    logger.info(f"Saving fitted scaler artifact to: {scaler_path}")
    with open(scaler_path, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

# Lines 89-100: Orchestrate execution block
if __name__ == "__main__":
    logger.info("Starting End-to-End Machine Learning Pipeline Execution.")
    X_tr, X_te, y_tr, y_te, data_scaler, target_labels = load_and_preprocess_data()
    optimized_model = tune_and_train_model(X_tr, y_tr)
    final_accuracy = evaluate_model_performance(optimized_model, X_te, y_te, target_labels)
    save_pipeline_artifacts(optimized_model, data_scaler)
    logger.info("Pipeline executed successfully without errors.")
