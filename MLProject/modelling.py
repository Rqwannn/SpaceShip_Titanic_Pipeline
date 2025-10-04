import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

import os
import dagshub

dagshub.init(repo_owner='Rqwannn', repo_name='SpaceShip_Titanic_Pipeline', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Rqwannn/SpaceShip_Titanic_Pipeline.mlflow/")
mlflow.set_experiment("Spaceship Titanic Modeling")

data = pd.read_csv("spaceship_titanic_preprocessing.csv")
X = data.drop(columns=['Transported', "VIP", 'AgeGroup', 'NoSpend', 'SoloTraveler', 'GroupSize', 'Name', 'Destination', 'Cabin', 'CryoSleep'])
y = data['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

with mlflow.start_run() as run:
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    cer = 1 - accuracy
    
    print("\nClassification Error Rate (CER):", cer)
    mlflow.log_metric("classification_error_rate", cer)
    
    report_str = classification_report(y_test, y_pred_test)
    print("\nClassification Report:")
    print(report_str)
    mlflow.log_text(report_str, "classification_report.txt")
    
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Transported", "Transported"],
                yticklabels=["Not Transported", "Transported"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Random Forest")

    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    mlflow.log_artifact(confusion_matrix_path)
    plt.close()

    print("\nSaving model locally...")

    local_dir = "artifacts/local"
    
    os.makedirs(local_dir, exist_ok=True)
    local_model_path = os.path.join(local_dir, "stacking_model.pkl")
    
    joblib.dump(model, local_model_path)
    print(f"Model juga disimpan secara lokal di: {local_model_path}")
    
    print(f"Untuk melihat hasil, cek run ID: {run.info.run_id}")