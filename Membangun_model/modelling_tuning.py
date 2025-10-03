import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import re

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Spaceship Titanic Modeling")

data = pd.read_csv("spaceship_titanic_preprocessing.csv")
X = data.drop(columns=['Transported', "VIP", 'AgeGroup', 'NoSpend', 'SoloTraveler', 'GroupSize', 'Name', 'Destination', 'Cabin', 'CryoSleep'])
y = data['Transported']

# FIX: Clean column names untuk LightGBM
def clean_column_names(df):
    """Clean column names by removing special characters"""
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
    return df

X = clean_column_names(X)
X = X.astype('float64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

def create_report(y_true, y_pred, dataset_name="Dataset"):
    """Function to create and print classification report"""
    print(f"\n{'='*50}")
    print(f"{dataset_name} - Classification Report")
    print('='*50)
    print(classification_report(y_true, y_pred))
    
    accuracy = accuracy_score(y_true, y_pred)
    cer = 1 - accuracy
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Classification Error Rate (CER): {cer:.4f}")
    
    return accuracy, cer

with mlflow.start_run() as run:
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        silent=True  
    )
    
    print("Building Stacking Classifier...")
    
    base_estimators = [
        ('xgb', XGBClassifier(eval_metric="logloss", random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42, verbose=-1)),
        ('cat', CatBoostClassifier(verbose=0, random_state=42))
    ]
    
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(
            max_iter=10000,
            solver='liblinear',
            random_state=42
        ),
        passthrough=True
    )
    
    print("Performing Hyperparameter Tuning...")
    
    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1],
        'lgbm__n_estimators': [100, 200],
        'lgbm__max_depth': [3, 5],
        'cat__iterations': [100, 200],
        'cat__depth': [4, 6]
    }
    
    grid_search = GridSearchCV(
        estimator=stack,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    print("\nBest Parameters:")
    print(grid_search.best_params_)
    mlflow.log_params(grid_search.best_params_)
    
    # Log final estimator parameters
    mlflow.log_param("final_estimator_solver", "liblinear")
    mlflow.log_param("final_estimator_max_iter", 10000)
    
    y_train_predict = best_model.predict(X_train)
    y_test_predict = best_model.predict(X_test)
    
    train_acc, train_cer = create_report(y_train, y_train_predict, "Training Set")
    test_acc, test_cer = create_report(y_test, y_test_predict, "Test Set")
    
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("train_classification_error_rate", train_cer)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_classification_error_rate", test_cer)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    train_report = classification_report(y_train, y_train_predict)
    test_report = classification_report(y_test, y_test_predict)
    mlflow.log_text(train_report, "train_classification_report.txt")
    mlflow.log_text(test_report, "test_classification_report.txt")
    
    print("\nGenerating Feature Importance Plot...")
    cat = best_model.named_estimators_['cat']
    feat_importance = cat.get_feature_importance()
    
    plt.figure(figsize=(10, 8))
    plt.barh(X_train.columns, feat_importance)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title("CatBoost Feature Importance")
    plt.tight_layout()

    feature_importance_path = "feature_importance.png"
    plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(feature_importance_path)
    plt.close()
    
    cm_test = confusion_matrix(y_test, y_test_predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Transported", "Transported"],
                yticklabels=["Not Transported", "Transported"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set (Stacking)")
    
    confusion_matrix_test_path = "confusion_matrix_test.png"
    plt.savefig(confusion_matrix_test_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(confusion_matrix_test_path)
    plt.close()
    
    cm_train = confusion_matrix(y_train, y_train_predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Not Transported", "Transported"],
                yticklabels=["Not Transported", "Transported"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Train Set (Stacking)")

    confusion_matrix_train_path = "confusion_matrix_train.png"
    plt.savefig(confusion_matrix_train_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(confusion_matrix_train_path)
    plt.close()
    
    print(f"\n{'='*50}")
    print(f"Untuk melihat hasil, cek run ID: {run.info.run_id}")
    print('='*50)