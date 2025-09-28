import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, RobustScaler
import math
import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
from typing import Text, Dict

def validate_inputs(file_path: Text):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File data tidak ditemukan di '{file_path}'")
    print("Validasi path input berhasil.")

def log_dataframe_summary(df: pd.DataFrame, stage_name: Text):
    print(f"Mencatat ringkasan data untuk tahap: {stage_name}...")
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    mlflow.log_text(buffer.getvalue(), f"summary/{stage_name}_info.txt")

    describe_str = df.describe(include='all').to_string()
    mlflow.log_text(describe_str, f"summary/{stage_name}_describe.txt")

    head_str = df.head().to_string()
    mlflow.log_text(head_str, f"summary/{stage_name}_head.txt")
    
    print(f"Ringkasan data {stage_name} berhasil dicatat.")

def log_eda_metrics(df: pd.DataFrame, stage_name: Text):
    print(f"Mencatat metrik EDA untuk tahap: {stage_name}...")
    
    mlflow.log_metric(f"{stage_name}_row_count", df.shape[0])
    mlflow.log_metric(f"{stage_name}_column_count", df.shape[1])
    mlflow.log_metric(f"{stage_name}_missing_values", df.isnull().sum().sum())
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    if 'Age' in numeric_cols:
        mlflow.log_metric(f"{stage_name}_age_mean", df['Age'].mean())
        mlflow.log_metric(f"{stage_name}_age_median", df['Age'].median())
    if 'TotalSpend' in df.columns:
        mlflow.log_metric(f"{stage_name}_total_spend_mean", df['TotalSpend'].mean())
        mlflow.log_metric(f"{stage_name}_total_spend_median", df['TotalSpend'].median())
    if 'Transported' in df.columns: 
        if df['Transported'].dtype in [int, float]:
             mlflow.log_metric(f"{stage_name}_target_balance", df['Transported'].value_counts(normalize=True).get(1, 0))

    print("Metrik EDA berhasil dicatat.")

def perform_eda_and_log(df: pd.DataFrame):
    print("Memulai tahap Exploratory Data Analysis (EDA)...")
    eda_dir = "eda_plots"
    os.makedirs(eda_dir, exist_ok=True)

    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    if not df_numeric.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of Numerical Features")
        
        corr_matrix_path = os.path.join(eda_dir, "correlation_matrix.png")

        plt.savefig(corr_matrix_path)
        plt.close()
        mlflow.log_artifact(corr_matrix_path, "eda_plots")
        print("Matriks korelasi dicatat.")

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df_numeric.columns, 1):
        plt.subplot(math.ceil(len(df_numeric.columns) / 2), 2, i)
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f'Distribution of {col}')
    
    plt.tight_layout()
    dist_num_path = os.path.join(eda_dir, "numerical_distributions.png")
    plt.savefig(dist_num_path)
    plt.close()

    mlflow.log_artifact(dist_num_path, "eda_plots")
    print("Distribusi fitur numerik dicatat.")

    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    plt.figure(figsize=(15, 5 * math.ceil(len(categorical_cols) / 2)))

    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(math.ceil(len(categorical_cols) / 2), 2, i)
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')

    plt.tight_layout()
    dist_cat_path = os.path.join(eda_dir, "categorical_distributions.png")
    plt.savefig(dist_cat_path)
    plt.close()

    mlflow.log_artifact(dist_cat_path, "eda_plots")
    print("Distribusi fitur kategorikal dicatat.")

    homeplanet_counts = df['HomePlanet'].value_counts()

    plt.style.use('default')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
              '#FFB6C1', '#98FB98', '#F0E68C', '#87CEEB', '#DEB887', '#F4A460']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    homeplanet_counts.plot(kind='barh', ax=ax1,
                          color=colors[:len(homeplanet_counts)],
                          alpha=0.8,
                          edgecolor='white',
                          linewidth=0.5)

    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax1.set_title('Distribusi Home Planet\n(Bar Chart)',
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Jumlah Penumpang', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Home Planet', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    for i, (planet, count) in enumerate(homeplanet_counts.items()):
        percentage = (count / homeplanet_counts.sum()) * 100
        ax1.text(count + max(homeplanet_counts) * 0.01, i,
                f'{count} ({percentage:.1f}%)',
                va='center', fontsize=10, fontweight='bold')

    explode = [0.1 if i == 0 else 0.05 if i < 3 else 0 for i in range(len(homeplanet_counts))]

    wedges, texts, autotexts = ax2.pie(homeplanet_counts.values,
                                      labels=homeplanet_counts.index,
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      colors=colors[:len(homeplanet_counts)],
                                      explode=explode,
                                      shadow=True,
                                      textprops={'fontsize': 10, 'fontweight': 'bold'})

    ax2.set_title('Distribusi Home Planet\n(Pie Chart)',
                 fontsize=16, fontweight='bold', pad=20)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    for text in texts:
        text.set_fontsize(9)
        text.set_fontweight('bold')

    ax2.legend(wedges, [f'{occ}: {count}' for occ, count in homeplanet_counts.items()],
              title="Home Planet Detail",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=9)

    plt.tight_layout()
    
    homeplanet_path = os.path.join(eda_dir, "homeplanet_distributions.png")
    plt.savefig(homeplanet_path)
    plt.close()
    mlflow.log_artifact(homeplanet_path, "eda_plots")
    print("Distribusi HomePlanet dicatat.")

    print("EDA selesai dan semua plot telah dicatat di MLflow.")

def preprocess_data(input_path: Text = "data.csv", output_dir: Text = "output"):
    try:
        validate_inputs(input_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # experiment_name = "SpaceshipTitanic_Preprocessing"

    # try:
    #     experiment_id = mlflow.create_experiment(experiment_name)
    # except:
    #     experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    # mlflow.set_experiment(experiment_name)

    # with mlflow.start_run(run_name="Data Preprocessing and EDA"):
    os.makedirs(output_dir, exist_ok=True)

    mlflow.log_param("preprocessing_version", "2.0")

    # ======================================
    # 1. Memuat Dataset
    # ======================================
    print("\n=== TAHAP 1: MEMUAT DATASET ===")
    df_train = pd.read_csv(input_path)
    print(f"Dataset dimuat dengan {df_train.shape[0]} baris.")
    mlflow.log_param("initial_rows", df_train.shape[0])
    
    log_dataframe_summary(df_train, "before_processing")
    log_eda_metrics(df_train, "before_processing")

    # ======================================
    # 2. Exploratory Data Analysis (EDA)
    # ======================================
    print("\n=== TAHAP 2: EDA VISUAL ===")
    perform_eda_and_log(df_train)

    # ======================================
    # 3. Menangani Nilai yang Hilang
    # ======================================
    print("\n=== TAHAP 3: MENANGANI NILAI HILANG ===")
    df_train.dropna(subset=['PassengerId', 'Name', 'Transported'], inplace=True)
    
    numeric_cols_to_fill = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in numeric_cols_to_fill:
        df_train[col] = df_train[col].fillna(df_train[col].median())

    categorical_cols_missing = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
    for col in categorical_cols_missing:
        fill_value = df_train[col].mode()[0] if not df_train[col].mode().empty else "Unknown"
        df_train[col] = df_train[col].fillna(fill_value).astype("string")
    
    print("Nilai yang hilang telah ditangani.")
    mlflow.log_param("rows_after_na_drop", df_train.shape[0])

    # ======================================
    # 4. Rekayasa Fitur
    # ======================================
    print("\n=== TAHAP 4: REKAYASA FITUR ===")
    df_train['GroupId'] = df_train['PassengerId'].apply(lambda x: x.split('_')[0])
    df_train['GroupSize'] = df_train.groupby('GroupId')['PassengerId'].transform('count')
    df_train['SoloTraveler'] = (df_train['GroupSize'] == 1).astype(int)

    df_train[['Deck','CabinNum','Side']] = df_train['Cabin'].str.split('/', expand=True)
    df_train['CabinNum'] = pd.to_numeric(df_train['CabinNum'], errors='coerce').fillna(0)

    df_train['TotalSpend'] = df_train[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)
    df_train['NoSpend'] = (df_train['TotalSpend'] == 0).astype(int)

    df_train['AgeGroup'] = pd.cut(df_train['Age'], bins=[0, 12, 18, 25, 40, 60, 80],
                            labels=['Child','Teen','YoungAdult','Adult','MiddleAge','Senior'])
    
    print("Rekayasa fitur selesai.")

    # ======================================
    # 5. Normalisasi dan Encoding
    # ======================================
    print("\n=== TAHAP 5: NORMALISASI DAN ENCODING ===")
    df_train.drop(columns=["PassengerId", "GroupId", "Name", "Cabin"], inplace=True)
    
    numeric_cols_to_scale = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "NoSpend", "TotalSpend", "CabinNum", "SoloTraveler", "GroupSize"]
    scalers = {}
    for col in numeric_cols_to_scale:
        df_train[col] = np.log1p(df_train[col])
        scaler = RobustScaler()
        df_train[col] = scaler.fit_transform(df_train[[col]])
        scalers[col] = scaler

    categorical_cols_to_encode = [col for col in df_train.columns if df_train[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_train[col])]
    label_encoders = {}
    for col in categorical_cols_to_encode:
        encoder = LabelEncoder()
        df_train[col] = encoder.fit_transform(df_train[col].astype(str))
        label_encoders[col] = encoder
        
    print("Normalisasi dan encoding selesai.")
        
    log_dataframe_summary(df_train, "after_processing")
    log_eda_metrics(df_train, "after_processing")

    # ======================================
    # 6. Menyimpan Artefak
    # ======================================
    print("\n=== TAHAP 6: MENYIMPAN ARTEFAK ===")
    processed_data_path = os.path.join(output_dir, "spaceship_titanic_processed.csv")
    df_train.to_csv(processed_data_path, index=False)
    mlflow.log_artifact(processed_data_path, "processed_data")

    for col, scaler_obj in scalers.items():
        path = os.path.join(output_dir, f"{col}_scaler.pkl")
        joblib.dump(scaler_obj, path)
        mlflow.log_artifact(path, "models/scalers")

    for col, encoder_obj in label_encoders.items():
        path = os.path.join(output_dir, f"{col}_encoder.pkl")
        joblib.dump(encoder_obj, path)
        mlflow.log_artifact(path, "models/encoders")
    
    print(f"Artefak berhasil disimpan di direktori '{output_dir}' dan dicatat di MLflow.")

if __name__ == '__main__':
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    data_path = sys.argv[1]
    preprocess_data(input_path=data_path)