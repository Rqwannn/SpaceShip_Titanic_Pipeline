## Menjalankan Server MLFlow

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 127.0.0.1 --port 5000
```

## DVC cmd

```bash
dvc init

dvc repro

dvc dag

dvc cache purge

dvc gc -w

```