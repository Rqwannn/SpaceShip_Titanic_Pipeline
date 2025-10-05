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

## Pertanyaan & Saran

Kurangnya flexibilitas menggunakan MLproject di Kriteria 1 dan 3

- Kenapa harus pakai MLproject?
- kenapa ga langsung manual custom agar lebih flexible
- seperti langsung execute python main.py
- Karena kalau pakai MLproject, bentrok terus sama Mlflow Dagshub
- Terus juga, saya rasa lebih enak jika buat Dockerfile langsung dari pada dari mlflow ribet
- beberapa error di CI/CD tapi bisa di local ( padahal disini udah saya otak atik kaya kasih tracking uri ke local dll)
- Permission Error ke /Users masa dia save ke root folder kalo CI/CD
- Mlflow internet error dagshub 500
- error bentrok Dagshub MLflow UI dengan mlflow.start_run() -> gabisa bersamaan di buat
- Jika pakai mlflow run . (MLproject) itu 50% parameter model saya ilang ga ke log
- tapi kalau pakai DVC atau execute python main.py itu 100% parameter ada, seperti estimator yang ada serta Tag

Saran

- Coba kalau custom Dockerfile lebih enak apalagi pake docker compose kan
- lebih enak untuk kerja sama tim jika pakai Dagshub MLflow UI? jika langsung execute script tanpa mlflow.run bisa

jujur ini ga tau script saya yang salah jika pakai MLproject atau tidak, tapi lebih mudah jika ga pakai menurut saya, mungkin saranya materinya bisa di sesuaikan kembali, pakai DVC lebih bagus jika tidak ada batasan stcruktur foldernya, terus CI/CD sangat bagus juga materinya, tapi saya rasa gasuka pakai MLproject karna kurangnya flexibilitas

berikut si yang saya harapkan

- CI/CD tanpa bergantung harus pakai MLproject
- bisa pakai Dockerfile / docker compose
- bisa menggunakan DVC
- Dagshub dengan MLflow UI lebih baik untuk kerja sama tim pada tahap experiment