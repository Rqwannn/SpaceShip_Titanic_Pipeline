# Build an image that can serve mlflow models.
FROM python:3.12.7-slim
 
RUN apt-get -y update && apt-get install -y --no-install-recommends nginx
 
WORKDIR /opt/mlflow
 
RUN pip install mlflow==2.22.2
 
COPY artifacts/local/ /opt/ml/model
RUN python -c "from mlflow.models import container as C; C._install_pyfunc_deps('/opt/ml/model', install_mlflow=False, enable_mlserver=False, env_manager='local');"
 
ENV MLFLOW_DISABLE_ENV_CREATION=True
ENV ENABLE_MLSERVER=False
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"
 
RUN chmod o+rwX /opt/mlflow/
 
RUN rm -rf /var/lib/apt/lists/*
 
ENTRYPOINT ["python", "-c", "from mlflow.models import container as C; C._serve('local')"]
