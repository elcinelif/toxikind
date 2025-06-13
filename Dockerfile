FROM python:3.12.9-slim-bullseye
WORKDIR /prod
COPY toxikind toxikind
COPY api api
COPY production_model production_model
COPY data/X_test.csv data/X_test.csv
COPY requirements_api.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
