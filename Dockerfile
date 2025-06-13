FROM python:3.12.9-slim-bullseye
WORKDIR /prod
COPY toxikind toxikind
COPY production_model production_model
COPY api api
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn toxikind.api.fast:app --host 0.0.0.0
