FROM python:3.12.9-buster
COPY toxikind /toxikind
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn toxikind.api.fast:app --host 0.0.0.0
