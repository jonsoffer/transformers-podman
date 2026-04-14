FROM docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN pip install transformers
RUN mkdir /app
COPY ./run.py /app

ENTRYPOINT ["/app/run.py"]
