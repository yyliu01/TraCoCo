# FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libglib2.0-0 git  && apt-get clean

COPY requirements.txt requirements.txt
COPY Code Code
COPY scripts scripts

RUN pip install -r requirements.txt
RUN pip install --upgrade google-cloud-storage
