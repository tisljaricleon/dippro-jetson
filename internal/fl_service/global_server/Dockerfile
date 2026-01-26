FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENBLAS_CORETYPE=ARMV8
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /home

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /home/

RUN pip install "numpy<2"

RUN pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN python -c "import torch, numpy as np; print(torch.__version__, np.__version__)"

RUN pip install --no-cache-dir -r requirements.txt

COPY global_server.py /home/

ENTRYPOINT ["python", "global_server.py"]
