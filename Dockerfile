FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENBLAS_CORETYPE=ARMV8
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# ------------------------------
# System dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Python tooling
# ------------------------------
RUN pip install --upgrade pip setuptools wheel

# ------------------------------
# NumPy FIRST (critical)
# ------------------------------
RUN pip install "numpy<2"

# ------------------------------
# PyTorch (ARM-safe CPU wheels)
# ------------------------------
RUN pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Early sanity check (fast fail)
RUN python -c "import torch, numpy as np; print('Torch OK:', torch.__version__, 'NumPy OK:', np.__version__)"

# ------------------------------
# Core scientific stack
# ------------------------------
RUN pip install \
    "scipy<1.12" \
    pandas \
    "scikit-learn<1.4" \
    geopy \
    pyyaml

# ------------------------------
# torch-geometric (utils only)
# ------------------------------
RUN pip install torch-geometric==2.4.0 --no-deps

# ------------------------------
# PyG pure-Python runtime deps
# ------------------------------
RUN pip install \
    tqdm \
    networkx \
    jinja2 \
    sympy \
    psutil \
    requests \
    pyparsing \
    typing-extensions \
    aiohttp \
    fsspec


# ------------------------------
# Final PyG sanity check
# ------------------------------
RUN python -c "from torch_geometric.utils import k_hop_subgraph; import scipy; print('PyG utils OK')"

# ------------------------------
# Flower
# ------------------------------
RUN pip install flwr==1.7.0 flwr_datasets

WORKDIR /app
CMD ["python3"]
