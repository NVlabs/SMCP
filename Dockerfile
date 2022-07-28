FROM nvcr.io/nvidia/pytorch:21.10-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install sudo dialog apt-utils && rm -rf /var/lib/apt/lists/*
USER root

# Install some basic utilities
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    pigz \
    unzip \
    htop \
    wget \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

# Setup workspace
RUN mkdir -p /workspace/code
ENV HOME=/workspace/code
RUN chmod 777 /workspace/code

CMD ["python3"]
WORKDIR /workspace/code
USER root
